import argparse
import random
import time

import torch

import utils
from model import LHCDA,Cly_net,DomainDiscriminator,grad_reverse
import dgl
from utils import *
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
from sklearn.cluster import KMeans

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0, help="which GPU to use. Set -1 to use CPU.")
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr-ini', type=float, default=0.01, help='Initial learning rate.')  #0.015
parser.add_argument('--l2-w', type=float, default=0.01, help='weight of L2-norm regularization')
parser.add_argument("--num-heads", type=int, default=4, help="number of hidden attention heads")
parser.add_argument("--num-layers", type=int, default=2, help="number of hidden layers")   #2
parser.add_argument("--num-hidden", type=int, default=16, help="number of hidden units")
parser.add_argument("--intra_view_gcl_wei", type=float, default=1, help="weight of network-specific GCL")
parser.add_argument("--inter_view_gcl_wei", type=float, default=1, help="weight of cross-network GCL")
parser.add_argument("--inter_view_gcl_wei_nc", type=float, default=1, help="weight of cross-network GCL")
parser.add_argument("--inter_view_gcl_wei_cc", type=float, default=1, help="weight of cross-network GCL")
parser.add_argument("--in-drop", type=float, default=0.4, help="input feature dropout")
parser.add_argument("--attn-drop", type=float, default=0.1, help="attention dropout")
parser.add_argument("--tau", type=float, default=1, help="temperature-scales")
parser.add_argument('--grl-weight', type=int, default=1, help="--lr-weight")
parser.add_argument("--num-out-heads", type=int, default=2, help="number of output attention heads")
parser.add_argument("--batch_size", type=int, default=4000, help="batch_size for each domain")
parser.add_argument("--threashold", type=float, default=0.1, help="remove nodes whose similarity less than 0.3")
parser.add_argument('--data_src', type=str, default='dblpv7', help='source dataset name')
parser.add_argument('--data_trg', type=str, default='citationv1', help='target dataset name')
args = parser.parse_args()

source = args.data_src
target = args.data_trg
emb_filename = str(source) + '-' + str(target)
f = open('./' + emb_filename + '.txt', 'a')
f.write('{}\n'.format(args))
f.flush()
# Load source data
A_s, X_s, Y_s = load_network('./Datasets_changed/2/'+emb_filename+'/' + str(source) + '.mat')
print("Y_s.shape: ",Y_s.shape)
num_feat = X_s.shape[1]
num_class = Y_s.shape[1]
num_nodes_s = X_s.shape[0]
g_s = dgl.from_scipy(A_s)
g_s = dgl.remove_self_loop(g_s)
g_s = dgl.add_self_loop(g_s)
# Load target data
A_t, X_t, Y_t = load_network('./Datasets_changed/2/'+emb_filename+'/' + str(target) + '.mat')
num_nodes_t = X_t.shape[0]
g_t = dgl.from_scipy(A_t)
g_t = dgl.remove_self_loop(g_t)
g_t = dgl.add_self_loop(g_t)

X_s=X_s.astype(np.int16)
X_t=X_t.astype(np.int16)
# Y_t1=Y_t.copy()
# features_s = torch.Tensor(X_s)
# features_t = torch.Tensor(X_t)
features_s = torch.Tensor(X_s.todense())
features_t = torch.Tensor(X_t.todense())

numRandom = 1
microAllRandom = []
macroAllRandom = []
best_microAllRandom = []
best_macroAllRandom = []

unknownclass=2
atten_threshold=0.4
random_state = 0
epoch_thre=30
heads = ([args.num_heads] * args.num_layers)
mlp_input_dim = int((args.num_hidden) * (args.num_heads))
ST_max = max(X_s.shape[0], X_t.shape[0])
Y_s_tensor = torch.LongTensor(Y_s)

deletecol = list(range(Y_s.shape[1]+ 1, Y_t.shape[1], 1))
if Y_t.shape[1] > Y_s.shape[1] + 1:
    for i in range(Y_s.shape[1] + 1, Y_t.shape[1]):
        # print(i)
        Y_t[:, Y_s.shape[1]] = np.where(np.logical_or(Y_t[:, Y_s.shape[1]], Y_t[:, i]), 1, Y_t[:, Y_s.shape[1]])
    Y_t = np.delete(Y_t, deletecol, 1)
print("Y_t.shape: ",Y_t.shape)

while random_state < numRandom:
    random_state = random_state + 1
    print('%d-th random split' % (random_state))

    random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state) if torch.cuda.is_available() else None
    np.random.seed(random_state)

    # clf_type = 'multi-label'
    clf_type = 'multi-label'
    model = LHCDA(
        num_layers=args.num_layers,
        in_dim=num_feat,
        num_hidden=args.num_hidden,
        heads=heads,
        activation=F.elu,
        feat_drop=args.in_drop,
        attn_drop=args.attn_drop,
        negative_slope=0.2,
        residual=False,
        num_classes=num_class)

    domain_dis=DomainDiscriminator(mlp_input_dim)

    cly_model = Cly_net(mlp_input_dim, Y_s.shape[1]+unknownclass,multi_class=False)

    clf_loss_f = nn.BCEWithLogitsLoss(reduction='none') if clf_type == 'multi-label' \
        else nn.CrossEntropyLoss(reduction='none')

    # criterion_instance = InstanceLoss(args.batch_size, 0.1)


    # domain_loss_f = nn.CrossEntropyLoss()

    t_total = time.time()
    domain_loss_all = []
    total_loss_all = []

    best_epoch = 0
    best_micro_f1 = 0
    best_macro_f1 = 0
    best_known_micro_f1 = 0
    best_known_macro_f1 = 0
    best_known_acc = 0
    best_class_average_acc = 0

    best_open_acc = 0
    best_hos=0

    pred_Y_t = np.zeros(Y_t.shape)

    for epoch in range(args.epochs):
        t = time.time()

        for batch_idx, (batch_s, batch_t) in enumerate(
                zip(mini_batch(X_s, Y_s, A_s, ST_max, args.batch_size),
                    mini_batch(X_t, pred_Y_t, A_t, ST_max, args.batch_size))):

            feat_s, label_s, adj_s, shuffle_index_s = batch_s
            feat_t, pred_label_t, adj_t_o, shuffle_index_t = batch_t


            feat_s, label_s, adj_s = (torch.FloatTensor(feat_s.toarray()),
                                      torch.LongTensor(label_s),
                                      torch.FloatTensor(adj_s.toarray())
                                      )
            feat_t, pred_label_t, adj_t = (torch.FloatTensor(feat_t.toarray()),
                                           torch.FloatTensor(pred_label_t),
                                           torch.FloatTensor(adj_t_o.toarray())
                                           )
            before_delete = torch.sum(adj_t)
            # print(before_delete)
            g_s1 = dgl.from_scipy(sp.coo_matrix(adj_s))
            g_s1 = dgl.remove_self_loop(g_s1)
            g_s1 = dgl.add_self_loop(g_s1)
            g_t1 = dgl.from_scipy(sp.coo_matrix(adj_t))
            g_t1 = dgl.remove_self_loop(g_t1)
            g_t1 = dgl.add_self_loop(g_t1)

            src, dst = g_t1.edges()

            # 打印出所有边的信息
            # for i in range(len(src)):
            #     print(f'边{i}: 由节点{src[i].item()}到节点{dst[i].item()}')

            p = float(epoch) / args.epochs
            lr = args.lr_ini / (1. + 10 * p) ** 0.75
            grl_lambda = 2. / (1. + np.exp(-10. * p)) - 1  # gradually change from 0 to 1

            model.train()
            cly_model.train()
            domain_dis.train()
            total_parameters = list(model.parameters()) + list(cly_model.parameters()) + list(domain_dis.parameters())
            optimizer = torch.optim.Adam(total_parameters, lr, weight_decay=args.l2_w)
            optimizer.zero_grad()

            mean_attention_t, emb_s, emb_t, head_s, head_t = model(feat_s, feat_t, g_s1, g_t1)

            # deleted_edges_index = mean_attention_t < 0.05
            mean_attention_t=torch.reshape(mean_attention_t,(mean_attention_t.shape[0],))


            _, deleted_edges_index = torch.topk(mean_attention_t, (int)(mean_attention_t.shape[0]*0.05), largest=False) #0.05

            # 然后，使用nonzero()函数找出所有为True的索引
            # deleted_edges_index = torch.nonzero(deleted_edges_index, as_tuple=True)[0]
            adj_t1=torch.FloatTensor(adj_t_o.toarray())
            for i in range(deleted_edges_index.shape[0]):
                adj_t1[src[deleted_edges_index[i]],dst[deleted_edges_index[i]]]=0

                # # 检验删除的边什么情况
                # kno_unk_count = 0
                # differentknoclass = 0
                # differentunkclass = 0
                # different = 0
                # same = 0
                # aaa += 1
                # Y_t2 = Y_t1[shuffle_index_t]
                # for i in range(deleted_edges_index.shape[0]):
                #
                #     start = src[deleted_edges_index[i]]
                #     end = dst[deleted_edges_index[i]]
                #     label1 = np.where(Y_t2[start] == 1)[0][0]
                #     label2 = np.where(Y_t2[end] == 1)[0][0]
                #     if label1 == label2:
                #         same += 1
                #     if label1 != label2:
                #         different += 1
                #     if (label1 < label_s.shape[1] and label2 >= label_s.shape[1]) or label1 >= label_s.shape[
                #         1] and label2 < label_s.shape[1]:
                #         kno_unk_count += 1
                #     if label1 < label_s.shape[1] and label2 < label_s.shape[1] and label1 != label2:
                #         differentknoclass += 1
                #     if label1 >= label_s.shape[1] and label2 >= label_s.shape[1] and label1 != label2:
                #         differentunkclass += 1
                # rat = (kno_unk_count + differentknoclass + differentunkclass) / deleted_edges_index.shape[0]
                # bbb += rat
                # print("total delete: ", deleted_edges_index.shape[0])
                # print("kno_unk_count: ", kno_unk_count)
                # print("differentknownclass: ", differentknoclass)
                # print("differentunknownclass: ", differentunkclass)
                # print("different: ", different)
                # print("same: ", same)

            adj_t1 = adj_t1.bool() & adj_t1.t().bool()

            # 确保对称性
            adj_t1 = adj_t1 | adj_t1.t()

            # 将结果转换回整数类型，其中True为1，False为0
            adj_t1 = adj_t1.float()
            before_delete = torch.sum(adj_t)
            after_delete=torch.sum(adj_t1)
            print(before_delete)
            print(after_delete)
            g_t2 = dgl.from_scipy(sp.coo_matrix(adj_t1))
            g_t2 = dgl.remove_self_loop(g_t2)
            g_t2 = dgl.add_self_loop(g_t2)
            _, emb_s1, emb_t1, head_s1, head_t1 = model(feat_s, feat_t, g_s1, g_t2)

            node_contrastive_loss=InstanceLoss(args.batch_size,0.1,emb_t,emb_t1 )
            print("node_contrastive_loss: ",node_contrastive_loss.item())

            pred_logit_s=cly_model(emb_s)
            # pred_logit_s=pred_logit_s[:,:label_s.shape[1]]

            pred_logit_t = cly_model(emb_t)
            # pred_logit_s = F.sigmoid(pred_logit_s) if clf_type == 'multi-label' else F.softmax(pred_logit_s)
            # pred_logit_t = F.sigmoid(pred_logit_t) if clf_type == 'multi-label' else F.softmax(pred_logit_t)



            entropy = -torch.sum(pred_logit_t * torch.log(pred_logit_t.clamp(min=1e-9)), dim=1)

            pseudo_unknown_num = int(entropy.shape[0] * 0.15)

            # 使用np.argsort()对元素进行排序，[-num_elements:]选择最大的30%
            _,pseudo_unknown = torch.topk(entropy,pseudo_unknown_num)
            # remaining_indices = torch.ones(entropy.shape[0]).byte()
            # remaining_indices[pseudo_unknown] = 0
            # pseudo_known = torch.nonzero(remaining_indices).squeeze()
            _,pseudo_known = torch.topk(entropy, (int)(entropy.shape[0]*0.5), largest=False)

            pseudo_known_emb_t=emb_t[pseudo_known]

            pred_logit_t_unknown = pred_logit_t[pseudo_unknown]

            # 利用增强样本获取未知类训练的监督标签
            pred_logit_t1=cly_model(emb_t1)
            # pred_logit_t1 = F.sigmoid(pred_logit_t1) if clf_type == 'multi-label' else F.softmax(pred_logit_t1)


            pred_logit_t1[pseudo_unknown, :label_s.shape[1]] = 0

            # 在剩下的列中选择最大的数置为1，其他的置为0
            max_vals_unknown, _ = torch.max(pred_logit_t1[pseudo_unknown, label_s.shape[1]:], dim=1, keepdim=True)
            pred_logit_t1[pseudo_unknown, label_s.shape[1]:] = (
                        pred_logit_t1[pseudo_unknown, label_s.shape[1]:] == max_vals_unknown).float()

            pred_logit_t1[pseudo_known, label_s.shape[1]:] = 0

            # 在剩下的列中选择最大的数置为1，其他的置为0
            max_vals_known, _ = torch.max(pred_logit_t1[pseudo_known, :label_s.shape[1]], dim=1, keepdim=True)
            pred_logit_t1[pseudo_known, :label_s.shape[1]] = (
                    pred_logit_t1[pseudo_known, :label_s.shape[1]] == max_vals_known).float()
            pred_logit_t1=pred_logit_t1.int()
            # print(torch.sum(pred_logit_t1,dim=0))
            pred_logit_t1_unknown=pred_logit_t1[pseudo_unknown]
            pred_logit_t1_known=pred_logit_t1[pseudo_known]




            zeros = torch.zeros(label_s.shape[0], unknownclass)
            aug_label_s=torch.cat((label_s, zeros), dim=1)
            combined_tensor = torch.cat((pred_logit_s, pred_logit_t_unknown), dim=0)
            combined_tensor_label = torch.cat((aug_label_s, pred_logit_t1_unknown), dim=0)

            if clf_type == 'multi-class':

                # clf_loss = clf_loss_f(combined_tensor, torch.argmax((combined_tensor_label), 1))
                # clf_loss = clf_loss / ((label_s).shape[0] + pred_logit_t1_unknown.shape[0])
                # clf_loss1 = clf_loss_f(pred_logit_s, torch.argmax((aug_label_s), 1))
                # print("clf_loss1: ", clf_loss1.item())
                # if epoch>=epoch_thre:
                #     pred = torch.cat((pred_logit_s, pred_logit_t_unknown), 0)
                #     label = torch.cat((aug_label_s, pred_logit_t1_unknown), 0)
                #     clf_loss2 = clf_loss_f(pred, torch.argmax((label), 1))
                #     print("clf_loss2: ",clf_loss2.item())


                if epoch<epoch_thre:
                    clf_loss = torch.sum(clf_loss_f(pred_logit_s, (aug_label_s).float()))/label_s.shape[0]
                    print("clf_loss1: ",clf_loss.item())
                else:
                    pred=torch.cat((pred_logit_s, pred_logit_t_unknown), 0)
                    label=torch.cat((aug_label_s, pred_logit_t1_unknown), 0)
                    clf_loss = torch.sum(clf_loss_f(pred, (label).float()))/label.shape[0]
                    print("clf_loss2: ",clf_loss.item())


            else:
                if epoch<epoch_thre:
                    clf_loss = torch.sum(clf_loss_f(pred_logit_s, (aug_label_s).float()))/label_s.shape[0]
                    print("clf_loss1: ",clf_loss.item())
                else:
                    pred=torch.cat((pred_logit_s, pred_logit_t_unknown), 0)
                    label=torch.cat((aug_label_s, pred_logit_t1_unknown), 0)
                    clf_loss = torch.sum(clf_loss_f(pred, (label).float()))/label.shape[0]
                    print("clf_loss2: ",clf_loss.item())



            # print("clf_loss: ",clf_loss)

            # k = 10  # 选择k值
            # emb_t_np =  emb_t.detach().numpy()
            # emb_s_np=emb_s.detach().numpy()
            # nbrs = NearestNeighbors(n_neighbors=k).fit(emb_t_np)
            #
            # distances, indices = nbrs.kneighbors(emb_s_np)
            # adjacency_matrix = np.zeros((emb_s.shape[0], emb_t.shape[0]))
            # for i in range(len(emb_s)):
            #     for j in indices[i]:
            #         # if j in pseudo_known:  # 检查是否在记忆库中
            #         adjacency_matrix[i, j] = 1
            #
            # adjacency_tensor=torch.tensor(adjacency_matrix)
            # mixup_emb=utils.fuse_embeddings(emb_s,emb_t,adjacency_tensor)
            #
            # # emb_s_guiyi=F.softmax(emb_s,dim=-1)
            # # emb_mixup_guiyi = F.softmax(mixup_emb,dim=-1)
            # # kl_loss=F.kl_div(emb_s_guiyi.log(),emb_mixup_guiyi,reduction='batchmean')
            # # print("domain_loss: ", kl_loss.item())
            #
            # mse_loss = nn.MSELoss()(emb_s, mixup_emb)
            # print("domain_loss: ", mse_loss.item())

            #领域适应/现在的
            k=20
            pseudo_known_emb_t_np = pseudo_known_emb_t.detach().numpy()
            emb_s_np=emb_s.detach().numpy()
            cosine_sim = cosine_similarity(emb_s_np, pseudo_known_emb_t_np)
            # 获取每一行相似度最高的前k个索引
            top_k_indices = np.argsort(-cosine_sim, axis=1)[:, :k]
            neighbor_count = np.zeros(pseudo_known_emb_t.shape[0])
            for indice in top_k_indices:
                for index in indice:
                    neighbor_count[index]+=1
            neighbor_count_tensor=torch.tensor(neighbor_count)
            # neighbor_weight=neighbor_count_tensor/neighbor_count_tensor.sum()
            neighbor_weight=F.sigmoid(neighbor_count_tensor)
            domain_weight=torch.cat((torch.ones(emb_s.shape[0]),neighbor_weight),0)
            domain_loss_f = nn.CrossEntropyLoss(reduction='none')
            domain_label = np.vstack([np.tile([1., 0.], [feat_s.shape[0], 1]), np.tile([0., 1.], [pseudo_known_emb_t.shape[0], 1])])
            emb = torch.cat((emb_s, pseudo_known_emb_t), 0)

            # Domain_Discriminator
            h_grl = grad_reverse(emb, grl_lambda)
            d_logit = domain_dis(h_grl)
            domain_loss = domain_loss_f(d_logit, torch.argmax(torch.FloatTensor(domain_label), 1))
            domain_loss=domain_loss*domain_weight
            domain_loss=domain_loss.mean()
            print("domain_loss: ",domain_loss.item())

            if epoch>=epoch_thre:
                pred_logit_t1_known_label=pred_logit_t1_known[:,:label_s.shape[1]]
                emb_t_known_label = emb_t[pseudo_known, :]
                class_counts_t = pred_logit_t1_known_label.sum(dim=0)
                class_sums_t = torch.matmul(pred_logit_t1_known_label.T.float(), emb_t_known_label)
                class_averages_known = class_sums_t / class_counts_t.unsqueeze(1)
                class_counts_s = label_s.sum(dim=0)
                class_sums_s = torch.matmul(label_s.T.float(), emb_s)
                class_averages_s = class_sums_s / class_counts_s.unsqueeze(1)

                # unknown_center
                pred_logit_t1_only_unknown_label = pred_logit_t1_unknown[:, label_s.shape[1]:]
                emb_t_only_unknown_label = emb_t[pseudo_unknown, :]

                havelabel = torch.nonzero(pred_logit_t1_only_unknown_label.sum(dim=0) > 0)[:, 0]
                # if nolabel.numel()!=0:
                pred_logit_t1_only_unknown_label = torch.index_select(pred_logit_t1_only_unknown_label, dim=1, index=havelabel)

                class_counts = pred_logit_t1_only_unknown_label.sum(dim=0)

                # 使用标签的转置乘以样本张量来获取每个类的总张量
                class_sums = torch.matmul(pred_logit_t1_only_unknown_label.T.float(), emb_t_only_unknown_label)

                # 计算每个类的平均张量
                class_averages_unknown = class_sums / class_counts.unsqueeze(1)

                class_averages_t=torch.cat((class_averages_known,class_averages_unknown),dim=0)

                # criterion_center = CenterLoss(Y_s.shape[1], Y_s.shape[1] + class_counts.shape[0], 1.0)
                center_contrastive_loss=CenterLoss(Y_s.shape[1], Y_s.shape[1] + class_counts.shape[0], 1.0,class_averages_s,class_averages_t)
                print("center_contrastive_loss: ",center_contrastive_loss.item())



            if epoch<epoch_thre:
                total_loss = domain_loss + clf_loss + node_contrastive_loss  #
            else:
                total_loss = domain_loss + clf_loss + node_contrastive_loss + center_contrastive_loss  #
            # optimizer.zero_grad()
            total_loss.backward()
            total_loss_all.append(total_loss.item())
            optimizer.step()

        '''Compute evaluation on test data by the end of each epoch'''
        model.eval()  # deactivates dropout during validation run.
        cly_model.eval()
        with torch.no_grad():
            _, emb_s, emb_t, _, _ = model(features_s, features_t, g_s, g_t)

            # _, indices = torch.max(pred_logit_t, dim=1)
            # pred_label_clf = one_hot_encode_torch(indices, pred_logit_t.shape[1])
            #
            # if clf_type == 'multi-class':
            #     clf_loss = clf_loss_f(pred_logit_s, torch.argmax((Y_s_tensor), 1))
            # else:
            #     clf_loss = clf_loss_f(pred_logit_s, (Y_s_tensor).float())
            #     clf_loss = torch.sum(clf_loss) / (Y_s_tensor).shape[0]
            # domain_label = np.vstack(
            #     [np.tile([1., 0.], [features_s.shape[0], 1]), np.tile([0., 1.], [features_t.shape[0], 1])])
            # domain_loss = domain_loss_f(d_logit, torch.argmax(torch.FloatTensor(domain_label), 1))
            # domain_loss_all.append(domain_loss.item())
            pred_s=cly_model(emb_s)
            pred_t = cly_model(emb_t)
            pred_prob_xs = F.sigmoid(pred_s) if clf_type == 'multi-label' else F.softmax(pred_s)
            pred_prob_xt = F.sigmoid(pred_t) if clf_type == 'multi-label' else F.softmax(pred_t)
            selected_columns = pred_prob_xt[:, Y_s.shape[1]:]

            # 计算这三列中的最大值
            max_values = torch.max(selected_columns, dim=1).values

            # 将最大值赋给第5列
            pred_prob_xt[:, Y_s.shape[1]] = max_values
            pred_prob_xt=pred_prob_xt[:,:Y_s.shape[1]+1]
            # f1_s = f1_scores(pred_prob_xs, Y_s)
            # print('epoch %d: Source micro-F1: %f, macro-F1: %f' % (epoch, f1_s[0], f1_s[1]))
            f1_t = f1_scores(pred_prob_xt, Y_t)
            print('epoch %d: Target testing micro-F1: %f, macro-F1: %f' % (epoch, f1_t[0], f1_t[1]))
            prediction = predictgetone(Y_t, pred_prob_xt)

            known_index = np.where(np.sum(Y_t[:, :Y_t.shape[1] - 1], axis=1) > 0)[0]
            # 找出求和结果大于0的行的索引
            known_prediction = prediction[known_index]
            known_prediction = known_prediction[:, :Y_t.shape[1] - 1]
            # print(known_prediction.shape)
            known_label = Y_t[known_index]
            # print(known_label.shape)
            known_label = known_label[:, :Y_t.shape[1] - 1]
            # print(known_label.shape)
            known_f1 = f1_known_score(known_label, known_prediction)
            # print('Target known-class  micro-F1: %f   macro-F1: %f' % (known_f1[0], known_f1[1]))
            known_pred_num = 0
            for i in range(known_index.shape[0]):
                true_label = np.where(known_label[i] == 1)[0]
                pred_label = np.where(known_prediction[i] == 1)[0]
                for j in range(pred_label.shape[0]):
                    if pred_label[j] in true_label:
                        known_pred_num += 1
            known_true_num = np.sum(known_label == 1)
            known_true_rate = known_pred_num / known_true_num
            # print('known sample acc: %.4f' % (known_true_rate))
            print('Target known-class  micro-F1: %.4f   macro-F1: %.4f   acc: %.4f' % (
            known_f1[0], known_f1[1], known_true_rate))

            class_average_acc_all = 0
            for i in range(Y_t.shape[1] - 1):
                everyclass_true_index = np.where(known_label[:, i] == 1)[0]
                # print(everyclass_true_index.shape[0])
                everyclass_pred = known_prediction[everyclass_true_index, i]
                everyclass_average_acc = np.sum(everyclass_pred) / np.sum(known_label[:, i] == 1)
                class_average_acc_all += everyclass_average_acc
            class_average_acc = class_average_acc_all / (Y_t.shape[1] - 1)

            print('OS*: %.4f' % (class_average_acc))

            # if known_true_rate>best_known_acc:
            #     best_known_acc=known_true_rate

            open_index = np.where(Y_t[:, Y_t.shape[1] - 1] == 1)  # 得到未知类节点的索引
            open_prediction = prediction[open_index[0]]  # 得到未知类的概率，已经转成1了
            pred_open_num = np.sum(open_prediction[:, Y_t.shape[1] - 1])
            open_true_rate = pred_open_num / open_index[0].shape[0]

            print('Unk: %.4f' % (open_true_rate))

            HOS=2*class_average_acc*open_true_rate/(class_average_acc+open_true_rate)
            print('HOS: %.4f' % (HOS))
            # print('domain_loss: %f, clf_loss: %f' % (domain_loss, clf_loss))

            # ###generate pseduo-label for target nodes by k-means
            # norm_Y_s = my_scale_sim_mat(Y_s.T)
            # norm_Y_s = torch.FloatTensor(norm_Y_s)
            # emb_s_class = torch.mm(norm_Y_s, emb_s)  ###c*d, each row represents avg rep for one class
            # kmeans = KMeans(n_clusters=num_class, init=emb_s_class, random_state=0).fit(emb_s)
            # pred_Y_t = kmeans.predict(emb_t)
            # pred_Y_t = np.eye(num_class)[pred_Y_t]
            # pred_label = calculate_pred_label_t(pred_label_clf, pred_Y_t).float()
            # # # ###remove noisy nodes, far away from the cluster centroid
            # X_dist = kmeans.transform(emb_t) ** 2
            # sim = np.exp(-X_dist)  ###convert distance to similarity
            # pred_Y_t_noisy = np.where(sim > args.threashold, 1, 0)  ## remove nodes whose similarity less than 0.3
            # pred_Y_t = np.multiply(pred_Y_t, pred_Y_t_noisy)


            # if f1_t[1] > best_macro_f1:
            #     best_micro_f1 = f1_t[0]
            #     best_macro_f1 = f1_t[1]
            #     best_epoch = epoch
            #     print("saving model")
            if HOS>best_hos:
                print("saving model...")
                best_micro_f1 = f1_t[0]
                best_macro_f1 = f1_t[1]
                best_known_acc = known_true_rate
                best_class_average_acc = class_average_acc
                best_known_micro_f1 = known_f1[0]
                best_known_macro_f1 = known_f1[1]
                best_open_acc = open_true_rate
                best_hos=HOS
                best_epoch = epoch
                scipy.io.savemat('./' + emb_filename + '_emb.mat',
                                                  {'rep_S': emb_s.numpy(), 'rep_T': emb_t.numpy(), 'label_S': Y_s,
                                                   'label_T': Y_t})

    # print('Target best epoch %d, micro-F1: %f, macro-F1: %f' % (best_epoch, best_micro_f1, best_macro_f1))

    # microAllRandom.append(float(f1_t[0]))
    # macroAllRandom.append(float(f1_t[1]))
    # best_microAllRandom.append(float(best_micro_f1))
    # best_macroAllRandom.append(float(best_macro_f1))

    '''avg best F1 scores over 5 random splits'''
    # best_micro = np.mean(best_microAllRandom)
    # best_macro = np.mean(best_macroAllRandom)
    # best_micro_sd = np.std(best_microAllRandom)
    # best_macro_sd = np.std(best_macroAllRandom)
    # print(
    #     "The avergae best micro and macro F1 scores over {} random initializations are:  {} +/- {} and {} +/- {}: ".format(
    #         numRandom, best_micro, best_micro_sd, best_macro, best_macro_sd))
    print('Target best epoch %d' % (best_epoch))
    print("all_class    micro-F1 {:.4f}     macro-F1 {:.4f}".format(best_micro_f1,best_macro_f1))
    print("known_class  micro-F1 {:.4f}     known_macro-F1 {:.4f}".format(best_known_micro_f1,best_known_macro_f1))
    print("known_class  acc {:.4f}".format(best_known_acc))
    print("OS* {:.4f}".format(best_class_average_acc))
    print("Unk {:.4f}".format(best_open_acc))
    print("HOS {:.4f}".format(best_hos))
    # print("去除的边不同类的占比：", bbb / aaa)
# f.write(
#     "The avergae best micro and macro F1 scores over {} random initializations are:  {} +/- {} and {} +/- {}: \n".format(
#         numRandom, best_micro, best_micro_sd, best_macro, best_macro_sd))

f.flush()
f.close()
