import numpy as np
import scipy.io as sio
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
import scipy
import torch
from sklearn.metrics import f1_score
import scipy.sparse as sp
import torch.nn.functional as F
from sklearn.decomposition import PCA
from warnings import filterwarnings
from sklearn.metrics.pairwise import cosine_similarity

import torch.nn as nn

filterwarnings('ignore')


def load_network(file):
    net = sio.loadmat(file)
    x, a, y = net['attrb'], net['network'], net['group']
    if not isinstance(x, scipy.sparse.lil_matrix):
        x = lil_matrix(x)
    return a, x, y


def my_scale_sim_mat(w):
    """L1 row norm of a matrix"""
    rowsum = np.array(np.sum(w, axis=1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    w = r_mat_inv.dot(w)
    return w


def my_scale_sim_mat_torch(w):
    """L1 row norm of a matrix of torch version"""
    r = 1 / w.sum(1)
    r[r.isinf()] = 0
    return r.diag() @ w

def calculate_pred_label_t(pred_label, cluster_label):
    """calculate predicted label by comparing clf loss with pred label kmean"""
    # _, indices = torch.max(pred_logit_s, dim=1)
    # pred_logit_s = one_hot_encode_torch(indices, pred_logit_t.shape[1])
    # pred_logit_s = pred_logit_s.to(pred_logit_t.device)
    return pred_label * cluster_label
def calculate_centroid_2(label, emb):
    """calculate centroid of each class"""
    norm_Y = my_scale_sim_mat_torch(label.T)
    centroid = torch.mm(norm_Y, emb)
    return centroid
def one_hot_encode_torch(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    x = x.type(torch.LongTensor)
    return torch.eye(n_classes)[x]


def mini_batch(x, y, a, n, batch_size):
    idx = list(range(x.shape[0]))
    np.random.shuffle(idx)
    n = np.ceil(n / batch_size).astype('int') * batch_size
    idx = (idx * (n // x.shape[0] + bool(n % x.shape[0])))[:n]
    x, y = x[idx], y[idx]
    a = a[idx][:, idx]

    for i in range(n // batch_size + bool(n % batch_size)):
        start, end = i * batch_size, (i + 1) * batch_size
        shuffle_index = idx[i * batch_size:(i + 1) * batch_size]
        yield x[start:end], y[start:end], a[start:end, start:end], shuffle_index


def f1_scores(y_pred, y_true):
    def predict(y_tru, y_pre):
        top_k_list = np.array(np.sum(y_tru, 1), np.int32)
        prediction = []
        for i in range(y_tru.shape[0]):
            pred_i = np.zeros(y_tru.shape[1])
            pred_i[np.argsort(y_pre[i, :])[-top_k_list[i]:]] = 1
            prediction.append(np.reshape(pred_i, (1, -1)))
        prediction = np.concatenate(prediction, axis=0)
        return np.array(prediction, np.int32)

    results = {}
    predictions = predict(y_true, y_pred)
    averages = ["micro", "macro"]
    for average in averages:
        results[average] = f1_score(y_true, predictions, average=average)
    return results["micro"], results["macro"]


def sim(z1: torch.Tensor, z2: torch.Tensor, hidden_norm: bool = True):
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def predictgetone(y_true, y_pred):
    top_k_list = np.array(np.sum(y_true, 1), np.int32)
    predictions = []
    for i in range(y_true.shape[0]):
        pred_i = np.zeros(y_true.shape[1])
        pred_i[np.argsort(y_pred[i, :])[-top_k_list[i]:]] = 1
        predictions.append(np.reshape(pred_i, (1, -1)))
    predictions = np.concatenate(predictions, axis=0)

    return np.array(predictions, np.int32)

def f1_known_score(y_true,predictions):
    results = {}

    averages = ["micro", "macro"]
    for average in averages:
        results[average] = f1_score(y_true, predictions, average=average)

    return results["micro"], results["macro"]

#  点 - 点
def inter_view_nei_loss(z1: torch.Tensor, z2: torch.Tensor, tau, adj, hidden_norm: bool = True):
    adj[adj > 0] = 1

    f = lambda x: torch.exp(x / tau)
    between_sim = f(sim(z1, z2, hidden_norm))

    nei_count = torch.sum(adj, 1) + 1
    nei_count = torch.squeeze(torch.tensor(nei_count))

    loss = (between_sim.mul(adj)).sum(1) / between_sim.sum(1)
    loss = loss / nei_count
    loss[loss == 0] = 1

    # return -torch.log(loss + 1e-10)
    return -torch.log(loss)


def nei_dis_loss1(z1: torch.Tensor, z2: torch.Tensor, tau, adj, hidden_norm: bool = True):
    '''neighbor discrimination contrastive loss'''
    ###先求和再log
    # np.fill_diagonal(adj, 0) #remove self-loop
    adj = adj - torch.diag_embed(adj.diag())  # remove self-loop
    adj[adj > 0] = 1
    # nei_count=np.sum(adj,1)*2+1 ###intra-view nei+inter-view nei+self inter-view
    nei_count = torch.sum(adj, 1) * 2 + 1  ###intra-view nei+inter-view nei+self inter-view
    nei_count = torch.squeeze(torch.tensor(nei_count))
    # adj = torch.tensor(adj)

    f = lambda x: torch.exp(x / tau)
    refl_sim = f(sim(z1, z1, hidden_norm))
    between_sim = f(sim(z1, z2, hidden_norm))

    loss = (between_sim.diag() + (refl_sim.mul(adj)).sum(1) + (between_sim.mul(adj)).sum(1)) / (
            refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
    loss = loss / nei_count  ###divided by the number of positive pairs for each node

    return -torch.log(loss)


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, adj,
                     mean: bool = True, tau: float = 1.0, hidden_norm: bool = True):
    h1 = z1
    h2 = z2

    l1 = nei_dis_loss1(h1, h2, tau, adj, hidden_norm)
    l2 = nei_dis_loss1(h2, h1, tau, adj, hidden_norm)

    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()
    return ret


def multihead_contrastive_loss(heads, adj, tau: float = 1.0):
    ###算每个head到第一个head
    loss = torch.tensor(0, dtype=float, requires_grad=True)
    for i in range(1, len(heads)):
        loss = loss + contrastive_loss(heads[0], heads[i], adj, tau=tau)
    return loss / (len(heads) - 1)


def class_class_inter_view_nei_loss_2(z1: torch.Tensor, z2: torch.Tensor, tau, O_inter, O_intra,
                                      hidden_norm: bool = True):
    """class-class inter view neighbor contrastive loss"""
    O_intra = O_intra - (O_intra.diag().diag())

    f = lambda x: torch.exp(x / tau)

    inter_sim_st = f(sim(z1, z2, hidden_norm))
    intra_sim_ss = f(sim(z1, z1, hidden_norm))

    # fenzi
    molecule = inter_sim_st.diag() * O_inter.diag()
    # fenmu
    denominator = (inter_sim_st * O_inter).sum(1) + (intra_sim_ss * O_intra).sum(1)
    # 分母为0，会导致nan
    denominator[denominator == 0] = 1
    loss = molecule / denominator
    loss[loss == 0] = 1
    # loss[loss.isnan()] = 1
    return -torch.log(loss)

#  点 类
def inter_view_nei_loss_NC(z1: torch.Tensor, z2: torch.Tensor, tau, label, hidden_norm: bool = True):
    """node-class inter view neighbor contrastive loss"""

    f = lambda x: torch.exp(x / tau)
    inter_sim_st = f(sim(z1, z2, hidden_norm))
    # 分子 正样本
    molecule = inter_sim_st * label
    molecule = molecule.sum(1)
    # 分母
    denominator = inter_sim_st.sum(1)
    # 分母为0，会导致nan
    denominator[denominator == 0] = 1
    loss = molecule / denominator
    loss[loss == 0] = 1
    # loss[loss.isnan()] = 1
    return -torch.log(loss)


def InstanceLoss(batch_size,temperature,z_i, z_j, k=0.001):
    N = 2 * batch_size
    mask = torch.ones((N, N))
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    mask = mask.bool()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    z = torch.cat((z_i, z_j), dim=0)

    sim = torch.matmul(z, z.T) /temperature
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    negative_samples = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    loss = criterion(logits, labels)
    loss /= N
    loss *= k

    return loss

def compute_cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度
    """
    dot_product = torch.dot(vec1, vec2)
    norm_vec1 = torch.norm(vec1)
    norm_vec2 = torch.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def transform_tensor(tensor):
    # 提取正数
    positive_values = tensor[tensor != 0]
    # print(positive_values.size())

    # 找到后5%的阈值
    k = int(0.1 * len(positive_values))
    threshold = torch.topk(positive_values, k,largest=False).values[-1]

    # 将正数按条件转换
    tensor[tensor > threshold] = 1
    tensor[tensor <= threshold] = 0

    # print(tensor.sum())





    # non_zero_mask = tensor != 0
    # non_zero_values = tensor[non_zero_mask]
    #
    # # 计算非零值的数量并确定后5%的索引
    # num_non_zero = non_zero_values.size(0)
    # k = max(1, int(num_non_zero * 0.05))
    #
    # # 找到后5%的非零值
    # topk_values, topk_indices = torch.topk(non_zero_values, k, largest=False)
    #
    # # 创建一个新的张量进行修改
    # new_tensor = tensor.clone()
    #
    # # 将后5%的非零值变成0
    # new_tensor[non_zero_mask][topk_indices] = 0
    # print(new_tensor.sum())
    #
    # # 将其余的非零值变成1
    # new_tensor[new_tensor!=0]=1
    # # new_tensor[non_zero_mask] = torch.where(new_tensor[non_zero_mask] != 0, torch.tensor(1), new_tensor[non_zero_mask])
    # print(new_tensor.sum())
    return tensor


def cosine_similarity1(A, B):
    # 计算 L2 范数
    A_normalized = F.normalize(A, p=2, dim=1)
    B_normalized = F.normalize(B, p=2, dim=1)

    # 计算余弦相似度
    cosine_sim = torch.matmul(A_normalized, B_normalized.t())
    return cosine_sim

def fuse_embeddings(emb_s, emb_t, adj_matrix):
    """
    计算源网络中所有节点在目标网络中的邻居融合后的向量表示
    :param emb_s: 源网络节点的二维张量表示 (num_nodes_s, embedding_dim)
    :param emb_t: 目标网络节点的二维张量表示 (num_nodes_t, embedding_dim)
    :param adj_matrix: 邻接矩阵 (num_nodes_s, num_nodes_t)
    :return: 融合后的向量表示 (num_nodes_s, embedding_dim)
    """
    num_nodes_s, embedding_dim = emb_s.shape
    fused_embeddings = torch.zeros((num_nodes_s, embedding_dim))

    for i in range(num_nodes_s):
        neighbors = torch.where(adj_matrix[i] > 0)[0]
        if len(neighbors) == 0:
            fused_embeddings[i] = emb_s[i]
        else:
            weights = torch.tensor([compute_cosine_similarity(emb_s[i], emb_t[j]) for j in neighbors])
            weights /= weights.sum()  # 归一化权重
            fused_embeddings[i] = torch.sum(weights[:, None] * emb_t[neighbors], dim=0)

    return fused_embeddings


def feature_compression(features, dim=200):
    """Preprcessing of features"""
    features = features.toarray()
    feat = lil_matrix(PCA(n_components=dim, random_state=0).fit_transform(features))
    return feat

def CenterLoss(size1,size2, temperature,z_i, z_j, k=1):
    N = size1 + size2
    mask = torch.ones((N, N))
    mask = mask.fill_diagonal_(0)
    for i in range(size1):
        mask[i, size1 + i] = 0
        mask[size1 + i, i] = 0
    mask = mask.bool()
    mask = mask[:2 * size1, :]
    criterion = nn.CrossEntropyLoss(reduction="none")
    z = torch.cat((z_i, z_j), dim=0)
    sim = torch.matmul(z, z.T) / temperature
    sim_need=sim[:2*size1,:2*size1]
    sim_i_j = torch.diag(sim_need, size1)
    sim_j_i = torch.diag(sim_need, -size1)
    sim=sim[:2*size1,:]

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2*size1, 1)
    negative_samples = sim[mask].reshape(2*size1, -1)

    labels = torch.zeros(2*size1).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    loss = criterion(logits, labels)
    # one = torch.ones(2*self.size1 - (self.size2-self.size1))
    weight = torch.cat((torch.full((2*size1 - (size2-size1),), 1), torch.full((size2-size1,), 1.0)))   #1  2
    loss=loss*weight
    loss=torch.sum(loss)
    loss /= 2*size1
    loss *= k

    return loss