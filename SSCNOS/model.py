from torch import nn
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv



        
        
class GradReverse(torch.autograd.Function):
    # lambd = 0.0

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return args[0].view_as(args[0])

    @staticmethod
    def backward(ctx, *grad_outputs):
        return grad_outputs[0]*-GradReverse.lambd


def grad_reverse(x, lambd=1.0):
    GradReverse.lambd = lambd
    return GradReverse.apply(x)


class GAT(nn.Module):
    def __init__(self, num_layers, in_dim, num_hidden, heads, activation, feat_drop, attn_drop, negative_slope, residual, num_classes):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, False, self.activation))
        # # # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs, g):
        heads = []
        h = inputs
        # get hidden_representation
        for l in range(self.num_layers):
            temp = h.flatten(1)  # 保存上一层multi-head flatten拼接的结果
            # h = self.gat_layers[l](self.g, temp)
            # h = self.gat_layers[l](g, temp,  get_attention=True)[0]
            h,attention = self.gat_layers[l](g, temp, get_attention=True)

            # h = self.gat_layers[l](self.g, h).flatten(1)
        # print(attention[0,:,:])
        mean_attention = torch.mean(attention, dim=1)
        # mean_attention=torch.sigmoid(mean_attention)  #修改

        # 确保结果是(9188, 1)的形状
        # mean_attention = mean_attention.unsqueeze(-1)
        # print(mean_attention[0, :])
        # get heads
        for i in range(h.shape[1]):
            heads.append(h[:, i])
        # # output projection
        # logits = self.gat_layers[-1](g, h.flatten(1)).mean(1)
        # # logits = self.gat_layers[-1](g, torch.cat(h_allLayers, axis=1)).mean(1)
        # # hidden_rep=h.flatten(1)
        return heads, mean_attention




class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim_mlp):
        super(DomainDiscriminator, self).__init__()
        self.h_dann_1 = nn.Linear(input_dim_mlp, 32)
        self.h_dann_2 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, 2)
        # std = 1/(input_dim_mlp/2)**0.5
        # nn.init.trunc_normal_(self.h_dann_1.weight, std=std, a=-2*std, b=2*std)
        nn.init.xavier_normal_(self.h_dann_1.weight, 1.414)
        nn.init.constant_(self.h_dann_1.bias, 0.1)
        # nn.init.trunc_normal_(self.h_dann_2.weight, std=0.125, a=-0.25, b=0.25)
        nn.init.xavier_normal_(self.h_dann_2.weight, 1.414)
        nn.init.constant_(self.h_dann_2.bias, 0.1)
        # nn.init.trunc_normal_(self.output_layer.weight, std=0.125, a=-0.25, b=0.25)
        nn.init.xavier_normal_(self.output_layer.weight, 1.414)
        nn.init.constant_(self.output_layer.bias, 0.1)

    def forward(self, h_grl):
        h_grl = F.relu(self.h_dann_1(h_grl))
        h_grl = F.relu(self.h_dann_2(h_grl))
        d_logit = self.output_layer(h_grl)
        return d_logit


# class GAT(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
#         """Dense version of GAT."""
#         super(GAT, self).__init__()
#         self.dropout = dropout
#
#         self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
#
#         self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
#
#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x_list = []
#         for att in self.attentions:
#             x_list.append(att(x, adj))
#         x = torch.cat(x_list, dim=1)
#         # x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj))
#         return F.log_softmax(x, dim=1)
#
# class GraphAttentionLayer(nn.Module):
#     """
#     Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """
#     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat
#
#         self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#         self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)
#
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#
#     def forward(self, h, adj):
#         Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
#         e = self._prepare_attentional_mechanism_input(Wh)
#
#         zero_vec = -9e15*torch.ones_like(e)
#         attention = torch.where(adj > 0, e, zero_vec)
#         attention = F.softmax(attention, dim=1)
#         attention = F.dropout(attention, self.dropout, training=self.training)
#         h_prime = torch.matmul(attention, Wh)
#
#         if self.concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime
#
#     def _prepare_attentional_mechanism_input(self, Wh):
#         # Wh.shape (N, out_feature)
#         # self.a.shape (2 * out_feature, 1)
#         # Wh1&2.shape (N, 1)
#         # e.shape (N, N)
#         Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
#         Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
#         # broadcast add
#         e = Wh1 + Wh2.T
#         return self.leakyrelu(e)
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'




class LHCDA(nn.Module):
    def __init__(self, num_layers, in_dim, num_hidden, heads, activation, feat_drop, attn_drop, negative_slope, residual, num_classes):
        super(LHCDA, self).__init__()
        self.network_embedding = GAT(num_layers, in_dim, num_hidden, heads, activation, feat_drop, attn_drop, negative_slope, residual, num_classes)
        # self.domain_discriminator = DomainDiscriminator(input_dim_mlp)

    def forward(self, features_s, features_t, g_s, g_t):
        # head_s,mean_attention_s = self.network_embedding(features_s, g_s)
        head_s, _ = self.network_embedding(features_s, g_s)  #原来
        emb_s = torch.cat(head_s, axis=1)
        head_t,mean_attention_t= self.network_embedding(features_t, g_t)
        emb_t = torch.cat(head_t, axis=1)      
        
        # emb = torch.cat((emb_s, emb_t), 0)
        #
        #
        # # Domain_Discriminator
        # h_grl = grad_reverse(emb, grl_lambda)
        # d_logit = self.domain_discriminator(h_grl)

        return mean_attention_t, emb_s, emb_t, head_s, head_t


class Cly_net(nn.Module):
    def __init__(self, ninput, noutput,multi_class):
        super(Cly_net, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        self.multi_class=multi_class

        self.layers = []
        layer = nn.Linear(self.ninput, noutput)
        self.layers.append(layer)
        self.add_module("layer0", layer)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        # if self.multi_class:
        #     x=torch.sigmoid(x)
        # else:
        #     x=F.softmax(x,dim=1)


        return x


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j, k=0.01):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        loss *= k

        return loss


class CenterLoss(nn.Module):
    def __init__(self, size1,size2, temperature):
        super(CenterLoss, self).__init__()
        self.size1 = size1
        self.size2 = size2
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(size1,size2)
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def mask_correlated_samples(self, size1,size2):
        N = size1+size2
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(size1):
            mask[i, size1 + i] = 0
            mask[size1 + i, i] = 0
        mask = mask.bool()
        mask = mask[:2 * size1, :]
        return mask

    def forward(self, z_i, z_j, k=1):
        N = self.size1+self.size2
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_need=sim[:2*self.size1,:2*self.size1]
        sim_i_j = torch.diag(sim_need, self.size1)
        sim_j_i = torch.diag(sim_need, -self.size1)
        sim=sim[:2*self.size1,:]

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2*self.size1, 1)
        negative_samples = sim[self.mask].reshape(2*self.size1, -1)

        labels = torch.zeros(2*self.size1).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        # one = torch.ones(2*self.size1 - (self.size2-self.size1))
        weight = torch.cat((torch.full((2*self.size1 - (self.size2-self.size1),), 1), torch.full((self.size2-self.size1,), 2)))   #1  2
        loss=loss*weight
        loss=torch.sum(loss)
        loss /= 2*self.size1
        loss *= k

        return loss