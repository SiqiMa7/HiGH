from torch.nn import Module
import torch, gc
from torch import nn
import numpy as np
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F
import torch.nn.init as init
import dhg
import networkx as nx
class HypergraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, drop_rate=0.4):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, X, hypergraph):
        X = self.theta(X)
        Y = hypergraph.v2e(X, aggr="mean")
        X_ = hypergraph.e2v(Y, aggr="mean")
        X_ = self.drop(self.act(X_))
        return X_


# class Fusion(nn.Module):
#     def __init__(self, input_size, out=1, dropout=0.2):
#         super(Fusion, self).__init__()
#         self.linear1 = nn.Linear(input_size, input_size)
#         self.linear2 = nn.Linear(input_size, out)
#         self.dropout = nn.Dropout(dropout)
#         self.init_weights()
#
#     def init_weights(self):
#         init.xavier_normal_(self.linear1.weight)
#         init.xavier_normal_(self.linear2.weight)
#
#     def forward(self, hidden, dy_emb):
#         '''
#         hidden: 这个子超图HGAT的输入，dy_emb: 这个子超图HGAT的输出
#         hidden和dy_emb都是用户embedding矩阵，大小为(用户数, 64)
#         '''
#         # tensor.unsqueeze(dim) 扩展维度，返回一个新的向量，对输入的既定位置插入维度1
#         # tensor.cat(inputs, dim=?) --> Tensor    inputs：待连接的张量序列     dim：选择的扩维，沿着此维连接张量序列
#         emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
#         emb_score = nn.functional.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
#         emb_score = self.dropout(emb_score)  # 随机丢弃每个用户embedding的权重
#         out = torch.sum(emb_score * emb, dim=0)  # 将输入的embedding和输出的embedding按照对应的用户加权求和
#         return out

def sim( z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())   #矩阵相乘

def get_contrastive_loss(enh_emb, enh_emb1, temp=0.86):
    enh_emb = F.elu(enh_emb)
    enh_emb1 = F.elu(enh_emb1)
    f = lambda x: torch.exp(x / temp)    # 参数：表达式

    refl_sim = f(sim(enh_emb, enh_emb))
    refl_sim_sum1 = refl_sim.sum(1) #Nintra
    refl_sim_diag = refl_sim.diag()
    del refl_sim

    between_sim = f(sim(enh_emb, enh_emb1))
    between_sim_sum1 = between_sim.sum(1) #正样本
    between_sim_diag = between_sim.diag()
    del between_sim

    loss1 = -torch.log(between_sim_diag / (between_sim_sum1 + refl_sim_sum1 - refl_sim_diag))

    refl_sim = f(sim(enh_emb1, enh_emb1))
    refl_sim_sum1 = refl_sim.sum(1) #对refl_sim按行求和，向量的每个元素表示相应行的相似度之和
    refl_sim_diag = refl_sim.diag() #表示取`refl_sim`矩阵的对角线元素
    del refl_sim

    between_sim = f(sim(enh_emb1, enh_emb))
    between_sim_sum1 = between_sim.sum(1)
    between_sim_diag = between_sim.diag()
    del between_sim

    loss2 = -torch.log(between_sim_diag / (between_sim_sum1 + refl_sim_sum1 - refl_sim_diag))

    loss = (loss1.sum() + loss2.sum()) / (2 * len(enh_emb))

    return loss



class MHGCN_douban(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout, loss_function, num_virtual_nodes=20, max_percentage=0.01):
        super(MHGCN_douban, self).__init__()
        """
        # Multilayer Graph Convolution
        """
        self.gc1 = GraphConvolution(nfeat, out)
        self.gc2 = GraphConvolution(out, out)
        self.gc3 = GraphConvolution(64, out)
        # self.gc3 = GraphConvolution(out, 1)
        # self.gc3 = GraphConvolution(out, out)
        # self.gc4 = GraphConvolution(out, out)
        # self.gc5 = GraphConvolution(out, out)
        self.dropout = dropout
        self.output_layer1 = nn.Sequential(
            nn.Linear(out, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.loss_fn = loss_function
        self.loss_lambda = Parameter(torch.tensor(0.5))  # Initialize with a starting value, e.g., 0.5
        self.loss_alpha = Parameter(torch.tensor(0.5))
        # self.linear =nn.Linear(out, 1)
        self.linear_layer = nn.Linear(in_features=200, out_features=64)

        # Learnable weights to fuse degree centrality and k-core scores
        self.centrality_weights = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))

        """
        Set the trainable weight of adjacency matrix aggregation
        """
        # Alibaba
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(4, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b,a = 0,b = 0.1)

        #doubanmovie
        self.weight_b = torch.nn.Parameter(torch.FloatTensor(3, 1), requires_grad=True)
        torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)

        self.virtual_nodes = nn.Parameter(torch.randn(num_virtual_nodes, nfeat), requires_grad=True)
        # learnable connections
        a = np.load('/home/msq/PycharmProjects/duoceng/MHGCN-master/data/DoubanMovie_msq/adj/movie_coactor_adjacency.npy')
        self.a = torch.from_numpy(a).to("cuda").to(torch.float32)
        b = np.load('/home/msq/PycharmProjects/duoceng/MHGCN-master/data/DoubanMovie_msq/adj/movie_codirector_adjacency.npy')
        self.b = torch.from_numpy(b).to("cuda").to(torch.float32)
        c = np.load(
            '/home/msq/PycharmProjects/duoceng/MHGCN-master/data/DoubanMovie_msq/adj/movie_couser_short_adj.npy')
        self.c = torch.from_numpy(c).to("cuda").to(torch.float32)
        # self.virtual_edges_A1 = nn.Parameter(torch.randn(num_virtual_nodes, a.shape[0]), requires_grad=True)
        # self.virtual_edges_A2 = nn.Parameter(torch.randn(num_virtual_nodes, b.shape[0]), requires_grad=True)
        max_virtual_edge_A1 = int(max_percentage * a.shape[0])
        max_virtual_edge_A2 = int(max_percentage * b.shape[0])
        max_virtual_edge_A3 = int(max_percentage * c.shape[0])
        self.adj_A1 = self._virtual_edges(num_virtual_nodes, a.shape[0], max_virtual_edge_A1, self.a)
        self.adj_A2 = self._virtual_edges(num_virtual_nodes, b.shape[0], max_virtual_edge_A2, self.b)
        self.adj_A3 = self._virtual_edges(num_virtual_nodes, b.shape[0], max_virtual_edge_A3, self.c)
        self.layer_token_A1 = nn.Parameter((torch.randn(12697, 10)).to(torch.float32), requires_grad=True)
        self.layer_token_A2 = nn.Parameter((torch.randn(12697, 10)).to(torch.float32), requires_grad=True)
        self.layer_token_A3 = nn.Parameter((torch.randn(12697, 10)).to(torch.float32), requires_grad=True)
        self.weight_token_A1 = nn.Parameter(torch.tensor(0.01), requires_grad=True)
        self.weight_token_A2 = nn.Parameter(torch.tensor(0.01), requires_grad=True)
        self.weight_token_A3 = nn.Parameter(torch.tensor(0.01), requires_grad=True)
        self.hgnn = HypergraphConv(out, out, drop_rate=self.dropout)
        # self.fus = Fusion(out)
        src, dst = self.a.nonzero(as_tuple=True)
        src_list = src.cpu().numpy().tolist()
        dst_list = dst.cpu().numpy().tolist()
        edge_list = list(zip(src_list, dst_list))
        # Create hypergraph: Degree centrality top 10% nodes + KNN hyperedges inside the top set
        hypergraph = self._deg_knn_hypergraph(12677, edge_list, top_pct=0.1, k=10)

        # 超图2
        src_1, dst_1 = self.b.nonzero(as_tuple=True)
        src_list_1 = src_1.cpu().numpy().tolist()
        dst_list_1 = dst_1.cpu().numpy().tolist()
        edge_list_b = list(zip(src_list_1, dst_list_1))
        # Create hypergraph: Degree centrality top 10% nodes + KNN hyperedges inside the top set
        hypergraph_b = self._deg_knn_hypergraph(12677, edge_list_b, top_pct=0.1, k=10)
        # self.hypergraph_list = [hypergraph, hypergraph_b]

        #超图3
        src_2, dst_2 = self.c.nonzero(as_tuple=True)
        src_list_2 = src_2.cpu().numpy().tolist()
        dst_list_2 = dst_2.cpu().numpy().tolist()
        edge_list_c = list(zip(src_list_2, dst_list_2))
        # Create hypergraph: Degree centrality top 10% nodes + KNN hyperedges inside the top set
        hypergraph_c = self._deg_knn_hypergraph(12677, edge_list_c, top_pct=0.1, k=10)
        self.hypergraph_list = [hypergraph, hypergraph_b, hypergraph_c]

    def _deg_knn_hypergraph(self, num_nodes, edge_list, top_pct=0.1, k=10):
        # Build an undirected NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        if len(edge_list) > 0:
            G.add_edges_from(edge_list)

        # Degree centrality (normalized by n-1)
        deg_centrality = nx.degree_centrality(G)
        deg_scores = torch.zeros(num_nodes, dtype=torch.float32)
        for n, s in deg_centrality.items():
            if n < num_nodes:
                deg_scores[n] = float(s)

        # K-core number (discrete), normalize to [0,1] by max core
        if G.number_of_edges() > 0:
            core_numbers = nx.core_number(G)
            core_scores = torch.zeros(num_nodes, dtype=torch.float32)
            for n, c in core_numbers.items():
                if n < num_nodes:
                    core_scores[n] = float(c)
            max_core = core_scores.max().clamp_min(1.0)
            core_scores = core_scores / max_core
        else:
            core_scores = torch.zeros(num_nodes, dtype=torch.float32)

        # Learnable fusion via softmax weights
        w = torch.softmax(self.centrality_weights, dim=0)  # [w_deg, w_core]
        scores = w[0] * deg_scores + w[1] * core_scores
        top_count = max(1, int(top_pct * num_nodes))
        _, top_idx = torch.topk(scores, k=top_count, largest=True, sorted=True)
        top_nodes = set(top_idx.tolist())

        # Build KNN hyperedges within top set using shortest-path distance in G
        if len(top_nodes) == 0:
            return dhg.Hypergraph(num_nodes, [[]])

        G_top = G.subgraph(top_nodes).copy()
        hyperedges = []

        # Precompute shortest path lengths within the top subgraph
        spl = dict(nx.all_pairs_shortest_path_length(G_top))

        for u in top_nodes:
            dist_map = spl.get(u, {})
            candidates = [(v, d) for v, d in dist_map.items() if v != u]
            if not candidates:
                continue
            candidates.sort(key=lambda x: (x[1], x[0]))
            neighbors = [v for v, _ in candidates[:k]]
            if len(neighbors) >= 1:
                hyperedges.append([u] + neighbors)

        if len(hyperedges) == 0:
            hyperedges = [list(top_nodes)]

        return dhg.Hypergraph(num_nodes, hyperedges)

    def _virtual_edges(self, num_virtual_nodes, num_nodes, total_edges, a):
        virtual_edges_A1 = torch.zeros((num_virtual_nodes, a.shape[0]), device=a.device)
        # Example: Randomly assign some connections (ensure this matches your logic)
        all_indices = [(i, j) for i in range(num_virtual_nodes) for j in range(num_nodes)]
        selected_indices = torch.randperm(len(all_indices))[:total_edges]

        for idx in selected_indices:
            i, j = all_indices[idx]
            virtual_edges_A1[i, j] = 1.0
        # Transpose to match dimensions for concatenation
        virtual_edges_A1_transposed = virtual_edges_A1.t()

        # Create a zero matrix for virtual-to-virtual connections
        virtual_to_virtual = torch.zeros((num_virtual_nodes, num_virtual_nodes), device=a.device)

        # Concatenate to form the new adjacency matrix
        # Top part: original nodes + virtual nodes connections
        top_part = torch.cat((a, virtual_edges_A1_transposed), dim=1)

        # Bottom part: virtual nodes connections + virtual-to-virtual connections
        bottom_part = torch.cat((virtual_edges_A1, virtual_to_virtual), dim=1)

        # Full adjacency matrix with virtual nodes
        full_adjacency_matrix = torch.cat((top_part, bottom_part), dim=0)
        return full_adjacency_matrix

    def _initialize_virtual_edges(self, num_virtual_nodes, num_nodes, total_edges):
        # Create a zero matrix
        edges = torch.zeros((num_virtual_nodes, num_nodes), dtype=torch.float32)

        # Randomly select positions to set to 1, ensuring the total number of 1s is exactly total_edges
        all_indices = [(i, j) for i in range(num_virtual_nodes) for j in range(num_nodes)]
        selected_indices = torch.randperm(len(all_indices))[:total_edges]

        for idx in selected_indices:
            i, j = all_indices[idx]
            edges[i, j] = 1.0

        return edges

    def forward(self, feature, labels=None, train_idx=None, use_relu=True):
        # final_A = adj_matrix_weight_merge(self.weight_b, self.a, self.b, self.c).to("cuda")
        final_A, A1_token, A2_token, A3_token = adj_matrix_weight_merge(self.weight_b, self.layer_token_A1, self.layer_token_A2, self.layer_token_A3,
                                                              self.adj_A1, self.adj_A2, self.adj_A3)
        final_A = final_A.to("cuda")
        A1_token = A1_token.to("cuda")
        A2_token = A2_token.to("cuda")
        A3_token = A3_token.to("cuda")
        try:
            feature = torch.tensor(feature.astype(float).toarray())
        except:
            try:
                feature = torch.from_numpy(feature.toarray())
            except:
                pass

        feature_A1 = self.gc3(feature, self.a)
        feature_A1 = F.dropout(feature_A1, self.dropout, training=self.training)
        feature_A2 = self.gc3(feature, self.b)
        feature_A2 = F.dropout(feature_A2, self.dropout, training=self.training)
        feature_A3 = self.gc3(feature, self.c)
        feature_A3 = F.dropout(feature_A3, self.dropout, training=self.training)


        contrastloss_1 = get_contrastive_loss(feature_A1, feature_A2)
        # contrastloss_1_nor = contrastloss_1 / contrastloss_1.detach().mean()
        contrastloss_2 = get_contrastive_loss(feature_A2, feature_A3)
        # contrastloss_2_nor = contrastloss_2 / contrastloss_2.detach().mean()
        contrastloss_3 = get_contrastive_loss(feature_A3, feature_A1)
        # contrastloss_3_nor = contrastloss_3 / contrastloss_3.detach().mean()
        # contrastloss = (contrastloss_1 + contrastloss_2 + contrastloss_3) / 3

        hg_embeddings = []
        subhg_embedding_A1 = self.hgnn(feature_A1, self.hypergraph_list[0])
        # hg_embeddings.append(subhg_embedding)
        subhg_embedding_A2 = self.hgnn(feature_A2, self.hypergraph_list[1])
        # subhg_embedding = self.fus(hg_embeddings[-1], subhg_embedding)
        # hg_embeddings.append(subhg_embedding)
        subhg_embedding_A3 = self.hgnn(feature_A3, self.hypergraph_list[2])
        # subhg_embedding = self.fus(hg_embeddings[-1], subhg_embedding)
        # hg_embeddings.append(subhg_embedding)

        # print(f'self.user_embeddings[{i}].weight = {self.user_embeddings[i].weight}')
        # 返回最后一个时刻的用户embedding
        # hypergraph_emb = hg_embeddings[-1]
        hypergraph_emb = subhg_embedding_A1 + subhg_embedding_A2 + subhg_embedding_A3
        # hypergraph_emb = F.softmax(hypergraph_emb, dim=0)
        # score_hypergraph = self.output_layer1(hypergraph_emb)
        feature_A1 = self.linear_layer(feature_A1)
        feature_A2 = self.linear_layer(feature_A2)
        feature_A3 = self.linear_layer(feature_A3)
        feature = 0.0001 * feature_A1 + 0.0001 * feature_A2 + feature + 0.0001 * feature_A3
        # Assuming feature is a 2D tensor with shape (num_nodes, feature_dim)
        num_virtual_nodes = 20
        feature_dim = feature.size(1)

        # Step 1: Add zero vectors
        zero_vectors = torch.zeros((num_virtual_nodes, feature_dim), device=feature.device)
        feature = torch.cat((feature, zero_vectors), dim=0)

        # Step 2: Concatenate with A1_feature and A2_feature
        feature = torch.cat((feature, self.weight_token_A1 * A1_token, self.weight_token_A2 * A2_token, self.weight_token_A3 * A3_token), dim=1)

        # Step 3: Replace zero vectors with virtual nodes
        feature[-num_virtual_nodes:] = self.virtual_nodes

        # Output of single-layer GCN
        # U1 = torch.sigmoid(self.gc1(feature, final_A))
        U1 =self.gc1(feature, final_A)
        # Output of two-layer GCN
        # U2 = self.gc2(U1, final_A)
        U2 = torch.sigmoid(self.gc2(U1, final_A))
        emb = (U1 +U2)/2
        emb_final = emb[:12677] + 1 * hypergraph_emb
        # scores = self.output_layer1((U1+U2)/2)
        scores = self.output_layer1(emb_final)
        # combined_scores_part = scores_intial[:12677] + 1000 * score_hypergraph
        # scores = torch.cat((combined_scores_part, scores_intial[12677:]), dim=0)
        # U2 = self.gc2(U1, final_A)
        # U3 = self.gc3(U2, final_A)
        # U4 = self.gc4(U2, final_A)
        # U5 = self.gc5(U2, final_A)
        # U3 = self.linear(U1)
        # max_U3 = max(U3)
        # U3 = torch.softmax(U2)

        # Average pooling
        # return (U1+U2)/2

        loss_lambda = torch.clamp(self.loss_lambda, min=0, max=1)
        loss_alpha = torch.clamp(self.loss_alpha, min=0, max=1)

        if self.training:
            loss_struct = self.loss_fn(scores[train_idx], labels[train_idx].unsqueeze(-1))
            loss_content = self.loss_fn(scores[train_idx], labels[train_idx].unsqueeze(-1))
            loss_all = self.loss_fn(scores[train_idx], labels[train_idx].unsqueeze(-1))
            loss = (1-loss_lambda) * loss_all + loss_lambda * (loss_struct + loss_content) / 2
            loss = loss_alpha * list_loss(scores[train_idx], labels[train_idx].unsqueeze(-1), 100) + loss +0.03 * (contrastloss_1 + contrastloss_2 + contrastloss_3) / 3
            return scores, loss
        else:
            return scores
        return scores

def adj_matrix_weight_merge(adj_weight, layer_token_A1, layer_token_A2, layer_token_A3, A1, A2, A3):
    """
    Multiplex Relation Aggregation
    """

    # #doubanmovie
    A1_feature = torch.matmul(A1, layer_token_A1)
    A2_feature = torch.matmul(A2, layer_token_A2)
    A3_feature = torch.matmul(A3, layer_token_A3)
    gc.collect()
    torch.cuda.empty_cache()
        # with torch.no_grad():
    #     a = A[0]
    #     b = A[1]
    #     c = A[2]
    # a = coototensor(A[0][0])
    # b = coototensor(A[0][1])
    # c = coototensor(A[0][2])
    A_t = torch.stack([A1, A2, A3], dim=2).to_dense()
    adj_weight = adj_weight.to("cuda")
    A_t = A_t.float()
    temp = torch.matmul(A_t, adj_weight)
    temp = torch.squeeze(temp, 2)
    temp_1 =temp + temp.transpose(0, 1)
    # temp_1 = F.normalize(temp_1, p=2, dim=1)
    del A_t, temp

    return temp_1, A1_feature, A2_feature, A3_feature

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))#鏉冮噸鐭╅樀
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))#鍋忕Щ鍚戦噺
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        try:
            input = input.float()
        except:
            pass
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        # output = F.normalize(output, p=2, dim=1)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


def list_loss(y_pred, y_true, list_num=10, eps=1e-10):
    '''
    y_pred: [n_node, 1]
    y_true: [n_node, 1]
    '''
    n_node = y_pred.shape[0]

    ran_num = list_num - 1
    indices = torch.multinomial(torch.ones(n_node), n_node*ran_num, replacement=True).to(y_pred.device)

    list_pred = torch.index_select(y_pred, 0, indices).reshape(n_node, ran_num)
    list_true = torch.index_select(y_true, 0, indices).reshape(n_node, ran_num)

    list_pred = torch.cat([y_pred, list_pred], -1) # [n_node, list_num]
    list_true = torch.cat([y_true, list_true], -1) # [n_node, list_num]

    list_pred = F.softmax(list_pred, -1)
    list_true = F.softmax(list_true, -1)

    list_pred = list_pred + eps
    log_pred = torch.log(list_pred)

    return torch.mean(-torch.sum(list_true * log_pred, dim=1))