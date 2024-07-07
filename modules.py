import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.autograd import Variable
from trainer import *
from numpy import *
import pandas as pd

# word-semantic attention mechanisms
class WATs(nn.Module):
    def __init__(self, input_dim=100, output_dim=100):
        super(WATs, self).__init__()
        self.W = nn.Parameter(torch.randn(output_dim, input_dim))
    def graph_attention(self, entity):
        h,r,v = entity # h torch.Size([b, 100]) # r torch.Size([b, 100, 100]) # b torch.Size([5, 100, 100])
        Wr = torch.randn(100, 100).to('cuda:0')
        b = h.shape[0]
        Wr_v = torch.bmm(v, Wr.unsqueeze(0).expand(b, 100, 100))
        Wr_h_r = torch.bmm(h.unsqueeze(2) + r, Wr.unsqueeze(0).expand(b, 100, 100))
        tanh_Wr_h_r = torch.tanh(Wr_h_r)
        Wr_v_transposed = Wr_v.transpose(1, 2)
        W_i = torch.bmm(Wr_v_transposed, tanh_Wr_h_r).squeeze(-1)
        W_i = W_i.mean(dim=-1)
        return W_i

    def forward(self, left, right):
        e_A_head = self.graph_attention(left)
        e_A_tail = self.graph_attention(right)
        return e_A_head,e_A_tail
        
# Cluster partitioning
class Cluster_group:
    def clustering(self, adj_matrix):
        A = adj_matrix
        D = torch.diag(torch.sum(A, axis=1))
        L = D - A
        # Decompose the Laplace matrix
        eigenvalues, eigenvectors = np.linalg.eig(L)
        eigenvalues = np.round(eigenvalues, decimals=4)
        # Calculate the inverse of the square root of D
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        # Compute symmetric normalized Laplace matrices
        L_sym = np.eye(len(D)) - D_inv_sqrt @ A.numpy() @ D_inv_sqrt
        # The normalized Laplace matrix is characterized by feature decomposition
        eigenvalues_sym, eigenvectors_sym = np.linalg.eig(L_sym)
        eigenvalues_sym=np.round(eigenvalues_sym, decimals=4)

        # calculate the absolute difference between consecutive elements
        abs_diffs = np.abs(np.diff(eigenvalues_sym))

        # find the maximum absolute difference
        max_abs_diff = np.max(abs_diffs)

        # find the indices of all elements equal to the maximum absolute difference
        max_abs_diff_indices = np.where(abs_diffs == max_abs_diff)[0]
        g = max_abs_diff_indices[0]+1
        x_g = eigenvectors[:, g]
        series_g = pd.Series(x_g)
        # group by values and collect indices for each group
        group = series_g.groupby(series_g).apply(lambda x: list(x.index)).tolist()
        return group

    def b_group_cut(self, adj_matrix_left, adj_matrix_right):
        num_samples1 = adj_matrix_left.shape[0]
        num_samples2 = adj_matrix_right.shape[0]
        group_left = []
        group_right = []
        for i in range(num_samples1):
            subgroup_left = self.clustering(adj_matrix_left[i])
            group_left.append(subgroup_left)
        for i in range(num_samples2):
            subgroup_right = self.clustering(adj_matrix_right[i])
            group_right.append(subgroup_right)
        group_left = [[[num for num in sublist if num != 100] for sublist in group] for group in group_left]
        group_right = [[[num for num in sublist if num != 100] for sublist in group] for group in group_right]
        groups = [group_left, group_right]
        return groups

# structure-aware graph attention encoder
class SAGAN(nn.Module):
    def __init__(self,dim=100, dropout=0.0,e1_rele2=None):
        super(SAGAN,self).__init__()
        self.creatematirx = CreateAdjacencyMatrix(e1_rele2)
        self.Linear_tail = nn.Linear(dim, dim, bias=False)
        self.Linear_head = nn.Linear(dim, dim, bias=False)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.C_group = Cluster_group()
    # structure-semantic attention mechanisms
    def nbr_struct_emb(self,connections_left,connections_right,left,right):# #left = [head_left, rel_embeds_left, ent_embeds_left]
        nbrs_left_id = connections_left[:,:,1].tolist()
        nbrs_right_id = connections_right[:,:,1].tolist()
        adj_matrix_left,adj_matrix_right = self.creatematirx.get_matrix(nbrs_left_id,nbrs_right_id)

        adj_matrix_left = torch.tensor(adj_matrix_left, dtype=torch.float32)
        adj_matrix_right = torch.tensor(adj_matrix_right, dtype=torch.float32)
        groups = self.C_group.b_group_cut(adj_matrix_left, adj_matrix_right)
        ent_embeds_left = left[2]
        ent_embeds_right = right[2]
        groups_left = groups[0]
        groups_right = groups[1]
        w_list_left = []
        for i, group in enumerate(groups_left): 
            subgroup = [torch.index_select(ent_embeds_left[i], 0, (torch.tensor(g)).to('cuda:0')) for g in group]
            subgroup_vector = torch.cat(subgroup, dim=0) #torch.Size([100, 100])
            v_i = subgroup_vector.mean(dim=0, keepdim=True)
            v_i_transposed = v_i.transpose(0, 1)
            scores = torch.matmul(subgroup_vector, v_i_transposed)
            scores_squeezed = scores.squeeze(-1)
            w_i = torch.softmax(scores_squeezed, dim=0)
            w_i = w_i.unsqueeze(0) #　torch.Size([b, 100])
            w_list_left.append(w_i)
        w_left = torch.stack(w_list_left).squeeze(1) # torch.Size([b, 100])
        w_list_right = []
        for i, group in enumerate(groups_right): 
            subgroup = [torch.index_select(ent_embeds_right[i], 0, (torch.tensor(g)).to('cuda:0')) for g in group]
            subgroup_vector = torch.cat(subgroup, dim=0) #torch.Size([100, 100])
            v_i = subgroup_vector.mean(dim=0, keepdim=True)
            v_i_transposed = v_i.transpose(0, 1)
            scores = torch.matmul(subgroup_vector, v_i_transposed)
            scores_squeezed = scores.squeeze(-1)
            w_i = torch.softmax(scores_squeezed, dim=0)
            w_i = w_i.unsqueeze(0) #　torch.Size([b, 100])
            w_list_right.append(w_i)
        w_right = torch.stack(w_list_right).squeeze(1) # torch.Size([b, 100])
        e_S_left = w_left
        e_S_right = w_right
        return e_S_left,e_S_right # [b,100]

    def forward(self,left,right,connections_left,connections_right):#left = [head_left, rel_embeds_left, ent_embeds_left]
        e_S_left,e_S_right = self.nbr_struct_emb(connections_left,connections_right,left,right)
        return e_S_left, e_S_right # Structural neighborhood information
       
# Neighborhood information aggregation and entity updates 
class Aggregation(nn.Module):
    def __init__(self, dim = 100, dropout=0.0,e1_rele2=None):
        super(Aggregation,self).__init__()
        self.dropout=0.0,
        self.e1_rele2=e1_rele2
        self.Linear_A = nn.Linear(dim, dim, bias=False)
        self.Linear_G = nn.Linear(dim, dim, bias=False) 
        self.Linear_left_j = nn.Linear(dim, dim, bias=False)
        self.Linear_right_j = nn.Linear(dim, dim, bias=False)
        self.Linear_left_c = nn.Linear(dim, dim, bias=False)
        self.Linear_right_c = nn.Linear(dim, dim, bias=False)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.WATencoder = WATs(input_dim=100, output_dim=100)
        self.sagan = SAGAN(dim=100, dropout=0.0,e1_rele2=e1_rele2)

    # Aggregate word semantics as well as structural semantic information
    def bil_aggregation(self, left, right,connections_left,connections_right):
        head_left, rel_left, tail_left = left #  head_left [b,100]
        head_right, rel_right, tail_right = right
        e_A_left,e_A_right =self.WATencoder(left, right)
        e_G_left,e_G_right = self.sagan(left,right,connections_left,connections_right)
        e_left = torch.relu(self.Linear_A(e_A_left) + self.Linear_G(e_G_left))
        e_right = torch.relu(self.Linear_A(e_A_right) + self.Linear_G(e_G_right))
        head_left = torch.relu(self.Linear_left_j(head_left+e_left))+torch.relu(self.Linear_left_c(head_left*e_left))
        head_right = torch.relu(self.Linear_right_j(head_left+e_left))+torch.relu(self.Linear_right_c(head_right*e_right))
        head_left = self.dropout(head_left)
        head_right = self.dropout(head_right)
        return head_left,head_right
    def forward(self,left, right, connections_left,connections_right):
        head,tail = self.bil_aggregation(left, right,connections_left,connections_right)
        return head,tail

# Create an adjacency matrix for the subgraph
class CreateAdjacencyMatrix():
    def __init__(self,e1_rele2):
        self.e1_rele2 = e1_rele2
    def get_graph_pairs(self,nbrs_id,e1_rele2):
        graph_pairs = []  
        graph_pairs_multi = []
        for sub_nbrs_id in nbrs_id:  
            sub_graph_pairs_multi = []
            for i in range(len(sub_nbrs_id)):  
                if list(sub_nbrs_id[i].values())[0] == 69126:
                    break
                for j in range(i + 1, len(sub_nbrs_id)):
                    dict1 = sub_nbrs_id[i]
                    dict2 = sub_nbrs_id[j]
                    value1 = list(dict1.values())[0]
                    value2 = list(dict2.values())[0]
                    if value1 == value2:
                        key1 = list(dict1.keys())[0]
                        key2 = list(dict2.keys())[0]
                        pair = (key1, key2)
                        sub_graph_pairs_multi.append(pair)
            graph_pairs_multi.append(sub_graph_pairs_multi)
            graph_pairs_connect = []  # Record connection tuples between neighbors
            for sub_nbrs_id in nbrs_id:
                sub_graph_pairs = []
                for nbr in sub_nbrs_id:
                    for i in range(len(sub_nbrs_id)):
                        if list(sub_nbrs_id[i].values())[0] == 69126:
                            break
                        pairs_list = e1_rele2[list(sub_nbrs_id[i].values())[0]]  # Extract two-hop neighbors                 
                        tmp_list = []
                        for pair in pairs_list:
                            for element in pair:
                                if element != list(sub_nbrs_id[i].values())[0]:
                                    tmp_list.append(element)
                        subnbr_list = list(set(tmp_list))  
                        for val in subnbr_list: 
                            if val == list(sub_nbrs_id[i].values())[0]:
                                key1 = list(nbr.keys())[0]
                                key2 = list(sub_nbrs_id[i].keys())[0]
                                sub_graph_pairs.append((key1, key2))
                graph_pairs_connect.append(sub_graph_pairs)
            for index in range(len(graph_pairs_multi)):
                graph_pairs.append([])
                graph_pairs[index].extend(graph_pairs_multi[index])
                graph_pairs[index].extend(graph_pairs_connect[index])
            return graph_pairs

    def create_adjacency_matrix(self, num_nodes, graph_pairs):
        adj_matrix = []
        for index, sub_graph_pairs in enumerate(graph_pairs):
            adj_matrix.append([])
            sub_adj_matrix = np.zeros((num_nodes + 1, num_nodes + 1), dtype=int)
            sub_adj_matrix[0, :] = 1
            sub_adj_matrix[:, 0] = 1
            # Complete the adjacency matrix based on the connection information in the sub_graph_pairs
            for edge in sub_graph_pairs:
                i, j = edge
                sub_adj_matrix[i, j] = 1
                sub_adj_matrix[j, i] = 1
            np.fill_diagonal(sub_adj_matrix, 0)
            adj_matrix[index].extend(sub_adj_matrix.tolist())
        adj_matrix = np.array(adj_matrix)
        return adj_matrix

    def get_matrix(self,nbrs_left_id,nbrs_right_id):
        new_nbrs_left_id = nbrs_left_id
        new_nbrs_right_id = nbrs_right_id
        for i, sub_nbrs_left_id in enumerate(nbrs_left_id):
            for j, value in enumerate(sub_nbrs_left_id):
                key = j + 1  
                new_nbrs_left_id[i][j] = {key: value} 
        for k, sub_nbrs_right_id in enumerate(nbrs_right_id):
            for l, value in enumerate(sub_nbrs_right_id):
                key = l + 1  
                new_nbrs_right_id[k][l] = {key: value} 
        graph_pairs_left = self.get_graph_pairs(new_nbrs_left_id, self.e1_rele2)
        graph_pairs_right = self.get_graph_pairs(new_nbrs_right_id, self.e1_rele2)
        adj_matrix_left = self.create_adjacency_matrix(len(new_nbrs_left_id[0]), graph_pairs_left)
        adj_matrix_right = self.create_adjacency_matrix(len(new_nbrs_right_id[0]), graph_pairs_right)
        return adj_matrix_left,adj_matrix_right 

# Transformer-based entity pair encoder
class TransformerEncoder(nn.Module):
    def __init__(self, model_dim=100, ffn_dim=800, num_heads=4, dropout=0.1, num_layers=6, max_seq_len=3,
                 with_pos=True):
        super(TransformerEncoder, self).__init__()
        self.with_pos = with_pos
        self.num_heads = num_heads

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim * num_heads, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        self.rel_embed = nn.Parameter(torch.rand(1, model_dim), requires_grad=True)

    def repeat_dim(self, emb):
        """
        :param emb: [batch, t, dim]
        :return:
        """
        return emb.repeat(1, 1, self.num_heads)

    def forward(self, left, right):
        """
        :param left: [batch, dim]
        :param right: [batch, dim]
        :return:
        """
        batch_size = left.size(0)
        left = left.unsqueeze(1)
        right = right.unsqueeze(1)
        seq = torch.cat((left, right), dim=1)
        pos = self.pos_embedding(batch_len=batch_size, seq_len=2)
        if self.with_pos:
            output = seq + pos
        else:
            output = seq
        output = self.repeat_dim(output)
        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output)
            attentions.append(attention)
        return output[:, 1, :]

# Semantic prototype matching network
class SemanticPrototypeNet(nn.Module):
    def __init__(self, r_dim):
        super(SemanticPrototypeNet, self).__init__()
        self.Attention = SoftSelectAttention(hidden_size=r_dim)

    def forward(self, support, query):
        center = self.Attention(support, query)
        return center

class SoftSelectAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SoftSelectAttention, self).__init__()

    def forward(self, support, query):
        """
        :param support: [few, dim]
        :param query: [batch, dim]
        :return:
        """
        query_ = query.unsqueeze(1).expand(query.size(0), support.size(0), query.size(1)).contiguous()  # [b, few, dim]
        support_ = support.unsqueeze(0).expand_as(query_).contiguous()  # [b, few, dim]
        scalar = support.size(1) ** -0.5  # dim ** -0.5
        score = torch.sum(query_ * support_, dim=2) * scalar
        att = torch.softmax(score, dim=1)
        center = torch.mm(att, support)
        return center
    
class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention
    """
    def __init__(self, attn_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        :param attn_mask: [batch, time]
        :param scale:
        :param q: [batch, time, dim]
        :param k: [batch, time, dim]
        :param v: [batch, time, dim]
        :return:
        """
        attn = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attn = attn * scale
        if attn_mask:
            attn = attn.masked_fill_(attn_mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    """ Implement without batch dim"""
    def __init__(self, model_dim, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads

        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        """
        To be efficient, multi- attention is cal-ed in a matrix totally
        :param attn_mask:
        :param query: [batch, time, per_dim * num_heads]
        :param key:
        :param value:
        :return: [b, t, d*h]
        """
        residual = query
        batch_size = key.size(0)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        key = key.view(batch_size * self.num_heads, -1, self.dim_per_head)
        value = value.view(batch_size * self.num_heads, -1, self.dim_per_head)
        query = query.view(batch_size * self.num_heads, -1, self.dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)

        scale = (key.size(-1) // self.num_heads) ** -0.5
        context, attn = self.dot_product_attention(query, key, value, scale, attn_mask)
        context = context.view(batch_size, -1, self.dim_per_head * self.num_heads)
        output = self.linear_final(context)
        output = self.dropout(output)
        output = self.layer_norm(residual + output)
        return output, attn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()

        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        pad_row = torch.zeros([1, d_model], dtype=torch.float)
        position_encoding = torch.tensor(position_encoding, dtype=torch.float)

        position_encoding = torch.cat((pad_row, position_encoding), dim=0)

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)

    def forward(self, batch_len, seq_len):
        """
        :param batch_len: scalar
        :param seq_len: scalar
        :return: [batch, time, dim]
        """
        input_pos = torch.tensor([list(range(1, seq_len + 1)) for _ in range(batch_len)]).cuda()
        return self.position_encoding(input_pos)


class GELU(nn.Module):
    """
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.gelu = GELU()

    def forward(self, x):
        """

        :param x: [b, t, d*h]
        :return:
        """
        output = x.transpose(1, 2)  # [b, d*h, t]
        output = self.w2(self.gelu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        # ffn_dim
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        output = self.feed_forward(context)
        return output, attention

