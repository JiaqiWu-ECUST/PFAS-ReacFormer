import torch
from torch import nn
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
import torch.nn.functional as F
from utils import scaled_dot_product_attention
from data import NUM_BOND_TYPE, NUM_BOND_DIRECTION, NUM_BOND_STEREO, \
    NUM_BOND_INRING, NUM_BOND_ISCONJ


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr="add", bond_feat_red="mean"):
        super().__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        self.edge_embedding3 = torch.nn.Embedding(NUM_BOND_STEREO, emb_dim)
        self.edge_embedding4 = torch.nn.Embedding(NUM_BOND_INRING, emb_dim)
        self.edge_embedding5 = torch.nn.Embedding(NUM_BOND_ISCONJ, emb_dim)
        self.edge_embedding6 = torch.nn.Embedding(12, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding5.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding6.weight.data)

        self.edge_embedding_lst = [self.edge_embedding1, self.edge_embedding2, self.edge_embedding3,
                                   self.edge_embedding4, self.edge_embedding5,self.edge_embedding6]

        self.aggr = aggr
        self.bond_feat_red = bond_feat_red


    def norm(self, edge_index, num_nodes=None, dtype=None, x=None):

        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index


        if x is not None:
            num_nodes = x.size(0)
        if row.numel():
            num_nodes = max(num_nodes, int(row.max().item()) + 1)

        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))


        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))


        if edge_attr.size(1) > 0:
            self_loop_attr[:, 0] = NUM_BOND_TYPE - 1

        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = []
        for i in range(edge_attr.shape[1]):
            edge_embeddings.append(self.edge_embedding_lst[i](edge_attr[:, i]))

        if self.bond_feat_red == "mean":
            edge_embeddings = torch.stack(edge_embeddings).mean(dim=0)
        elif self.bond_feat_red == "sum":
            edge_embeddings = torch.stack(edge_embeddings).sum(dim=0)
        else:
            raise ValueError("Invalid bond feature reduction method. Please choose from 'mean' or 'sum'")

        norm = self.norm(edge_index, x.size(0), x.dtype,x)
        x = self.linear(x)
        return self.propagate(edge_index=edge_index, aggr=self.aggr, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GINConv(MessagePassing):

    def __init__(self, emb_dim, out_dim, aggr="add", bond_feat_red="mean"):
        self.aggr = aggr
        super().__init__()

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, out_dim))
        self.edge_embedding1 = torch.nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        self.edge_embedding3 = torch.nn.Embedding(NUM_BOND_STEREO, emb_dim)
        self.edge_embedding4 = torch.nn.Embedding(NUM_BOND_INRING, emb_dim)
        self.edge_embedding5 = torch.nn.Embedding(NUM_BOND_ISCONJ, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding5.weight.data)

        self.edge_embedding_lst = [self.edge_embedding1, self.edge_embedding2, self.edge_embedding3,
                                   self.edge_embedding4, self.edge_embedding5]
        self.bond_feat_red = bond_feat_red

    def forward(self, x, edge_index, edge_attr):

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))


        self_loop_attr = torch.zeros(x.size(0), len(self.edge_embedding_lst))
        self_loop_attr[:, 0] = NUM_BOND_TYPE - 1
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = []
        for i in range(edge_attr.shape[1]):
            edge_embeddings.append(self.edge_embedding_lst[i](edge_attr[:, i]))
        if self.bond_feat_red == "mean":
            edge_embeddings = torch.stack(edge_embeddings).mean(dim=0)

        elif self.bond_feat_red == "sum":
            edge_embeddings = torch.stack(edge_embeddings).sum(dim=0)

        else:
            raise ValueError("Invalid bond feature reduction method. Please choose from 'mean' or 'sum'")

        return self.propagate(edge_index=edge_index, aggr=self.aggr, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GATConv(MessagePassing):

    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add", bond_feat_red="mean"):
        super().__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(NUM_BOND_TYPE, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(NUM_BOND_DIRECTION, heads * emb_dim)
        self.edge_embedding3 = torch.nn.Embedding(NUM_BOND_STEREO, heads * emb_dim)
        self.edge_embedding4 = torch.nn.Embedding(NUM_BOND_INRING, heads * emb_dim)
        self.edge_embedding5 = torch.nn.Embedding(NUM_BOND_ISCONJ, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding5.weight.data)

        self.edge_embedding_lst = [self.edge_embedding1, self.edge_embedding2, self.edge_embedding3,
                                   self.edge_embedding4, self.edge_embedding5]

        self.reset_parameters()
        self.bond_feat_red = bond_feat_red

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):


        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))


        self_loop_attr = torch.zeros(x.size(0), len(self.edge_embedding_lst))
        self_loop_attr[:, 0] = NUM_BOND_TYPE - 1
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = []
        for i in range(edge_attr.shape[1]):
            edge_embeddings.append(self.edge_embedding_lst[i](edge_attr[:, i]))
        if self.bond_feat_red == "mean":
            edge_embeddings = torch.stack(edge_embeddings).mean(dim=0)

        elif self.bond_feat_red == "sum":
            edge_embeddings = torch.stack(edge_embeddings).sum(dim=0)

        else:
            raise ValueError("Invalid bond feature reduction method. Please choose from 'mean' or 'sum'")


        x = self.weight_linear(x).view(-1, self.heads * self.emb_dim)
        return self.propagate(edge_index=edge_index, aggr=self.aggr, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j = x_j.view(-1, self.heads, self.emb_dim)
        x_i = x_i.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])
        return (x_j * alpha.view(-1, self.heads, 1)).mean(dim=1)

    def update(self, aggr_out):

        aggr_out = aggr_out + self.bias

        return aggr_out


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)


        self.last_attn = None

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None, return_attn=False):
        if return_attn:
            out, attn = scaled_dot_product_attention(
                self.q(query), self.k(key), self.v(value),
                query_mask, key_mask, mask,
                return_attn=True
            )
            self.last_attn = attn.detach()
            return out, attn
        else:
            out = scaled_dot_product_attention(
                self.q(query), self.k(key), self.v(value),
                query_mask, key_mask, mask
            )


            return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)


        self.last_attn = None

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None, return_attn=False):
        outs = []
        attns = []

        for h in self.heads:
            if return_attn:
                o, a = h(query, key, value, query_mask, key_mask, mask, return_attn=True)
                outs.append(o)
                attns.append(a.detach())
            else:
                outs.append(h(query, key, value, query_mask, key_mask, mask))

        x = torch.cat(outs, dim=-1)
        x = self.output_linear(x)

        if return_attn:

            self.last_attn = torch.stack(attns, dim=1)
            return x, self.last_attn


        if len(self.heads) > 0 and getattr(self.heads[0], "last_attn", None) is not None:
            self.last_attn = torch.stack([hh.last_attn for hh in self.heads], dim=1)

        return x


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, intermediate_size)
        self.linear_2 = nn.Linear(intermediate_size, hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
