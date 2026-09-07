import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from layers import GCNConv, GINConv, GATConv, MultiHeadAttention, FeedForward


NUM_ATOM_TYPE   = 65
NUM_DEGRESS_TYPE= 11
NUM_FORMCHRG_TYPE=5
NUM_HYBRIDTYPE  = 6
NUM_CHIRAL_TYPE = 3
NUM_AROMATIC_NUM= 2
NUM_VALENCE_TYPE= 7
NUM_Hs_TYPE     = 5
NUM_RS_TPYE     = 3

class DescriptorEncoder(nn.Module):
    def __init__(self, d_in, d_out=128, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_out), nn.GELU(),
            nn.Dropout(p),
            nn.Linear(d_out, d_out)
        )
    def forward(self, x):
        return self.net(x)


class FiLM_Legacy(nn.Module):
    def __init__(self, d_c, d_h):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_c, d_h),
            nn.GELU(),
            nn.Linear(d_h, 2*d_h)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h, c_node):
        d = self.mlp(c_node)
        d_gamma, beta = d.chunk(2, dim=-1)
        gamma = 1.0 + d_gamma
        return gamma * h + beta


class GraphEncoder_Legacy(nn.Module):
    def __init__(self,
                 gnum_layer, emb_dim,
                 gnn_type="gcn", gnn_aggr="add", bond_feat_red="mean",
                 JK="last", drop_ratio=0.1, node_readout="sum",
                 use_edge_head=True, edge_attr_dim=6,
                 use_film=True, d_descr_in=100, d_c=128,
                 film_pos="pre", use_late_fuse=True):
        super().__init__()
        assert gnum_layer >= 2
        self.gnum_layer = gnum_layer
        self.emb_dim = emb_dim
        self.JK = JK
        self.drop_ratio = drop_ratio
        self.node_readout = node_readout
        self.use_edge_head = use_edge_head
        self.edge_attr_dim = edge_attr_dim

        self.use_film = use_film and (d_descr_in > 0)
        self.d_c = d_c
        self.film_pos = film_pos
        self.use_late_fuse = use_late_fuse and self.use_film


        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_DEGRESS_TYPE, emb_dim)
        self.x_embedding3 = nn.Embedding(NUM_FORMCHRG_TYPE, emb_dim)
        self.x_embedding4 = nn.Embedding(NUM_HYBRIDTYPE, emb_dim)
        self.x_embedding5 = nn.Embedding(NUM_CHIRAL_TYPE, emb_dim)
        self.x_embedding6 = nn.Embedding(NUM_AROMATIC_NUM, emb_dim)
        self.x_embedding7 = nn.Embedding(NUM_VALENCE_TYPE, emb_dim)
        self.x_embedding8 = nn.Embedding(NUM_Hs_TYPE, emb_dim)
        self.x_embedding9 = nn.Embedding(NUM_RS_TPYE, emb_dim)

        self.x_embedding_list = [
            self.x_embedding1, self.x_embedding2, self.x_embedding3,
            self.x_embedding4, self.x_embedding5, self.x_embedding6,
            self.x_embedding7, self.x_embedding8, self.x_embedding9
        ]


        self.gnns = nn.ModuleList()
        for _ in range(gnum_layer):
            if gnn_type.lower() == "gcn":
                self.gnns.append(GCNConv(emb_dim, aggr=gnn_aggr, bond_feat_red=bond_feat_red))
            elif gnn_type.lower() == "gin":

                self.gnns.append(GINConv(emb_dim, aggr=gnn_aggr, bond_feat_red=bond_feat_red))
            elif gnn_type.lower() == "gat":
                self.gnns.append(GATConv(emb_dim, aggr=gnn_aggr, bond_feat_red=bond_feat_red))
            else:
                raise ValueError(gnn_type)

        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(gnum_layer)])


        if self.use_film:

            self.descr_proj = nn.Identity()

            self.descr_enc = DescriptorEncoder(d_in=d_descr_in, d_out=self.d_c, p=drop_ratio)

            self.films = nn.ModuleList([FiLM_Legacy(self.d_c, emb_dim) for _ in range(gnum_layer)])

        if self.use_edge_head:
            self.edge_head = nn.Sequential(
                nn.Linear(2*emb_dim + edge_attr_dim, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim)
            )

        if self.use_late_fuse:
            self.late_proj = nn.Sequential(
                nn.Linear(emb_dim + self.d_c, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim)
            )

    def forward(self, x, mol_index, edge_index, edge_attr, batch, x_cont=None, descr_vec=None):

        x_emb = torch.stack([emb(x[:, i]) for i, emb in enumerate(self.x_embedding_list)], dim=0)
        h = x_emb.mean(0) if self.node_readout == "mean" else x_emb.sum(0)

        c_graph, c_node = None, None
        if self.use_film and (descr_vec is not None):

            if descr_vec.dim() == 1:
                descr_vec = descr_vec.unsqueeze(0)
            if descr_vec.dim() == 3 and descr_vec.size(1) == 1:
                descr_vec = descr_vec.squeeze(1)

            if descr_vec.size(-1) != self.descr_enc.net[0].normalized_shape[0]:
                expect = self.descr_enc.net[0].normalized_shape[0]
                raise ValueError(f"expected descr_vec last dim={expect}, got {tuple(descr_vec.shape)}")

            B = int(batch.max().item()) + 1
            if descr_vec.size(0) == 1 and B > 1:
                descr_vec = descr_vec.expand(B, -1)
            if descr_vec.size(0) != B:
                raise RuntimeError(
                    f"[descr_vec batch mismatch] descr_vec={tuple(descr_vec.shape)} but batch implies B={B}")

            descr_vec = descr_vec.float()
            descr_in = self.descr_proj(descr_vec)
            c_graph = self.descr_enc(descr_in)
            c_node = c_graph[batch]


            bmax = int(batch.max().item())
            if bmax >= c_graph.size(0):
                raise RuntimeError(f"[batch oob] batch.max={bmax} >= c_graph.size(0)={c_graph.size(0)}")

            c_node = c_graph[batch]


        h_list = [h]
        for l in range(self.gnum_layer):
            h_in = h_list[-1]

            if c_node is not None and self.film_pos in ("pre", "both"):
                h_in = self.films[l](h_in, c_node)

            h_out = self.gnns[l](h_in, edge_index, edge_attr)

            if c_node is not None and self.film_pos in ("post", "both"):
                h_out = self.films[l](h_out, c_node)


            h_out = self.batch_norms[l](h_out)
            if l != self.gnum_layer - 1:
                h_out = F.relu(h_out)
            h_out = F.dropout(h_out, p=self.drop_ratio, training=self.training)

            h_list.append(h_out)

        if self.JK == "last":
            node_rep = h_list[-1]
        elif self.JK == "sum":
            node_rep = torch.stack(h_list, 0).sum(0)
        elif self.JK == "mean":
            node_rep = torch.stack(h_list, 0).mean(0)
        elif self.JK == "max":
            node_rep = torch.stack(h_list, 0).max(0)[0]
        elif self.JK == "concat":
            node_rep = torch.cat(h_list, dim=-1)
        else:
            raise ValueError(self.JK)

        edge_rep = None
        if self.use_edge_head:
            src, dst = edge_index
            edge_feat = torch.cat([node_rep[src], node_rep[dst], edge_attr.float()], dim=1)
            edge_rep = self.edge_head(edge_feat)

        graph_rep = global_mean_pool(node_rep, batch)
        graph_rep_fused = None
        if self.use_late_fuse and (c_graph is not None):
            graph_rep_fused = self.late_proj(torch.cat([graph_rep, c_graph], dim=1))

        return node_rep, edge_rep, graph_rep, graph_rep_fused, mol_index, batch


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size,intermediate_size,num_heads,hidden_dropout_prob):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size,num_heads)
        self.feed_forward = FeedForward(hidden_size,intermediate_size,hidden_dropout_prob)

    def forward(self, x, mask=None, return_attn=False):
        hidden_state = self.layer_norm_1(x)

        if return_attn:
            attn_out, attn = self.attention(
                hidden_state, hidden_state, hidden_state,

                mask=mask, return_attn=True
            )
            x = x + attn_out
        else:
            x = x + self.attention(hidden_state, hidden_state, hidden_state, mask=mask)

        x = x + self.feed_forward(self.layer_norm_2(x))

        if return_attn:
            return x, attn
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layer,hidden_size,intermediate_size,num_heads,hidden_dropout_prob):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(hidden_size=hidden_size,intermediate_size=intermediate_size,
                                    num_heads=num_heads,hidden_dropout_prob=hidden_dropout_prob) for _ in range(num_layer)])

    def forward(self, hidden_states, key_mask=None, query_mask=None, mask=None, return_attn=False):
        x = hidden_states
        attn_all = []

        for layer in self.layers:
            h = layer.layer_norm_1(x)

            if return_attn:
                attn_out, attn = layer.attention(
                    h, h, h,
                    query_mask=query_mask,
                    key_mask=key_mask,
                    mask=mask,
                    return_attn=True
                )
                attn_all.append(attn.detach())
            else:
                attn_out = layer.attention(
                    h, h, h,
                    query_mask=query_mask,
                    key_mask=key_mask,
                    mask=mask
                )

            x = x + attn_out
            x = x + layer.feed_forward(layer.layer_norm_2(x))

        if return_attn:

            return x, torch.stack(attn_all, dim=0)
        return x


class CenterHead(nn.Module):
    def __init__(self, hid=256):
        super().__init__()
        self.atom_head = nn.Linear(hid, 1)
        self.bond_mlp  = nn.Sequential(
            nn.Linear(2 * hid, hid), nn.GELU(), nn.Linear(hid, 1)
        )

    def forward(self, h, edge_index):
        atom_logits = self.atom_head(h).squeeze(-1)
        hi, hj = h[edge_index[0]], h[edge_index[1]]
        bond_logits = self.bond_mlp(torch.cat([hi, hj], dim=-1)).squeeze(-1)
        return atom_logits, bond_logits
