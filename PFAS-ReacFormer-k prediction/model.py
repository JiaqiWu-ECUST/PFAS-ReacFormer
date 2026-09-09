import math
from torch_geometric.utils import to_dense_batch
from layers import GCNConv, GINConv, GATConv, FeedForward
import torch.nn as nn
import torch
import torch.nn.functional as F
from mol2graphinfo import NUM_ATOM_TYPE,NUM_DEGRESS_TYPE,\
                   NUM_FORMCHRG_TYPE,NUM_HYBRIDTYPE,NUM_CHIRAL_TYPE,NUM_AROMATIC_NUM,\
                   NUM_VALENCE_TYPE,NUM_Hs_TYPE,NUM_RS_TPYE
from layer import GCNConv,GINConv,GATConv,MultiHeadAttention,FeedForward
from torch_geometric.nn import global_mean_pool

class DescriptorEncoder(nn.Module):
    def __init__(self, d_in, d_out=128, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_out), nn.GELU(),
            nn.Dropout(p),
            nn.Linear(d_out, d_out)
        )
    def forward(self, x):         # x: [B, d_in]
        return self.net(x)

class FiLM(nn.Module): 
    def __init__(self, d_c, d_h):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2*d_c, d_h), nn.GELU(), nn.Linear(d_h, 2*d_h))
        # 中性初始化：gamma≈1, beta≈0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    def forward(self, h, c_node):
        # h: [N, d_h]; c_node: [N, d_c]
        d = self.mlp(c_node)            # [N, 2*d_h]
        d_gamma, beta = d.chunk(2, dim=-1)
        gamma = 1.0 + d_gamma
        return gamma * h + beta

class GraphEncoder(nn.Module):
    def __init__(
        self,
        gnum_layer,
        emb_dim,
        gnn_aggr="add",
        bond_feat_red="mean",
        gnn_type='gcn',
        JK="last",
        drop_ratio=0.0,
        node_readout="sum",
        use_cont=True,
        cont_dim=8,
        use_edge_head=True,
        edge_attr_dim=6,
        use_film=True,
        d_descr_in=0,
        cond_dim=0,
        num_system=0,
        system_emb_dim=16,
        d_c=None,
        film_pos="pre",
        use_late_fuse=True,
    ):
        super().__init__()
        assert gnum_layer >= 2
        self.gnum_layer = gnum_layer
        self.emb_dim = emb_dim
        self.gnn_aggr = gnn_aggr
        self.gnn_type = gnn_type
        self.JK = JK
        self.drop_ratio = drop_ratio
        self.node_readout = node_readout
        self.use_cont = use_cont
        self.use_edge_head = use_edge_head
        self.edge_attr_dim = edge_attr_dim
        self.use_film = use_film and (d_descr_in > 0)
        self.cond_dim = cond_dim
        self.num_system = num_system
        self.film_pos = film_pos
        self.use_late_fuse = use_late_fuse and self.use_film
        self.d_c = d_c or emb_dim
        self.system_proj = nn.Linear(128, 64)

        
        emb_setting = (emb_dim,)
        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, *emb_setting)
        self.x_embedding2 = nn.Embedding(NUM_DEGRESS_TYPE, *emb_setting)
        self.x_embedding3 = nn.Embedding(NUM_FORMCHRG_TYPE, *emb_setting)
        self.x_embedding4 = nn.Embedding(NUM_HYBRIDTYPE, *emb_setting)
        self.x_embedding5 = nn.Embedding(NUM_CHIRAL_TYPE, *emb_setting)
        self.x_embedding6 = nn.Embedding(NUM_AROMATIC_NUM, *emb_setting)
        self.x_embedding7 = nn.Embedding(NUM_VALENCE_TYPE, *emb_setting)
        self.x_embedding8 = nn.Embedding(NUM_Hs_TYPE, *emb_setting)
        self.x_embedding9 = nn.Embedding(NUM_RS_TPYE, *emb_setting)
        for emb in [self.x_embedding1, self.x_embedding2, self.x_embedding3,
                    self.x_embedding4, self.x_embedding5, self.x_embedding6,
                    self.x_embedding7, self.x_embedding8, self.x_embedding9]:
            nn.init.xavier_uniform_(emb.weight)
        self.x_embedding_list = [
            self.x_embedding1, self.x_embedding2, self.x_embedding3,
            self.x_embedding4, self.x_embedding5, self.x_embedding6,
            self.x_embedding7, self.x_embedding8, self.x_embedding9
        ]

     
        if self.use_cont:
            self.proj = nn.Linear(emb_dim + cont_dim, emb_dim)

      
        self.gnns = nn.ModuleList()
        for _ in range(gnum_layer):
            if gnn_type.lower() == 'gcn':
                self.gnns.append(GCNConv(emb_dim, aggr=gnn_aggr, bond_feat_red=bond_feat_red))
            elif gnn_type.lower() == 'gin':
                self.gnns.append(GINConv(emb_dim, aggr=gnn_aggr, bond_feat_red=bond_feat_red))
            elif gnn_type.lower() == 'gat':
                self.gnns.append(GATConv(emb_dim, aggr=gnn_aggr, bond_feat_red=bond_feat_red))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
        self.batch_norms = nn.ModuleList(nn.BatchNorm1d(emb_dim) for _ in range(gnum_layer))

 
        self.use_film = use_film and (d_descr_in > 0)   
        self.descr_enc = None         
        self.films = nn.ModuleList(FiLM(d_c=self.d_c, d_h=emb_dim) for _ in range(gnum_layer))
        self.film_mlp = None       


        if self.use_edge_head:
            self.edge_head = nn.Sequential(
                nn.Linear(2 * emb_dim + edge_attr_dim, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim)
            )


        if self.use_late_fuse:
            self.late_proj = nn.Sequential(
                nn.Linear(emb_dim + self.d_c, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim)
            )


    def forward(self, x, mol_index, edge_index, edge_attr, batch, x_cont=None,
                descr_vec=None, cond_vec=None, system_id=None):
        device = x.device
        B = batch.max().item() + 1 if batch.numel() else 1

       
        x_emb = torch.stack([emb(x[:, i]) for i, emb in enumerate(self.x_embedding_list)]).mean(0)
        if self.use_cont and x_cont is not None:
            h = self.proj(torch.cat([x_emb, x_cont], dim=-1))
        else:
            h = x_emb

       
        film_parts = []
       
        if descr_vec is not None:
            if descr_vec.dim() == 1:
                descr_vec = descr_vec.unsqueeze(0).expand(B, -1)
            if self.descr_enc is None:
                d_in = descr_vec.shape[1]
                self.descr_enc = DescriptorEncoder(d_in=d_in, d_out=self.d_c, p=self.drop_ratio).to(device)
            film_parts.append(self.descr_enc(descr_vec))
 
        if cond_vec is not None:
            if cond_vec.dim() == 1:
                cond_vec = cond_vec.unsqueeze(0).expand(B, -1)
            film_parts.append(cond_vec)
   
        if system_id is not None and self.system_proj is not None:
            if system_id.dim() == 0:
                system_id = system_id.unsqueeze(0)
            if system_id.dim() == 1 and system_id.size(0) == 1:
                system_id = system_id.expand(B)
            # film_parts.append(self.system_emb(system_id))
            system_id = system_id.clamp(min=0) 
            max_id = 127 

            system_onehot = F.one_hot(system_id.long(), num_classes=max_id + 1).float()  # [B, 128]
            system_vec = self.system_proj(system_onehot)  # [B, 64]
            film_parts.append(system_vec)

     
        film_vec = torch.cat(film_parts, dim=1) if film_parts else None
        gamma = beta = None
        if film_vec is not None:
            if self.film_mlp is None:
                film_in_dim = film_vec.shape[1]
                self.film_mlp = nn.Sequential(
                    nn.LayerNorm(film_in_dim),
                    nn.Linear(film_in_dim, 2 * self.emb_dim)
                ).to(device)
            film_out = self.film_mlp(film_vec)
            gamma, beta = film_out.chunk(2, dim=1)


        h_list = [h]
        for layer in range(self.gnum_layer):
            h_in = h_list[-1]
            if self.use_film and self.film_pos in ('pre', 'both'):
                g = gamma[batch] if gamma is not None else None
                b = beta[batch] if beta is not None else None
                c_node = torch.cat([g, b], dim=1) if g is not None else None
                h_in = self.films[layer](h_in, c_node)
            h_out = self.gnns[layer](h_in, edge_index, edge_attr)
            h_out = F.relu(h_out)
            h_out = F.dropout(h_out, p=self.drop_ratio, training=self.training)
            if self.use_film and self.film_pos in ('post', 'both'):
                g = gamma[batch] if gamma is not None else None
                b = beta[batch] if beta is not None else None
                c_node = torch.cat([g, b], dim=1) if g is not None else None
                h_out = self.films[layer](h_out, c_node)
            h_list.append(h_out)


        if self.JK == 'last':
            node_rep = h_list[-1]
        elif self.JK == 'concat':
            node_rep = torch.cat(h_list, dim=1)
        elif self.JK == 'max':
            node_rep = torch.stack(h_list, dim=0).max(dim=0)[0]
        elif self.JK == 'sum':
            node_rep = torch.stack(h_list, dim=0).sum(0)
        elif self.JK == 'mean':
            node_rep = torch.stack(h_list, dim=0).mean(0)
        elif self.JK == 'last+first':
            node_rep = h_list[-1] + h_list[0]
        else:
            raise ValueError(self.JK)


        edge_rep = None
        if self.use_edge_head:
            src, dst = edge_index
            edge_feat = torch.cat([node_rep[src], node_rep[dst], edge_attr.float()], dim=1)
            edge_rep = self.edge_head(edge_feat)
        graph_rep = global_mean_pool(node_rep, batch)
        graph_rep_fused = None
        if self.use_late_fuse and gamma is not None:
            graph_rep_fused = self.late_proj(torch.cat([graph_rep, gamma], dim=1))

        return node_rep, edge_rep, graph_rep, graph_rep_fused, mol_index, batch

class Graph2VecEncoder(nn.Module):
    def __init__(self, graph_enc, trans_enc=None, pool="mean"):
        super().__init__()
        self.graph_enc = graph_enc
        self.trans_enc = trans_enc
        self.pool = pool

    def forward(self, batch, descr_vec=None, cond_vec=None, system_id=None, return_attn=False):
        node_rep, _, _, graph_rep_fused, _, node_batch = self.graph_enc(
            x=batch.x,
            mol_index=batch.batch,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch,
            x_cont=None,
            descr_vec=descr_vec,
            cond_vec=cond_vec,
            system_id=system_id,
        )

        if self.trans_enc is not None:
            memory, valid_mask = to_dense_batch(node_rep, node_batch)
            if return_attn:
                memory, self_attn = self.trans_enc(memory, mask=valid_mask, return_attn=True)
            else:
                memory = self.trans_enc(memory, mask=valid_mask, return_attn=False)
                self_attn = None

            m = valid_mask.unsqueeze(-1).float()
            graph_vec = (memory * m).sum(1) / (m.sum(1).clamp_min(1.0))
        else:
            graph_vec = graph_rep_fused
            self_attn = None

        if return_attn:
            return graph_vec, self_attn, valid_mask
        return graph_vec

class PFASGraphRegressor(nn.Module):
    def __init__(self, encoder, d_descr_in, d_model, use_mol_desc=True,
                 head_hidden=256, head_dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.use_mol_desc = use_mol_desc

        self.cond_proj = None
        if use_mol_desc:
            self.cond_proj = nn.Sequential(
                nn.LayerNorm(d_descr_in),
                nn.Linear(d_descr_in, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

        self.head = RegressorHead(d_model, hidden=head_hidden, dropout=head_dropout)

    def forward(self, batch, descr_vec=None, cond_vec=None, system_id=None):
        if descr_vec is not None and descr_vec.dim() == 3 and descr_vec.size(1) == 1:
            descr_vec = descr_vec.squeeze(1)

        graph_vec = self.encoder(batch, descr_vec=descr_vec, cond_vec=cond_vec, system_id=system_id)

    
        if graph_vec is None:
            graph_vec = torch.zeros(
                batch.num_graphs if hasattr(batch, "num_graphs") else batch.x.size(0),
                self.head.net[0].normalized_shape[0],
                device=batch.x.device,
            )

        if self.use_mol_desc and (self.cond_proj is not None):
            if descr_vec is None:
                descr_vec = torch.zeros(
                    graph_vec.size(0),
                    self.cond_proj[0].normalized_shape[0],
                    device=graph_vec.device,
                    dtype=graph_vec.dtype
                )
            graph_vec = graph_vec + self.cond_proj(descr_vec)

        pred = self.head(graph_vec)
        return pred

class RegressorHead(nn.Module):
    def __init__(self, d_in, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v, mask=None, return_attn=False):

        B, Lq, D = q.shape
        _, Lk, _ = k.shape

        q = self.q_proj(q).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)


        if mask is not None:

            if mask.dim() == 2:

                key_mask = mask[:, None, None, :].to(dtype=torch.bool)
                scores = scores.masked_fill(~key_mask, float("-inf"))

            elif mask.dim() == 3:
                attn_mask = mask[:, None, :, :].to(dtype=torch.bool)
                scores = scores.masked_fill(~attn_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Lq, D)
        out = self.proj_drop(self.out_proj(out))

        if return_attn:
            return out, attn
        return out


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
            x = x + self.feed_forward(self.layer_norm_2(x))
            return x, attn
        else:
            x = x + self.attention(hidden_state, hidden_state, hidden_state, mask=mask)
            x = x + self.feed_forward(self.layer_norm_2(x))
            return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layer,hidden_size,intermediate_size,num_heads,hidden_dropout_prob):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(hidden_size=hidden_size,intermediate_size=intermediate_size,
                                    num_heads=num_heads,hidden_dropout_prob=hidden_dropout_prob) for _ in range(num_layer)])


    def forward(self, x, mask=None, return_attn=False):
        if not return_attn:
            for layer in self.layers:
                x = layer(x, mask=mask)
            return x

        attn_list = []
        for layer in self.layers:
            x, attn = layer(x, mask=mask, return_attn=True)
            attn_list.append(attn)
        return x, attn_list

