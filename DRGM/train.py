import os, sys, json, logging, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader as PyG_DataLoader
from tensorboardX import SummaryWriter
from addict import Dict


from model import GraphEncoder_Legacy as GraphEncoder, TransformerEncoder, CenterHead


def get_lr(optimizer):
    for pg in optimizer.param_groups:
        return pg['lr']


def grad_norm(model):
    total = 0.
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def get_linear_scheduler_with_warmup(optimizer, num_warmup, total_steps):
    def lr_lambda(step):
        if step < num_warmup:
            return float(step) / float(max(1, num_warmup))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - num_warmup)))

    return LambdaLR(optimizer, lr_lambda)


class ReactionCenterDataset(Dataset):
    def __init__(self, data_file, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.data, self.slices = torch.load(data_file, weights_only=False)
        self._idx_list = None

    def len(self):
        return len(self._idx_list) if self._idx_list is not None else len(self.slices['x']) - 1

    def get(self, idx):
        orig = self._idx_list[idx] if self._idx_list is not None else idx
        x_s, x_e = self.slices['x'][orig], self.slices['x'][orig + 1]
        e_s, e_e = self.slices['edge_index'][orig], self.slices['edge_index'][orig + 1]
        data_dict = {
            'x': self.data['x'][x_s:x_e],
            'y_atom': self.data['y_atom'][x_s:x_e],
            'y_bond': self.data['y_bond'][e_s:e_e],
            'edge_index': self.data['edge_index'][:, e_s:e_e],
            'edge_attr': self.data['edge_attr'][e_s:e_e],
            'mol_index': torch.tensor([orig])
        }

        if 'mol_desc' in self.data:
            s0 = int(self.slices['mol_desc'][orig])
            s1 = int(self.slices['mol_desc'][orig + 1])
            data_dict['mol_desc'] = self.data['mol_desc'][s0:s1].unsqueeze(0)


        return Data(**data_dict)


    def set_indices(self, indices):
        self._idx_list = indices


def reaction_center_collate_fn(batch):

    collated = {}


    for key in ['x', 'y_atom', 'y_bond', 'edge_attr']:
        if hasattr(batch[0], key):
            collated[key] = torch.cat([getattr(item, key) for item in batch], 0)


    edge_index_list, offset = [], 0
    for item in batch:

        off = torch.tensor(offset, dtype=item.edge_index.dtype, device=item.edge_index.device)
        edge_index_list.append(item.edge_index + off)
        offset += item.x.shape[0]
    collated['edge_index'] = torch.cat(edge_index_list, 1)


    collated['mol_index'] = torch.cat([item.mol_index for item in batch], 0)
    num_nodes_per_graph = [item.x.shape[0] for item in batch]
    batch_vec = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(num_nodes_per_graph))
    collated['batch'] = batch_vec


    if hasattr(batch[0], 'mol_desc'):
        collated['mol_desc'] = torch.cat([item.mol_desc for item in batch], 0)


    max_id = collated['edge_index'].max().item()
    min_id = collated['edge_index'].min().item()
    num_nodes = collated['x'].shape[0]

    if max_id >= num_nodes or min_id < 0:
        raise RuntimeError(f"edge_index 越界: max={max_id}, min={min_id}, num_nodes={num_nodes}")

    return collated


class ReactionCenterModel(nn.Module):
    def __init__(self, graph_enc, trans_enc=None):
        super().__init__()
        self.graph_enc = graph_enc
        self.trans_enc = trans_enc
        self.center_head = CenterHead(hid=graph_enc.emb_dim)

    def forward(self, x, mol_index, edge_index, edge_attr, batch, x_cont=None, descr_vec=None):

      
        if descr_vec is not None:

            if descr_vec.dim() == 3 and descr_vec.size(1) == 1:
                descr_vec = descr_vec.squeeze(1)
            elif descr_vec.dim() == 1:
                descr_vec = descr_vec.unsqueeze(0)


        node_rep, edge_rep, graph_rep, graph_rep_fused, mol_index, batch = self.graph_enc(
            x, mol_index, edge_index, edge_attr, batch, x_cont, descr_vec
        )


        if self.trans_enc is not None:

            from torch_geometric.utils import to_dense_batch
            node_rep_seq, mask = to_dense_batch(node_rep, batch)


            node_rep_seq, attn_all = self.trans_enc(
                node_rep_seq,
                key_mask=mask,
                query_mask=mask,
                return_attn=True
            )


            self.last_attn_all = attn_all
            node_rep = node_rep_seq[mask]

        atom_logits, bond_logits = self.center_head(node_rep, edge_index)
        return atom_logits, bond_logits


class FocalBCEWithLogitsLoss(nn.Module):
   
    def __init__(self, gamma_pos=0.5, gamma_neg=2.0):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, logits, targets, pos_weight=None, alpha=None, sample_mask=None, eps=1e-6):

        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='none',
            pos_weight=pos_weight
        )
        p = torch.sigmoid(logits)
        pt = torch.where(targets.bool(), p, 1 - p)


        gamma = torch.where(targets.bool(),
                            torch.as_tensor(self.gamma_pos, device=logits.device, dtype=logits.dtype),
                            torch.as_tensor(self.gamma_neg, device=logits.device, dtype=logits.dtype))
        focal_factor = (1.0 - pt).clamp_min(eps).pow(gamma)


        if alpha is None:
            alpha_t = torch.where(targets.bool(),
                                  torch.as_tensor(0.5, device=logits.device, dtype=logits.dtype),
                                  torch.as_tensor(0.5, device=logits.device, dtype=logits.dtype))
        else:

            alpha_t = torch.where(targets.bool(),
                                  torch.as_tensor(alpha, device=logits.device, dtype=logits.dtype),
                                  torch.as_tensor(1.0 - alpha, device=logits.device, dtype=logits.dtype))

        loss = alpha_t * focal_factor * bce

        if sample_mask is not None:
            loss = loss * sample_mask


        if sample_mask is not None:
            denom = sample_mask.sum()
        else:

            denom = torch.tensor(loss.numel(), device=logits.device, dtype=loss.dtype)
        denom = denom.clamp_min(1)


        return loss.sum() / denom


class CombinedFocalLoss(nn.Module):
  
    def __init__(self,
                 lambda_bond=10.0,
                 atom_pos_weight=None, bond_pos_weight=None,
                 atom_alpha=None, bond_alpha=None,
                 gamma_pos=0.5, gamma_neg=2.0):
        super().__init__()
        self.lambda_bond = float(lambda_bond)
        self.atom_pos_weight = atom_pos_weight
        self.bond_pos_weight = bond_pos_weight
        self.atom_alpha = atom_alpha
        self.bond_alpha = bond_alpha

        self.atom_crit = FocalBCEWithLogitsLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg)
        self.bond_crit = FocalBCEWithLogitsLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg)

    @torch.no_grad()
    def _estimate_pos_weight(self, targets):

        pos = targets.sum().float()
        tot = torch.numel(targets)
        neg = tot - pos
        pw = (neg / pos.clamp_min(1.0)).to(targets.device)
        return pw

    def forward(self, atom_logits, bond_logits, atom_targets, bond_targets,
                atom_mask=None, bond_mask=None):

        atom_pw = (self.atom_pos_weight.to(atom_logits.device) if torch.is_tensor(self.atom_pos_weight)
                   else self._estimate_pos_weight(atom_targets))
        bond_pw = (self.bond_pos_weight.to(bond_logits.device) if torch.is_tensor(self.bond_pos_weight)
                   else self._estimate_pos_weight(bond_targets))

        atom_loss = self.atom_crit(
            atom_logits, atom_targets,
            pos_weight=atom_pw, alpha=self.atom_alpha, sample_mask=atom_mask
        )
        bond_loss = self.bond_crit(
            bond_logits, bond_targets,
            pos_weight=bond_pw, alpha=self.bond_alpha, sample_mask=bond_mask
        )

        total_loss = atom_loss + self.lambda_bond * bond_loss
        return total_loss, atom_loss, bond_loss


class ReactionCenterTrainer:

    def __init__(self, config):
        self.cfg = Dict(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        self.save_dir = f"{self.cfg.model.save_dir}/{datetime.datetime.now():%Y%m%d_%H%M%S}"
        os.makedirs(f"{self.save_dir}/log", exist_ok=True)
        os.makedirs(f"{self.save_dir}/model", exist_ok=True)


        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        file_handler = logging.FileHandler(f"{self.save_dir}/log/training.log", mode='w', encoding='utf-8')
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter);
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler);
        logger.addHandler(stream_handler)
        logging.info(f"======  Using device: {self.device}  ======")
        logging.info(f"======  Save dir: {self.save_dir}  ======")


        graph_enc = GraphEncoder(
            gnum_layer=self.cfg.model.gnn_num_layer,
            emb_dim=self.cfg.model.emb_dim,
            gnn_type=self.cfg.model.gnn_type,
            gnn_aggr=self.cfg.model.gnn_aggr,
            JK=self.cfg.model.gnn_jk,
            drop_ratio=self.cfg.model.drop_ratio,
            node_readout=self.cfg.model.node_readout,

            use_film=self.cfg.model.use_film,
            d_descr_in=self.cfg.model.d_descr_in,
            use_edge_head=self.cfg.model.use_edge_head,
            edge_attr_dim=self.cfg.model.edge_attr_dim)


        trans_enc = TransformerEncoder(
            num_layer=self.cfg.model.trans_num_layer,
            hidden_size=self.cfg.model.emb_dim,
            intermediate_size=self.cfg.model.trans_intermediate_size,
            num_heads=self.cfg.model.num_heads,
            hidden_dropout_prob=self.cfg.model.drop_ratio) if self.cfg.model.use_transformer else None

        self.model = ReactionCenterModel(graph_enc, trans_enc).to(self.device)

        lr = float(self.cfg.optimizer.learning_rate)
        wd = float(self.cfg.optimizer.weight_decay)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=wd
        )
        logging.info(f"Optimizer AdamW: lr={lr} (type={type(lr)}), weight_decay={wd}")


        self.dataset = ReactionCenterDataset(self.cfg.data.data_path)
        total = len(self.dataset)
        if hasattr(self.cfg.data, 'split_file') and self.cfg.data.split_file:
            if self.load_data_split(self.cfg.data.split_file):
                logging.info("使用保存的数据集划分")
            else:
                logging.info("创建新的数据集划分")
                self.create_new_split(total)
                self.save_data_split()
        else:
            logging.info("创建新的数据集划分")
            self.create_new_split(total)
            self.save_data_split()

        self.train_loader = PyG_DataLoader(
            self.train_set, batch_size=self.cfg.data.batch_size,
            shuffle=True, collate_fn=reaction_center_collate_fn
        )
        self.valid_loader = PyG_DataLoader(
            self.valid_set, batch_size=self.cfg.data.batch_size,
            shuffle=False, collate_fn=reaction_center_collate_fn
        )
        self.test_loader = PyG_DataLoader(
            self.test_set, batch_size=self.cfg.data.batch_size,
            shuffle=False, collate_fn=reaction_center_collate_fn
        )


        import math
        self.accum = int(getattr(self.cfg.training, "accum", 1))
        steps_per_epoch = math.ceil(len(self.train_loader) / max(1, self.accum))
        total_steps = max(1, self.cfg.training.epoch * steps_per_epoch)
        warmup_steps = int(getattr(self.cfg.scheduler, 'warmup_step', 0))

        warmup_steps = min(warmup_steps, max(1, total_steps - 1))

        self.scheduler = get_linear_scheduler_with_warmup(
            self.optimizer,
            num_warmup=warmup_steps,
            total_steps=total_steps
        )
        logging.info(f"LR Scheduler: linear_warmup, total_steps={total_steps}, warmup={warmup_steps}, "
                     f"steps_per_epoch={steps_per_epoch}, accum={self.accum}")


        self.global_atom_pw, self.global_bond_pw = self._compute_global_pos_weights()
        logging.info(f"Global pos_weight: atom={self.global_atom_pw.item():.3f}, bond={self.global_bond_pw.item():.3f}")


        if getattr(self.cfg.training, 'use_focal_loss', True):
            gamma_pos = getattr(self.cfg.training, 'focal_gamma_pos', 0.5)
            gamma_neg = getattr(self.cfg.training, 'focal_gamma_neg', 2.0)
            lambda_bond = float(getattr(self.cfg.training, 'lambda_bond', 10.0))
            atom_alpha = getattr(self.cfg.training, 'atom_alpha', None)
            bond_alpha = getattr(self.cfg.training, 'bond_alpha', None)

            self.criterion = CombinedFocalLoss(
                lambda_bond=lambda_bond,
                atom_pos_weight=self.global_atom_pw,
                bond_pos_weight=self.global_bond_pw,
                atom_alpha=atom_alpha,
                bond_alpha=bond_alpha,
                gamma_pos=gamma_pos,
                gamma_neg=gamma_neg
            )
            logging.info(f"使用 Focal+BCE: gamma_pos={gamma_pos}, gamma_neg={gamma_neg}, "
                         f"lambda_bond={lambda_bond}")
        else:
            self.atom_crit = nn.BCEWithLogitsLoss(pos_weight=self.global_atom_pw)
            self.bond_crit = nn.BCEWithLogitsLoss(pos_weight=self.global_bond_pw)
            logging.info("使用 BCE Loss（含全局pos_weight）")


        self.best_t_atom = 0.5
        self.best_t_bond = 0.1


        logging.info("=== 检查数据分割 ===")
        has_leakage = self.check_data_leakage()
        if has_leakage:
            logging.error(" 发现数据泄露！建议停止训练并修复数据分割问题")
        else:
            logging.info(" 数据分割正常")


        self.writer = SummaryWriter(log_dir=f"{self.save_dir}/log")
        logging.info("Trainer initialized successfully")

    def _get_descr(self, batch):
        d = batch.get("mol_desc")
        if d is None:
            return None

        return d.squeeze(1) if (d.dim() == 3 and d.size(1) == 1) else d

    def _make_atom_mask(self, y_atom: torch.Tensor):
      
        y = y_atom
        if y.dim() > 1:
            y = y.view(-1)

        mask = torch.ones_like(y, dtype=torch.float32)


        soft = (y > 0.0) & (y < 0.5)
        soft_w = float(getattr(self.cfg.training, "soft_atom_weight", 0.2))
        mask[soft] = soft_w

        return mask

    @torch.no_grad()
    def _compute_global_pos_weights(self):
      
        pos_a = neg_a = pos_b = neg_b = 0

        for batch in self.train_loader:

            y_atom = batch['y_atom']
            y_bond = batch['y_bond']


            pa = (y_atom >= 0.5).sum().item()
            na = y_atom.numel() - pa
            pos_a += pa;
            neg_a += na


            pb = (y_bond >= 0.5).sum().item()
            nb = y_bond.numel() - pb
            pos_b += pb;
            neg_b += nb


        atom_pw = (neg_a / max(1.0, pos_a)) if pos_a > 0 else 1.0
        bond_pw = (neg_b / max(1.0, pos_b)) if pos_b > 0 else 1.0

        atom_pw = torch.tensor([atom_pw], device=self.device, dtype=torch.float32).clamp_min(1.0)
        bond_pw = torch.tensor([bond_pw], device=self.device, dtype=torch.float32).clamp_min(1.0)
        return atom_pw, bond_pw


    def _precision_recall_f1(self, y_true, y_score, t):
        y_pred = (y_score >= t)
        tp = np.logical_and(y_pred, y_true).sum()
        fp = np.logical_and(y_pred, ~y_true).sum()
        fn = np.logical_and(~y_pred, y_true).sum()
        P = tp / (tp + fp + 1e-12)
        R = tp / (tp + fn + 1e-12)
        F1 = 2 * P * R / (P + R + 1e-12)
        return P, R, F1

    def _find_best_threshold(self, y_true, y_score, metric="f1", grid=None):
        if grid is None:
            grid = np.linspace(0.02, 0.5, 50)
        best_t, best_score, best_triplet = 0.5, -1, (0, 0, 0)
        y_true = y_true.astype(bool)
        for t in grid:
            P, R, F1 = self._precision_recall_f1(y_true, y_score, t)
            s = {"f1": F1, "recall": R, "precision": P}[metric]
            if s > best_score:
                best_t, best_score, best_triplet = t, s, (P, R, F1)
        return best_t, best_triplet

    def calculate_model_score(self, atom_f1, bond_f1, atom_recall, bond_recall, atom_precision, bond_precision):
     
        f1_score = 0.4 * atom_f1 + 0.2 * bond_f1


        recall_score = 0.15 * atom_recall + 0.15 * bond_recall


        precision_score = 0.05 * atom_precision + 0.05 * bond_precision

        return f1_score + recall_score + precision_score

    def check_data_leakage(self):
      
        train_indices = set(self.train_set.indices)
        valid_indices = set(self.valid_set.indices)
        overlap = train_indices.intersection(valid_indices)

        logging.info(f"训练集大小: {len(train_indices)}")
        logging.info(f"验证集大小: {len(valid_indices)}")
        logging.info(f"数据重叠: {len(overlap)} 个样本")

        if len(overlap) > 0:
            logging.error(" 发现数据泄露！训练集和验证集有重叠！")

            overlap_list = list(overlap)
            logging.error(f"重叠样本索引 (前10个): {overlap_list[:10]}{'...' if len(overlap_list) > 10 else ''}")
            return True
        else:
            logging.info(" 没有发现数据泄露")
            return False

    def save_data_split(self):
       
        split_info = {
            'train_indices': self.train_set.indices.tolist(),
            'valid_indices': self.valid_set.indices.tolist(),
            'test_indices': self.test_set.indices.tolist(),
            'seed': self.cfg.data.seed,
            'total_size': len(self.dataset),
            'timestamp': datetime.datetime.now().isoformat()
        }

        split_file = f'{self.save_dir}/data_split.json'
        with open(split_file, 'w') as f:
            json.dump(split_info, f, indent=2)

        logging.info(f" 数据集划分已保存到: {split_file}")

    def load_data_split(self, split_file):
       
        if not os.path.exists(split_file):
            logging.warning(f"划分文件不存在: {split_file}")
            return False

        with open(split_file, 'r') as f:
            split_info = json.load(f)


        self.train_set = torch.utils.data.Subset(self.dataset, split_info['train_indices'])
        self.valid_set = torch.utils.data.Subset(self.dataset, split_info['valid_indices'])
        self.test_set = torch.utils.data.Subset(self.dataset, split_info['test_indices'])

        logging.info(f" 数据集划分已从 {split_file} 加载")
        return True

    def create_new_split(self, total):
      
        train_end = int(total * self.cfg.data.train_ratio)
        valid_end = train_end + int(total * self.cfg.data.valid_ratio)
        test_end = valid_end + int(total * self.cfg.data.test_ratio)
        indices = np.random.default_rng(self.cfg.data.seed).permutation(total)

        self.train_set = torch.utils.data.Subset(self.dataset, indices[:train_end])
        self.valid_set = torch.utils.data.Subset(self.dataset, indices[train_end:valid_end])
        self.test_set = torch.utils.data.Subset(self.dataset, indices[valid_end:test_end])


    def calculate_detailed_metrics_from_stats(self, metrics, name):
      
        tp, fp, fn, tn = metrics['tp'], metrics['fp'], metrics['fn'], metrics['tn']
        total = tp + fp + fn + tn

        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'balanced_accuracy': (recall + specificity) / 2,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'support': total,
            'positive_ratio': (tp + fn) / total if total > 0 else 0
        }

    def log_detailed_metrics(self, atom_metrics, bond_metrics, dataset_name, epoch):
      

        logging.info(f"=== {dataset_name} 详细指标 ===")
        logging.info(f"原子 - Acc: {atom_metrics['accuracy']:.4f}, "
                     f"Precision: {atom_metrics['precision']:.4f}, Recall: {atom_metrics['recall']:.4f}, "
                     f"F1: {atom_metrics['f1']:.4f}, Spec: {atom_metrics['specificity']:.4f},"
                     f"Balanced Acc: {atom_metrics['balanced_accuracy']:.4f}")
        logging.info(f"      TP: {atom_metrics['tp']}, FP: {atom_metrics['fp']}, "
                     f"FN: {atom_metrics['fn']}, TN: {atom_metrics['tn']},"
                     f"正样本比例: {atom_metrics['positive_ratio']:.4f}")

        logging.info(f"键   - Acc: {bond_metrics['accuracy']:.4f}, "
                     f"Precision: {bond_metrics['precision']:.4f}, Recall: {bond_metrics['recall']:.4f}, "
                     f"F1: {bond_metrics['f1']:.4f}, Spec: {bond_metrics['specificity']:.4f},"
                     f"Balanced Acc: {bond_metrics['balanced_accuracy']:.4f}")
        logging.info(f"      TP: {bond_metrics['tp']}, FP: {bond_metrics['fp']}, "
                     f"FN: {bond_metrics['fn']}, TN: {bond_metrics['tn']},"
                     f"正样本比例: {bond_metrics['positive_ratio']:.4f}")


        if hasattr(self, 'writer'):

            self.writer.add_scalar(f'{dataset_name}/atom_accuracy', atom_metrics['accuracy'], epoch)
            self.writer.add_scalar(f'{dataset_name}/atom_precision', atom_metrics['precision'], epoch)
            self.writer.add_scalar(f'{dataset_name}/atom_recall', atom_metrics['recall'], epoch)
            self.writer.add_scalar(f'{dataset_name}/atom_f1', atom_metrics['f1'], epoch)
            self.writer.add_scalar(f'{dataset_name}/atom_specificity', atom_metrics['specificity'], epoch)
            self.writer.add_scalar(f'{dataset_name}/atom_balanced_accuracy', atom_metrics['balanced_accuracy'], epoch)


            self.writer.add_scalar(f'{dataset_name}/bond_accuracy', bond_metrics['accuracy'], epoch)
            self.writer.add_scalar(f'{dataset_name}/bond_precision', bond_metrics['precision'], epoch)
            self.writer.add_scalar(f'{dataset_name}/bond_recall', bond_metrics['recall'], epoch)
            self.writer.add_scalar(f'{dataset_name}/bond_f1', bond_metrics['f1'], epoch)
            self.writer.add_scalar(f'{dataset_name}/bond_specificity', bond_metrics['specificity'], epoch)
            self.writer.add_scalar(f'{dataset_name}/bond_balanced_accuracy', bond_metrics['balanced_accuracy'], epoch)


    def train_epoch(self):
        self.model.train()
        total_loss, atom_loss_sum, bond_loss_sum = 0.0, 0.0, 0.0
        atom_acc_sum, bond_acc_sum = 0.0, 0.0


        atom_metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
        bond_metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}


        self.accum = int(getattr(self, "accum", getattr(self.cfg.training, "accum", 1)))
        self.accum = max(1, self.accum)


        t_atom = float(getattr(self, "best_t_atom", 0.5))
        t_bond = float(getattr(self, "best_t_bond", 0.1))


        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            atom_logits, bond_logits = self.model(
                batch['x'], batch['mol_index'], batch['edge_index'],
                batch['edge_attr'], batch['batch'], descr_vec=self._get_descr(batch)
            )


            atom_mask = self._make_atom_mask(batch['y_atom']).to(self.device)

            loss, atom_loss, bond_loss = self.criterion(
                atom_logits, bond_logits,
                batch['y_atom'], batch['y_bond'],
                atom_mask=atom_mask,
                bond_mask=None
            )

            total_loss += float(loss.item())
            atom_loss_sum += float(atom_loss.item())
            bond_loss_sum += float(bond_loss.item())


            atom_prob = torch.sigmoid(atom_logits)
            bond_prob = torch.sigmoid(bond_logits)


            src, dst = batch['edge_index'][0], batch['edge_index'][1]
            gate = torch.sqrt(
                torch.clamp(atom_prob[src], 0.0, 1.0) * torch.clamp(atom_prob[dst], 0.0, 1.0)
            )
            bond_prob_adj = bond_prob * gate


            atom_pred = (atom_prob >= t_atom).float()
            bond_pred = (bond_prob_adj >= t_bond).float()


            y_bond = batch['y_bond']
            y_atom = batch['y_atom'].view(-1)


            eval_mask = (y_atom == 0) | (y_atom >= 0.5)

            y_atom_bin = (y_atom >= 0.5).float()
            atom_pred_ = atom_pred.view(-1)


            atom_pred_m = atom_pred_[eval_mask]
            y_atom_m = y_atom_bin[eval_mask]
            atom_metrics['tp'] += ((atom_pred_m == 1) & (y_atom_m == 1)).sum().item()
            atom_metrics['fp'] += ((atom_pred_m == 1) & (y_atom_m == 0)).sum().item()
            atom_metrics['fn'] += ((atom_pred_m == 0) & (y_atom_m == 1)).sum().item()
            atom_metrics['tn'] += ((atom_pred_m == 0) & (y_atom_m == 0)).sum().item()

            bond_metrics['tp'] += ((bond_pred == 1) & (y_bond == 1)).sum().item()
            bond_metrics['fp'] += ((bond_pred == 1) & (y_bond == 0)).sum().item()
            bond_metrics['fn'] += ((bond_pred == 0) & (y_bond == 1)).sum().item()
            bond_metrics['tn'] += ((bond_pred == 0) & (y_bond == 0)).sum().item()


            atom_acc = (atom_pred_m == y_atom_m).float().mean()
            bond_acc = (bond_pred == y_bond).float().mean()
            atom_acc_sum += float(atom_acc.item())
            bond_acc_sum += float(bond_acc.item())


            scaled_loss = loss / float(self.accum)
            scaled_loss.backward()

            do_update = ((step + 1) % self.accum == 0)

            if do_update:
                clip = float(getattr(self.cfg.training, "clip_norm", 0.0) or 0.0)
                if clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                if hasattr(self, "scheduler") and self.scheduler is not None:
                    self.scheduler.step()


            if (step + 1) % self.cfg.training.log_iter_step == 0:
                logging.info(
                    f'Step {step + 1}  loss={loss.item():.4f}  '
                    f'atom_acc={atom_acc.item():.4f}  bond_acc={bond_acc.item():.4f}  '
                    f'(t_atom={t_atom:.3f}, t_bond={t_bond:.3f}, '
                    f'accum={self.accum}, lr={get_lr(self.optimizer):.6g})'
                )

                global_step = len(self.train_loader) * (self.current_epoch - 1) + step
                self.writer.add_scalar('Loss/train_step', float(loss.item()), global_step=global_step)
                self.writer.add_scalar('Accuracy/train_atom_step', float(atom_acc.item()), global_step=global_step)
                self.writer.add_scalar('Accuracy/train_bond_step', float(bond_acc.item()), global_step=global_step)
                self.writer.add_scalar('Learning_Rate', get_lr(self.optimizer), global_step=global_step)


        total_batches = len(self.train_loader)
        if total_batches > 0 and (total_batches % self.accum != 0):
            clip = float(getattr(self.cfg.training, "clip_norm", 0.0) or 0.0)
            if clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if hasattr(self, "scheduler") and self.scheduler is not None:
                self.scheduler.step()


        n = max(1, len(self.train_loader))
        epoch_loss = total_loss / n
        epoch_atom_acc = atom_acc_sum / n
        epoch_bond_acc = bond_acc_sum / n

        atom_detailed = self.calculate_detailed_metrics_from_stats(atom_metrics, "Atom")
        bond_detailed = self.calculate_detailed_metrics_from_stats(bond_metrics, "Bond")
        self.log_detailed_metrics(atom_detailed, bond_detailed, "Train", self.current_epoch)


        self.writer.add_scalar('Loss/train_epoch', epoch_loss, global_step=self.current_epoch)
        self.writer.add_scalar('Accuracy/train_atom_epoch', epoch_atom_acc, global_step=self.current_epoch)
        self.writer.add_scalar('Accuracy/train_bond_epoch', epoch_bond_acc, global_step=self.current_epoch)
        self.writer.add_scalar('F1/train_atom', atom_detailed['f1'], global_step=self.current_epoch)
        self.writer.add_scalar('F1/train_bond', bond_detailed['f1'], global_step=self.current_epoch)
        self.writer.add_scalar('Precision/train_atom', atom_detailed['precision'], global_step=self.current_epoch)
        self.writer.add_scalar('Precision/train_bond', bond_detailed['precision'], global_step=self.current_epoch)
        self.writer.add_scalar('Recall/train_atom', atom_detailed['recall'], global_step=self.current_epoch)
        self.writer.add_scalar('Recall/train_bond', bond_detailed['recall'], global_step=self.current_epoch)
        self.writer.add_scalar('Specificity/train_atom', atom_detailed['specificity'], global_step=self.current_epoch)
        self.writer.add_scalar('Specificity/train_bond', bond_detailed['specificity'], global_step=self.current_epoch)
        self.writer.flush()

        return (
            epoch_loss, epoch_atom_acc, epoch_bond_acc,
            atom_detailed['precision'], bond_detailed['precision'],
            atom_detailed['recall'], bond_detailed['recall'],
            atom_detailed['f1'], bond_detailed['f1']
        )


    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        total_loss = 0.0


        beta_a = float(getattr(self.cfg.training, "beta_atom", 0.5))
        beta_b = float(getattr(self.cfg.training, "beta_bond", 0.8))
        pos_rate_lo = float(getattr(self.cfg.training, "pos_rate_lo", 0.5))
        pos_rate_hi = float(getattr(self.cfg.training, "pos_rate_hi", 2.0))


        atom_probs_all, atom_labels_all = [], []
        bond_probs_all_adj, bond_labels_all = [], []

        for batch in self.valid_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            atom_logits, bond_logits = self.model(
                batch['x'], batch['mol_index'], batch['edge_index'],
                batch['edge_attr'], batch['batch'],descr_vec=self._get_descr(batch)
            )


            atom_mask = self._make_atom_mask(batch['y_atom']).to(self.device)

            loss, atom_loss, bond_loss = self.criterion(
                atom_logits, bond_logits,
                batch['y_atom'], batch['y_bond'],
                atom_mask=atom_mask,
                bond_mask=None
            )

            total_loss += loss.item()


            atom_prob_full = torch.sigmoid(atom_logits).detach().flatten().cpu().numpy()
            y_atom = batch['y_atom'].detach().flatten().cpu().numpy()
            eval_mask = (y_atom == 0) | (y_atom >= 0.5)
            atom_prob_eval = atom_prob_full[eval_mask]
            bond_prob = torch.sigmoid(bond_logits).detach().flatten().cpu().numpy()
            atom_lab = (y_atom[eval_mask] >= 0.5)
            bond_lab = (batch['y_bond'].detach().flatten() >= 0.5).cpu().numpy()


            src = batch['edge_index'][0].detach().cpu().numpy()
            dst = batch['edge_index'][1].detach().cpu().numpy()
            gate = np.sqrt(np.clip(atom_prob_full[src], 0, 1) * np.clip(atom_prob_full[dst], 0, 1))
            bond_prob_adj = bond_prob * gate

            atom_probs_all.append(atom_prob_eval);
            atom_labels_all.append((y_atom[eval_mask] >= 0.5))
            bond_probs_all_adj.append(bond_prob_adj);
            bond_labels_all.append(bond_lab)


        a_probs = np.concatenate(atom_probs_all, axis=0)
        a_labs = np.concatenate(atom_labels_all, axis=0)
        b_probs = np.concatenate(bond_probs_all_adj, axis=0)
        b_labs = np.concatenate(bond_labels_all, axis=0)


        def prf_beta(y_true, y_score, t, beta=0.5, min_pos_rate=None, max_pos_rate=None):
            y_pred = (y_score >= t)
            tp = np.logical_and(y_pred, y_true).sum()
            fp = np.logical_and(y_pred, ~y_true).sum()
            fn = np.logical_and(~y_pred, y_true).sum()
            tn = np.logical_and(~y_pred, ~y_true).sum()
            P = tp / (tp + fp + 1e-12)
            R = tp / (tp + fn + 1e-12)
            F = (1 + beta * beta) * P * R / (beta * beta * P + R + 1e-12)
            Acc = (tp + tn) / max(1, tp + fp + fn + tn)

            if min_pos_rate is not None and max_pos_rate is not None:
                pred_pos = (tp + fp)
                total = tp + fp + fn + tn
                pos_rate = pred_pos / max(1, total)
                if not (min_pos_rate <= pos_rate <= max_pos_rate):
                    F -= 1.0
            return P, R, F, Acc, tp, fp, fn, tn


        a_grid = np.linspace(0.10, 0.95, 60)
        b_grid = np.linspace(0.01, 0.95, 100)


        a_pos_rate = a_labs.mean()
        b_pos_rate = b_labs.mean()


        best_a = (-1.0, 0.5, (0, 0, 0, 0, 0, 0, 0, 0))
        for t in a_grid:
            aP, aR, aF, aAcc, atp, afp, afn, atn = prf_beta(
                a_labs, a_probs, t, beta=beta_a,
                min_pos_rate=pos_rate_lo * a_pos_rate, max_pos_rate=pos_rate_hi * a_pos_rate
            )
            if aF > best_a[0]:
                best_a = (aF, t, (aP, aR, aF, aAcc, atp, afp, afn, atn))


        best_b = (-1.0, 0.1, (0, 0, 0, 0, 0, 0, 0, 0))
        for t in b_grid:
            bP, bR, bF, bAcc, btp, bfp, bfn, btn = prf_beta(
                b_labs, b_probs, t, beta=beta_b,
                min_pos_rate=pos_rate_lo * b_pos_rate, max_pos_rate=pos_rate_hi * b_pos_rate
            )
            if bF > best_b[0]:
                best_b = (bF, t, (bP, bR, bF, bAcc, btp, bfp, bfn, btn))


        self.best_t_atom = float(best_a[1])
        self.best_t_bond = float(best_b[1])


        def _refine_if_on_edge(best_t, lo, hi, y, p, pos_rate, beta):
            edge_eps = 1e-9
            if abs(best_t - hi) < edge_eps:
                grid2 = np.linspace(hi, min(0.999, hi + 0.30), 80)
            elif abs(best_t - lo) < edge_eps:
                grid2 = np.linspace(max(0.001, lo - 0.30), lo, 80)
            else:
                return best_t, None
            best_s, bt, bst = -1.0, best_t, None
            for t in grid2:
                P, R, F, Acc, tp, fp, fn, tn = prf_beta(
                    y, p, t, beta=beta,
                    min_pos_rate=pos_rate_lo * pos_rate, max_pos_rate=pos_rate_hi * pos_rate
                )
                if F > best_s:
                    best_s, bt, bst = F, t, (P, R, F, Acc, tp, fp, fn, tn)
            return float(bt), bst


        new_t, new_stats = _refine_if_on_edge(self.best_t_atom, a_grid[0], a_grid[-1], a_labs, a_probs, a_pos_rate,
                                              beta_a)
        if new_t != self.best_t_atom and new_stats is not None:
            self.best_t_atom = new_t
            aP, aR, aFbeta, aAcc, atp, afp, afn, atn = new_stats
        else:
            aP, aR, aFbeta, aAcc, atp, afp, afn, atn = best_a[2]


        new_t, new_stats = _refine_if_on_edge(self.best_t_bond, b_grid[0], b_grid[-1], b_labs, b_probs, b_pos_rate,
                                              beta_b)
        if new_t != self.best_t_bond and new_stats is not None:
            self.best_t_bond = new_t
            bP, bR, bFbeta, bAcc, btp, bfp, bfn, btn = new_stats
        else:
            bP, bR, bFbeta, bAcc, btp, bfp, bfn, btn = best_b[2]

        logging.info(f"[Valid] best_t_atom={self.best_t_atom:.3f} (beta={beta_a}), "
                     f"best_t_bond={self.best_t_bond:.3f} (beta={beta_b})")


        val_atom_acc = float(aAcc)
        val_bond_acc = float(bAcc)

        atom_metrics = {
            'accuracy': float(aAcc), 'precision': float(aP), 'recall': float(aR), 'f1': float(aFbeta),
            'specificity': float(atn / (atn + afp + 1e-12)),
            'balanced_accuracy': float((aR + (atn / (atn + afp + 1e-12))) / 2),
            'tp': int(atp), 'fp': int(afp), 'fn': int(afn), 'tn': int(atn),
            'support': int(atp + afp + afn + atn), 'positive_ratio': float(a_labs.mean())
        }
        bond_metrics = {
            'accuracy': float(bAcc), 'precision': float(bP), 'recall': float(bR), 'f1': float(bFbeta),
            'specificity': float(btn / (btn + bfp + 1e-12)),
            'balanced_accuracy': float((bR + (btn / (btn + bfp + 1e-12))) / 2),
            'tp': int(btp), 'fp': int(bfp), 'fn': int(bfn), 'tn': int(btn),
            'support': int(btp + bfp + bfn + btn), 'positive_ratio': float(b_labs.mean())
        }

        self.log_detailed_metrics(atom_metrics, bond_metrics, "Valid", self.current_epoch)
        val_loss = total_loss / max(1, len(self.valid_loader))

        return (
            val_atom_acc, val_bond_acc, float(val_loss),
            float(aP), float(bP), float(aR), float(bR), float(aFbeta), float(bFbeta)
        )


    @torch.no_grad()
    def test_epoch(self):
        self.model.eval()


        t_atom = float(getattr(self, "best_t_atom", 0.5))
        t_bond = float(getattr(self, "best_t_bond", 0.1))
        logging.info(f"[Test] use thresholds: t_atom={t_atom:.3f}, t_bond={t_bond:.3f}")

        atom_acc_sum, bond_acc_sum = 0.0, 0.0


        atom_metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
        bond_metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

        for batch in self.test_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            atom_logits, bond_logits = self.model(
                batch['x'], batch['mol_index'], batch['edge_index'],
                batch['edge_attr'], batch['batch'],descr_vec=self._get_descr(batch)
            )


            atom_prob = torch.sigmoid(atom_logits)
            bond_prob = torch.sigmoid(bond_logits)


            src, dst = batch['edge_index'][0], batch['edge_index'][1]
            gate = torch.sqrt(
                torch.clamp(atom_prob[src], 0.0, 1.0) * torch.clamp(atom_prob[dst], 0.0, 1.0)
            )
            bond_prob_adj = bond_prob * gate


            atom_pred = (atom_prob >= t_atom).float()
            bond_pred = (bond_prob_adj >= t_bond).float()


            atom_acc = (atom_pred == batch['y_atom']).float().mean()
            bond_acc = (bond_pred == batch['y_bond']).float().mean()
            atom_acc_sum += atom_acc.item()
            bond_acc_sum += bond_acc.item()


            y_atom = batch['y_atom']
            y_bond = batch['y_bond']

            atom_metrics['tp'] += ((atom_pred == 1) & (y_atom == 1)).sum().item()
            atom_metrics['fp'] += ((atom_pred == 1) & (y_atom == 0)).sum().item()
            atom_metrics['fn'] += ((atom_pred == 0) & (y_atom == 1)).sum().item()
            atom_metrics['tn'] += ((atom_pred == 0) & (y_atom == 0)).sum().item()

            bond_metrics['tp'] += ((bond_pred == 1) & (y_bond == 1)).sum().item()
            bond_metrics['fp'] += ((bond_pred == 1) & (y_bond == 0)).sum().item()
            bond_metrics['fn'] += ((bond_pred == 0) & (y_bond == 1)).sum().item()
            bond_metrics['tn'] += ((bond_pred == 0) & (y_bond == 0)).sum().item()


        test_atom_acc = atom_acc_sum / len(self.test_loader)
        test_bond_acc = bond_acc_sum / len(self.test_loader)


        atom_detailed = self.calculate_detailed_metrics_from_stats(atom_metrics, "Atom")
        bond_detailed = self.calculate_detailed_metrics_from_stats(bond_metrics, "Bond")

        return (test_atom_acc, test_bond_acc, atom_detailed, bond_detailed)










    def run(self):
        logging.info("Starting training...")
        best = 0
        for epoch in range(1, self.cfg.training.epoch + 1):
            self.current_epoch = epoch
            logging.info(f'======== Epoch {epoch} ========')


            (train_loss, train_atom_acc, train_bond_acc,
             train_atom_prec, train_bond_prec,
             train_atom_rec, train_bond_rec,
             train_atom_f1, train_bond_f1) = self.train_epoch()


            (val_atom_acc, val_bond_acc, val_loss,
             val_atom_prec, val_bond_prec,
             val_atom_rec, val_bond_rec,
             val_atom_f1, val_bond_f1) = self.val_epoch()


            model_score = self.calculate_model_score(
                val_atom_f1, val_bond_f1,
                val_atom_rec, val_bond_rec,
                val_atom_prec, val_bond_prec
            )


            logging.info(f"=== Epoch {epoch} 总结 ===")
            logging.info(f"训练集 - Loss: {train_loss:.4f}")
            logging.info(
                f"        原子: Acc={train_atom_acc:.4f}, P={train_atom_prec:.4f}, R={train_atom_rec:.4f}, F1={train_atom_f1:.4f}")
            logging.info(
                f"        键:   Acc={train_bond_acc:.4f}, P={train_bond_prec:.4f}, R={train_bond_rec:.4f}, F1={train_bond_f1:.4f}")

            logging.info(f"验证集 - Loss: {val_loss:.4f}, Combined: {model_score:.4f}")
            logging.info(
                f"        原子: Acc={val_atom_acc:.4f}, P={val_atom_prec:.4f}, R={val_atom_rec:.4f}, F1={val_atom_f1:.4f}")
            logging.info(
                f"        键:   Acc={val_bond_acc:.4f}, P={val_bond_prec:.4f}, R={val_bond_rec:.4f}, F1={val_bond_f1:.4f}")


            self.writer.add_scalar('loss/train', train_loss, epoch)
            self.writer.add_scalar('acc/train_atom', train_atom_acc, epoch)
            self.writer.add_scalar('acc/train_bond', train_bond_acc, epoch)
            self.writer.add_scalar('loss/val', val_loss, epoch)
            self.writer.add_scalar('acc/val_atom', val_atom_acc, epoch)
            self.writer.add_scalar('acc/val_bond', val_bond_acc, epoch)
            self.writer.add_scalar('acc/val_combined', model_score, epoch)
            self.writer.add_scalar('f1/val_atom', val_atom_f1, epoch)
            self.writer.add_scalar('f1/val_bond', val_bond_f1, epoch)
            self.writer.add_scalar('precision/val_atom', val_atom_prec, epoch)
            self.writer.add_scalar('precision/val_bond', val_bond_prec, epoch)
            self.writer.add_scalar('recall/val_atom', val_atom_rec, epoch)
            self.writer.add_scalar('recall/val_bond', val_bond_rec, epoch)


            if model_score > best:
                best = model_score
                ckpt = {
                    'model': self.model.state_dict(),
                    'epoch': epoch,
                    'best_score': best,
                    'atom_f1': val_atom_f1,
                    'bond_f1': val_bond_f1,
                    'atom_recall': val_atom_rec,
                    'bond_recall': val_bond_rec,
                    'atom_precision': val_atom_prec,
                    'bond_precision': val_bond_prec,
                    'best_t_atom': getattr(self, 'best_t_atom', 0.5),
                    'best_t_bond': getattr(self, 'best_t_bond', 0.1)
                }
                torch.save(ckpt, f'{self.save_dir}/model/best.pt')
                logging.info(f'★ new best saved - Score: {best:.4f} '
                             f'(Atom F1: {val_atom_f1:.4f}, Bond F1: {val_bond_f1:.4f}, '
                             f'Atom Recall: {val_atom_rec:.4f}, Bond Recall: {val_bond_rec:.4f})')

        logging.info("Training completed!")


        import json, os
        with open(f"{self.save_dir}/best_score.json", "w") as f:
            json.dump({"val_combined": float(best)}, f)


        logging.info('======== Final Test (using best model) ========')
        best_ckpt = torch.load(f'{self.save_dir}/model/best.pt', map_location=self.device)
        self.model.load_state_dict(best_ckpt['model'])
        self.best_t_atom = float(best_ckpt.get('best_t_atom', 0.5))
        self.best_t_bond = float(best_ckpt.get('best_t_bond', 0.1))

        test_atom_acc, test_bond_acc,test_atom_metrics, test_bond_metrics = self.test_epoch()



        logging.info("=== 测试集最终结果 ===")
        logging.info(f"原子任务:")
        logging.info(f"  准确率: {test_atom_acc:.4f}")
        logging.info(f"  精确率: {test_atom_metrics['precision']:.4f}")
        logging.info(f"  召回率: {test_atom_metrics['recall']:.4f}")
        logging.info(f"  F1分数: {test_atom_metrics['f1']:.4f}")
        logging.info(f"  特异度: {test_atom_metrics['specificity']:.4f}")
        logging.info(f"  平衡准确率: {test_atom_metrics['balanced_accuracy']:.4f}")
        logging.info(f"  正样本比例: {test_atom_metrics['positive_ratio']:.4f}")
        logging.info(
            f"  混淆矩阵 - TP: {test_atom_metrics['tp']}, FP: {test_atom_metrics['fp']}, FN: {test_atom_metrics['fn']}, TN: {test_atom_metrics['tn']}")

        logging.info(f"键任务:")
        logging.info(f"  准确率: {test_bond_acc:.4f}")
        logging.info(f"  精确率: {test_bond_metrics['precision']:.4f}")
        logging.info(f"  召回率: {test_bond_metrics['recall']:.4f}")
        logging.info(f"  F1分数: {test_bond_metrics['f1']:.4f}")
        logging.info(f"  特异度: {test_bond_metrics['specificity']:.4f}")
        logging.info(f"  平衡准确率: {test_bond_metrics['balanced_accuracy']:.4f}")
        logging.info(f"  正样本比例: {test_bond_metrics['positive_ratio']:.4f}")
        logging.info(
            f"  混淆矩阵 - TP: {test_bond_metrics['tp']}, FP: {test_bond_metrics['fp']}, FN: {test_bond_metrics['fn']}, TN: {test_bond_metrics['tn']}")


        self.writer.add_scalar('test/atom_accuracy', test_atom_metrics['accuracy'], 0)
        self.writer.add_scalar('test/atom_precision', test_atom_metrics['precision'], 0)
        self.writer.add_scalar('test/atom_recall', test_atom_metrics['recall'], 0)
        self.writer.add_scalar('test/atom_f1', test_atom_metrics['f1'], 0)
        self.writer.add_scalar('test/bond_accuracy', test_bond_metrics['accuracy'], 0)
        self.writer.add_scalar('test/bond_precision', test_bond_metrics['precision'], 0)
        self.writer.add_scalar('test/bond_recall', test_bond_metrics['recall'], 0)
        self.writer.add_scalar('test/bond_f1', test_bond_metrics['f1'], 0)

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    trainer = ReactionCenterTrainer(cfg)
    trainer.run()
