import argparse
import datetime
import glob
import json
import logging
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from addict import Dict
from rdkit import Chem
from tensorboardX import SummaryWriter
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader

from model import (
    GraphEncoder,
    TransformerEncoder,
    SeqDecoder,
    Graph2SmilesEncoder,
    PFASGraph2SmilesModel,
)


def set_global_seed(seed):

    if seed is None:
        seed = 0
    if isinstance(seed, str):
        seed = int(seed)
    if isinstance(seed, (list, tuple)) and len(seed) > 0:
        seed = int(seed[0])

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def canonical(smiles: str) -> str:
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
    except Exception:
        return smiles


class CombinedLoss(nn.Module):
    def __init__(self, pad_idx):
        super().__init__()
        self.token_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, logits, target):

        return self.token_loss_fn(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1)
        )


class SmilesCharTokenizer:
    def __init__(self):
        self.stoi, self.itos = {}, []

    def build_from_dataset(self, dataset):
        chars = set()
        for i in range(len(dataset)):
            smi = getattr(dataset[i], "product_smiles", "")
            smi = canonical(smi)
            for ch in smi.strip():
                chars.add(ch)
        chars = sorted(list(chars))
        self.itos = ["_PAD", "_SOS", "_EOS", "_UNK"] + chars
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    @property
    def pad_idx(self): return self.stoi["_PAD"]
    @property
    def sos_idx(self): return self.stoi["_SOS"]
    @property
    def eos_idx(self): return self.stoi["_EOS"]

    def encode(self, smiles, add_sos_eos=True, max_len=None):
        smiles = canonical(smiles)
        tokens = [self.sos_idx] if add_sos_eos else []
        for ch in smiles.strip():
            tokens.append(self.stoi.get(ch, self.stoi["_UNK"]))
        if add_sos_eos:
            tokens.append(self.eos_idx)
        if max_len:
            tokens = tokens[:max_len]
        return tokens

    def decode(self, token_ids, remove_special=True):
        res = []
        for tid in token_ids:
            if isinstance(tid, torch.Tensor):
                tid = tid.item()
            ch = self.itos[tid]


            if ch == "_EOS":
                break

            if remove_special and ch in {"_PAD", "_SOS"}:
                continue

            res.append(ch)
        return "".join(res)


class PFASSeqDataset(Dataset):
    def __init__(self, data_file):
        super().__init__(None, None, None)
        self.data, self.slices = torch.load(data_file, weights_only=False)
        self._num_graphs = self.slices['x'].shape[0] - 1

    def len(self): return self._num_graphs

    def get(self, idx):
        x_s, x_e = self.slices['x'][idx], self.slices['x'][idx + 1]
        e_s, e_e = self.slices['edge_index'][idx], self.slices['edge_index'][idx + 1]
        d = Data(x=self.data.x[x_s:x_e],
                 edge_index=self.data.edge_index[:, e_s:e_e],
                 edge_attr=self.data.edge_attr[e_s:e_e])
        if hasattr(self.data, "mol_desc"):
            md_s, md_e = self.slices["mol_desc"][idx], self.slices["mol_desc"][idx + 1]
            d.mol_desc = self.data.mol_desc[md_s:md_e]


        if hasattr(self.data, "cond_vec"):
            cd_s, cd_e = self.slices["cond_vec"][idx], self.slices["cond_vec"][idx + 1]
            vec = self.data.cond_vec[cd_s:cd_e]
            vec = torch.nan_to_num(vec)
            d.cond_vec = vec.unsqueeze(0) if vec.dim() == 1 else vec


        if hasattr(self.data, "system_id"):
            sid = int(self.data.system_id[idx])
            d.system_id = torch.tensor([sid],dtype=torch.long)


        raw_smiles = self.data.product_smiles[idx]

        if getattr(self, 'random_smiles', False) and random.random() < 0.5:
            mol = Chem.MolFromSmiles(raw_smiles)
            if mol is not None:
                raw_smiles = Chem.MolToSmiles(mol, doRandom=True)
        d.product_smiles = canonical(raw_smiles)

        if hasattr(self.data, "reactant_smiles"):
            d.reactant_smiles = self.data.reactant_smiles[idx]

        if hasattr(self.data, "pathway_id"):
            d.pathway_id = self.data.pathway_id[idx]
        return d


class PFASSeqTrainer:
    def __init__(self, config_path):


        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = Dict(yaml.safe_load(f))


        self.cfg.optimizer.learning_rate = float(self.cfg.optimizer.learning_rate)
        self.cfg.optimizer.encoder_lr = float(self.cfg.optimizer.encoder_lr)


        self.UNFREEZE_ENCODER = bool(self.cfg.model.get("UNFREEZE_ENCODER", False))

        pass
        pass
        pass
        set_global_seed(self.cfg.data.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(self.cfg.model.save_dir, time_tag)
        os.makedirs(os.path.join(self.save_dir, "log"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "model"), exist_ok=True)
        self._set_logger()
        self.start_epoch = 1
        self.best_val_acc = 0.0
        self.best_seq_acc = 0.0


        self.ckpt_path = None


        if self.cfg.model.get("resume_path"):
            self.ckpt_path = self.cfg.model.resume_path
        else:
            latest = sorted(glob.glob(os.path.join(self.save_dir, "model", "*.pt")))
            if latest:
                self.ckpt_path = latest[-1]

        self.is_resume = bool(self.ckpt_path and os.path.exists(self.ckpt_path))


        self._build_data()
        self._build_model()

        self.criterion = CombinedLoss(pad_idx=self.tokenizer.pad_idx).to(self.device)


    def _load_stage1_weights(self):
        if self.ckpt_path:
            logging.info(f"Loading Stage 1 checkpoint from {self.ckpt_path}")
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"], strict=False)
            logging.info(f"Loaded Stage 1 checkpoint, epoch={ckpt.get('epoch', 'unknown')}")


            if not self.UNFREEZE_ENCODER:
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
                logging.info("Frozen encoder during Stage 1 training.")

    def _load_stage2_weights(self):

        if self.UNFREEZE_ENCODER:
            logging.info("Unfreezing encoder and continuing training in Stage 2.")
            for param in self.model.encoder.parameters():
                param.requires_grad = True


            enc_param = list(self.model.encoder.parameters())
            dec_param = list(self.model.decoder.parameters())
            if self.model.cond_proj is not None:
                dec_param += list(self.model.cond_proj.parameters())


            logging.info("Stage 2: encoder FINETUNE / decoder continue")


    def _set_logger(self):
        log = logging.getLogger()
        log.setLevel(logging.INFO)
        [log.removeHandler(h) for h in log.handlers[:]]
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(os.path.join(self.save_dir, "log", "train.log"),
                                 mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(sh)

    def _build_data(self):
        ds = PFASSeqDataset(self.cfg.data.data_path)
        n = len(ds)


        split_file = self.cfg.data.get("split_file", None)

        if split_file:
            if not os.path.exists(split_file):
                raise FileNotFoundError(
                    f"split_file not found: {split_file}"
                )

            logging.info(f"Loading fixed dataset split from: {split_file}")

            with open(split_file, "r", encoding="utf-8") as f:
                split_info = json.load(f)

            train_idx = np.asarray(split_info["train"], dtype=np.int64)
            valid_idx = np.asarray(split_info["valid"], dtype=np.int64)
            test_idx = np.asarray(split_info["test"], dtype=np.int64)

            logging.info(
                f"Loaded fixed split: "
                f"train={len(train_idx)}, "
                f"valid={len(valid_idx)}, "
                f"test={len(test_idx)}"
            )


        else:
            logging.info(
                "No split_file provided. Using random dataset split."
            )

            idx = np.random.default_rng(
                self.cfg.data.seed
            ).permutation(n)

            tr = int(n * self.cfg.data.train_ratio)
            va = tr + int(n * self.cfg.data.valid_ratio)

            train_idx = idx[:tr]
            valid_idx = idx[tr:va]
            test_idx = idx[
                va:va + int(n * self.cfg.data.test_ratio)
            ]

            split_info = {
                "train": train_idx.tolist(),
                "valid": valid_idx.tolist(),
                "test": test_idx.tolist()
            }

        self.train_set = torch.utils.data.Subset(ds, train_idx)
        self.valid_set = torch.utils.data.Subset(ds, valid_idx)
        self.test_set = torch.utils.data.Subset(ds, test_idx)


        split_info_file = os.path.join(self.save_dir, "split_info.json")
        with open(split_info_file, 'w') as f:
            json.dump(split_info, f, indent=4)

        logging.info(f"Saved dataset split info to {split_info_file}")

        nw = int(self.cfg.data.get("num_workers", 4))
        self.train_loader = PyGDataLoader(self.train_set, batch_size=self.cfg.data.batch_size,
                                          shuffle=True, num_workers=4, pin_memory=True)
        self.valid_loader = PyGDataLoader(self.valid_set, batch_size=self.cfg.data.batch_size,
                                          shuffle=False, num_workers=4, pin_memory=True)
        self.test_loader = PyGDataLoader(self.test_set, batch_size=self.cfg.data.batch_size,
                                         shuffle=False, num_workers=4, pin_memory=True)

        self.tokenizer = SmilesCharTokenizer()
        self.tokenizer.build_from_dataset(self.train_set)
        logging.info(f"Vocab size: {len(self.tokenizer.stoi)}")


        sample = ds[0]
        self.cond_dim = 0
        self.num_system = 0
        if hasattr(sample, "cond_vec"):
            self.cond_dim = sample.cond_vec.view(-1).size(0)
        if hasattr(sample, "system_id"):
            all_sid = [int(ds[i].system_id) for i in range(len(ds))]
            self.num_system = max(all_sid) + 1
        logging.info(f"cond_dim={self.cond_dim}, num_system={self.num_system}")

    def _load_upstream_pretrained(self):
        pretrained_path = self.cfg.model.get("pretrained_path", None)
        if not pretrained_path:
            logging.info("No pretrained_path provided. Train downstream model from scratch.")
            return

        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Upstream pretrained checkpoint not found: {pretrained_path}")

        logging.info(f"Loading upstream pretrained weights from: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location=self.device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        def load_partial_by_prefix(module, prefix, module_name,rename_map=None):
            src_sd = {
                k[len(prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }


            if rename_map:
                renamed_sd = {}
                for k, v in src_sd.items():
                    new_k = k
                    for old_name, new_name in rename_map.items():
                        new_k = new_k.replace(old_name, new_name)
                    renamed_sd[new_k] = v
                src_sd = renamed_sd

            if not src_sd:
                logging.warning(f"[Pretrain] No {prefix}* keys found in upstream checkpoint.")
                return

            tgt_sd = module.state_dict()
            matched_sd = {}
            skipped = []

            for k, v in src_sd.items():
                if k in tgt_sd:
                    if tgt_sd[k].shape == v.shape:
                        matched_sd[k] = v
                    else:
                        skipped.append(
                            f"{k}: ckpt{tuple(v.shape)} != model{tuple(tgt_sd[k].shape)}"
                        )

            missing, unexpected = module.load_state_dict(matched_sd, strict=False)

            logging.info(f"[Pretrain] {module_name} loaded params: {len(matched_sd)}")
            logging.info(f"[Pretrain] {module_name} missing keys: {missing}")
            logging.info(f"[Pretrain] {module_name} unexpected keys: {unexpected}")

            if skipped:
                logging.warning(f"[Pretrain] {module_name} skipped {len(skipped)} mismatched params:")
                for s in skipped[:20]:
                    logging.warning(f"    {s}")
                if len(skipped) > 20:
                    logging.warning(f"    ... and {len(skipped) - 20} more")


        load_partial_by_prefix(
            self.model.encoder.graph_enc,
            prefix="graph_enc.",
            module_name="graph_enc"
        )


        if self.model.encoder.trans_enc is not None:
            load_partial_by_prefix(
                self.model.encoder.trans_enc,
                prefix="trans_enc.",
                module_name="trans_enc",
                rename_map = {"self_attn": "attention"}
            )
        else:
            logging.info("[Pretrain] Downstream trans_enc is None, skip loading trans_enc.")

        logging.info("Upstream pretrained loading finished.")

    def _build_model(self):
        d_descr = self.cfg.model.d_descr_in
        graph_enc = GraphEncoder(
            gnum_layer=self.cfg.model.gnn_num_layer, emb_dim=self.cfg.model.emb_dim,
            gnn_type=self.cfg.model.gnn_type, gnn_aggr=self.cfg.model.gnn_aggr,
            JK=self.cfg.model.gnn_jk, drop_ratio=self.cfg.model.drop_ratio,
            node_readout=self.cfg.model.node_readout, use_cont=self.cfg.model.use_cont,
            use_edge_head=self.cfg.model.use_edge_head, edge_attr_dim=self.cfg.model.edge_attr_dim,
            use_film=self.cfg.model.use_film, d_descr_in=d_descr,cond_dim=self.cond_dim,
            num_system=self.num_system,
            system_emb_dim=self.cfg.model.get("system_emb_dim", 16))
        trans_enc = TransformerEncoder(
            num_layer=self.cfg.model.trans_num_layer, hidden_size=self.cfg.model.emb_dim,
            intermediate_size=self.cfg.model.trans_intermediate_size,
            num_heads=self.cfg.model.num_heads, hidden_dropout_prob=self.cfg.model.drop_ratio
        ) if self.cfg.model.use_transformer else None
        encoder = Graph2SmilesEncoder(graph_enc, trans_enc)
        decoder = SeqDecoder(
            vocab_size=len(self.tokenizer.stoi), d_model=self.cfg.model.emb_dim,
            num_layers=self.cfg.model.decoder_num_layers, num_heads=self.cfg.model.decoder_num_heads,
            dropout_prob=self.cfg.model.decoder_dropout, pad_idx=self.tokenizer.pad_idx,
            max_len=self.cfg.model.max_seq_len)
        self.model = PFASGraph2SmilesModel(encoder, decoder, self.tokenizer.pad_idx, self.cfg.model.use_mol_desc)
        if self.cfg.model.use_mol_desc:
            self.model.init_cond_proj(d_descr, self.cfg.model.emb_dim)
        self.model.to(self.device)
        self._load_upstream_pretrained()


        if not self.UNFREEZE_ENCODER:
            for p in self.model.encoder.parameters():
                p.requires_grad = False
            trainable = list(self.model.decoder.parameters())
            if self.model.cond_proj is not None:
                trainable += list(self.model.cond_proj.parameters())
            self.optimizer = optim.AdamW(
                trainable,
                lr=self.cfg.optimizer.learning_rate,
                weight_decay=self.cfg.optimizer.weight_decay,
            )
            logging.info("Stage 1: encoder FROZEN / decoder+cond_proj trainable")
        else:
            for p in self.model.parameters():
                p.requires_grad = True
            enc_param = list(self.model.encoder.parameters())
            dec_param = list(self.model.decoder.parameters())
            if self.model.cond_proj is not None:
                dec_param += list(self.model.cond_proj.parameters())
            enc_lr = float(self.cfg.optimizer.get("encoder_lr", 1e-5))
            dec_lr = float(self.cfg.optimizer.learning_rate)
            self.optimizer = optim.AdamW(
                [{"params": enc_param, "lr": enc_lr},
                 {"params": dec_param, "lr": dec_lr}],
                weight_decay=self.cfg.optimizer.weight_decay
            )


            logging.info("Stage 2: encoder FINETUNE / decoder continue")

        steps = math.ceil(len(self.train_loader) / max(1, self.cfg.training.accum)) * self.cfg.training.epoch
        warmup = min(self.cfg.scheduler.warmup_step, max(1, steps - 1))
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda s: min(1.0, s / warmup) * max(0.0, (steps - s) / max(1, steps - warmup)))


        weight = torch.ones(len(self.tokenizer.stoi))


        self.writer = SummaryWriter(os.path.join(self.save_dir, "log"))


        if self.ckpt_path and os.path.exists(self.ckpt_path):
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.model.load_state_dict(ckpt["model"], strict=False)

            ckpt_groups = len(ckpt["optimizer"]["param_groups"]) if "optimizer" in ckpt else 0
            cur_groups = len(self.optimizer.param_groups)

            if "optimizer" in ckpt and ckpt_groups == cur_groups:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                if "scheduler" in ckpt:
                    self.scheduler.load_state_dict(ckpt["scheduler"])
                logging.info(f">>> Resume OPT+SCH from {self.ckpt_path}, groups={cur_groups}")
            else:
                logging.warning(f">>> Resume MODEL only: ckpt_groups={ckpt_groups}, cur_groups={cur_groups}")

            logging.info(f">>> Resume from {self.ckpt_path} epoch={self.start_epoch - 1}")
        else:
            self.start_epoch = 1


    def _batch_mol_desc(self, batch):
        if not hasattr(batch, "mol_desc"):
            return None
        d = batch.mol_desc
        if d.dim() == 3 and d.size(1) == 1:
            d = d.squeeze(1)
        return d.to(self.device)

    def _batch_cond_vec(self, batch):
        if not hasattr(batch, "cond_vec"):
            return None
        c = batch.cond_vec

        if c.dim() == 3 and c.size(1) == 1:
            c = c.squeeze(1)
        return c.to(self.device)

    def _batch_system_id(self, batch):
        if not hasattr(batch, "system_id"):
            return None
        sid = batch.system_id
        return sid.view(-1).to(self.device)

    def _batch_encode(self, batch):
        smiles = batch.product_smiles
        if not isinstance(smiles, (list, tuple)):
            smiles = [smiles]
        enc = [self.tokenizer.encode(s, add_sos_eos=True, max_len=self.cfg.model.max_seq_len) for s in smiles]
        lengths = [len(t) for t in enc]
        max_len = max(lengths) if lengths else 1
        full = torch.full((len(enc), max_len), self.tokenizer.pad_idx, dtype=torch.long)
        for i, t in enumerate(enc):
            full[i, :len(t)] = torch.tensor(t)

        return full[:, :-1].to(self.device), full[:, 1:].to(self.device)


    def train_epoch(self, epoch):
        self.model.train()
        accum = max(1, self.cfg.training.accum)
        total_loss, total_correct, total_tokens = 0.0, 0, 0


        if epoch <= 100:
            label_smoothing = 0.0
        else:
            label_smoothing = 0.1

        log_each = max(1, len(self.train_loader) // 10)

        for step, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            descr = self._batch_mol_desc(batch)
            cond_vec = self._batch_cond_vec(batch)
            sysid = batch.system_id.view(-1).to(self.device)
            tgt_in, tgt_out = self._batch_encode(batch)
            B, L = tgt_out.shape


            logits = self.model(batch, tgt_in, descr, cond_vec, sysid)


            if label_smoothing > 0:
                log_probs = torch.log_softmax(logits, dim=-1)
                n_classes = logits.size(-1)
                smoothed_targets = torch.zeros_like(log_probs).scatter_(
                    2, tgt_out.unsqueeze(-1), 1.0 - label_smoothing
                )
                smoothed_targets = smoothed_targets + label_smoothing / n_classes
                loss = -(smoothed_targets * log_probs).sum(dim=-1)
                mask = (tgt_out != self.tokenizer.pad_idx)
                loss = (loss * mask.float()).sum() / mask.sum()
            else:
                loss = self.criterion(logits, tgt_out)

            loss = loss / accum
            loss.backward()

            if (step + 1) % accum == 0:
                if self.cfg.training.clip_norm and self.cfg.training.clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                if self.scheduler is not None:
                    self.scheduler.step()


            with torch.no_grad():
                pred = logits.argmax(-1)
                mask = (tgt_out != self.tokenizer.pad_idx)
                total_correct += ((pred == tgt_out) & mask).sum().item()
                total_tokens += mask.sum().item()
                total_loss += loss.item() * accum


            if (step + 1) % log_each == 0:
                current_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
                lr_msg = f"LR={current_lrs[0]:.2e}" if len(
                    current_lrs) == 1 else f"LR(enc)={current_lrs[0]:.2e} LR(dec)={current_lrs[1]:.2e}"
                logging.info(
                    f"[Train] Epoch {epoch} Step {step + 1}/{len(self.train_loader)}  "
                    f"Loss={loss.item() * accum:.4f}  LS={label_smoothing:.2f}  {lr_msg}"
                )


            del logits, batch, descr, pred, mask
            if label_smoothing > 0:
                del log_probs, smoothed_targets
            torch.cuda.empty_cache()

        token_acc = total_correct / max(1, total_tokens)
        avg_loss = total_loss / len(self.train_loader)


        final_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        logging.info(
            f"[Train] Epoch {epoch} finished. Avg Loss={avg_loss:.4f}, Token Acc={token_acc:.4f}, LRs={final_lrs}")

        return avg_loss, token_acc


    @torch.no_grad()
    def eval_epoch(self, epoch, loader, name="Valid"):
        self.model.eval()
        total_loss, total_correct, total_tokens = 0.0, 0, 0
        for batch in loader:
            batch = batch.to(self.device)
            descr = self._batch_mol_desc(batch)
            cond_vec = self._batch_cond_vec(batch)
            sysid = batch.system_id.view(-1).to(self.device)
            tgt_in, tgt_out = self._batch_encode(batch)
            B, L = tgt_out.shape
            logits = self.model(batch, tgt_in, descr,cond_vec,sysid)

            loss = self.criterion(logits, tgt_out)

            total_loss += loss.item()
            pred = logits.argmax(-1)
            mask = (tgt_out != self.tokenizer.pad_idx)
            total_correct += ((pred == tgt_out) & mask).sum().item()
            total_tokens += mask.sum().item()
        acc = total_correct / max(1, total_tokens)
        logging.info(f"[{name}] Epoch {epoch} - TOKEN-Level(teacher): "
                     f"Loss={total_loss / len(loader):.4f}  Top-1={acc:.4f}")
        return total_loss / len(loader), acc


    @torch.no_grad()
    def test_model(self):
        self.model.eval()


        top1_hits, top5_hits = [], []


        beam_size = int(getattr(self.cfg.eval, "beam_size", 5))
        topk = 5
        if beam_size < topk:
            beam_size = topk


        for batch in self.test_loader:
            batch = batch.to(self.device)
            descr = self._batch_mol_desc(batch)
            cond_vec = self._batch_cond_vec(batch)
            sysid = batch.system_id.view(-1).to(self.device)


            gt_seq, gt_len = self._get_gt_seq(batch)
            B = gt_seq.size(0)


            cands = self._beam_search_topk(
                batch, descr,
                beam_size=beam_size,
                topk=topk,
                cond_vec=cond_vec,
                sysid=sysid
            )


            for i in range(B):
                L = int(gt_len[i].item())
                gt_smiles = canonical(self.tokenizer.decode(gt_seq[i, :L]))


                cand_pack = cands[i] if i < len(cands) else []
                top5_smiles = [x[1] for x in cand_pack[:5]]


                hit5 = (gt_smiles in top5_smiles)
                top5_hits.append(float(hit5))


                hit1 = (gt_smiles == top5_smiles[0])
                top1_hits.append(float(hit1))


        top1 = float(np.mean(top1_hits)) if top1_hits else 0.0
        top5 = float(np.mean(top5_hits)) if top5_hits else 0.0


        logging.info(
            f"Test Set - Top-1 Accuracy: {top1:.4f}, Top-5 Accuracy: {top5:.4f}"
        )

        return top1, top5


    @torch.no_grad()
    def _greedy_decode(self, batch, descr,cond_vec=None, sysid=None,temperature=1.0):
        memory, mask = self.model.encode(batch, descr,cond_vec=cond_vec, system_id=sysid)
        B = memory.size(0)
        max_len = min(self.cfg.model.max_seq_len,
                      getattr(self.cfg.eval, 'max_len', self.cfg.model.max_seq_len))
        device = memory.device
        sos, eos, pad = self.tokenizer.sos_idx, self.tokenizer.eos_idx, self.tokenizer.pad_idx
        seqs = torch.full((B, max_len), pad, dtype=torch.long, device=device)
        seqs[:, 0] = sos


        min_len = 8
        eos_idx = self.tokenizer.eos_idx
        for t in range(1, max_len):
            logits = self.model.decoder(memory, mask, seqs[:, :t])[:, -1, :]
            logits = logits / temperature
            if t <= min_len:
                logits[:, eos_idx] = -1e9
            else:
                logits[:, eos_idx] += 0.2


            recent = seqs[:, max(0, t - 5):t]
            for b in range(B):
                for tok in set(recent[b].tolist()):
                    logits[b, tok] -= 0.8
            next_tok = logits.argmax(-1)
            seqs[:, t] = next_tok
            if (next_tok == eos_idx).all():
                break
        return seqs

    @torch.no_grad()


    @torch.no_grad()
    def _beam_search_topk(self, batch, descr, beam_size=5, topk=5, cond_vec=None, sysid=None,temperature=1.0):
        self.model.eval()
        memory, mask = self.model.encode(batch, descr, cond_vec=cond_vec, system_id=sysid)
        B = memory.size(0)

        max_len = min(self.cfg.model.max_seq_len,
                      getattr(self.cfg.eval, "max_len", self.cfg.model.max_seq_len))
        device = memory.device
        sos, eos, pad = self.tokenizer.sos_idx, self.tokenizer.eos_idx, self.tokenizer.pad_idx


        if beam_size < topk:
            beam_size = topk

        alpha = float(getattr(self.cfg.eval, "length_penalty_alpha", 0.0))

        def norm_score(lp, length):
            if alpha <= 0:
                return lp
            return lp / ((float(length) ** alpha) + 1e-9)

        all_cands = []

        for b in range(B):
            mem_b = memory[b:b + 1]
            mask_b = mask[b:b + 1]

            beams = [(0.0, torch.tensor([[sos]], dtype=torch.long, device=device))]
            finished = []

            for t in range(1, max_len):
                new_beams = []

                for log_p, seq in beams:
                    last = seq[0, -1].item()
                    if last == eos:
                        finished.append((log_p, seq))
                        continue

                    logits = self.model.decoder(mem_b, mask_b, seq)[:, -1, :]
                    log_probs = torch.log_softmax(logits / temperature, dim=-1)

                    min_len = int(getattr(self.cfg.eval, "min_len", 8))
                    if seq.size(1) <= min_len:
                        log_probs[:, self.tokenizer.eos_idx] = -1e9


                    rep_pen = float(getattr(self.cfg.eval, "rep_penalty", 0.2))
                    recent = set(seq[0, -5:].tolist())
                    special = {self.tokenizer.sos_idx, self.tokenizer.pad_idx}
                    recent = [t for t in recent if t not in special and t != self.tokenizer.eos_idx]
                    for tok in recent:
                        log_probs[0, tok] -= rep_pen


                    topk_logp, topk_idx = torch.topk(log_probs, beam_size, dim=-1)

                    for k in range(beam_size):
                        next_id = topk_idx[0, k].view(1, 1)
                        next_lp = log_p + topk_logp[0, k].item()
                        new_seq = torch.cat([seq, next_id], dim=1)
                        new_beams.append((next_lp, new_seq))

                if not new_beams:
                    break

                new_beams.sort(key=lambda x: norm_score(x[0], x[1].size(1)), reverse=True)
                beams = new_beams[:beam_size]


            pool = finished + beams
            pool.sort(key=lambda x: norm_score(x[0], x[1].size(1)), reverse=True)

            uniq = []
            seen = set()

            for score, seq in pool:
                s = self.tokenizer.decode(seq[0].detach().cpu().numpy())
                s = canonical(s)


                if not s:
                    continue


                if s not in seen:
                    seen.add(s)

                    uniq.append((score, s, seq.detach()))

                if len(uniq) >= topk:
                    break


            all_cands.append(uniq)

        return all_cands

    @torch.no_grad()
    def _get_gt_seq(self, batch):
        smiles = batch.product_smiles
        if not isinstance(smiles, (list, tuple)):
            smiles = [smiles]
        enc = [self.tokenizer.encode(s, add_sos_eos=True, max_len=self.cfg.model.max_seq_len) for s in smiles]
        lengths = [len(t) for t in enc]
        max_len = max(lengths) if lengths else 1
        full = torch.full((len(enc), max_len), self.tokenizer.pad_idx, dtype=torch.long)
        for i, t in enumerate(enc):
            full[i, :len(t)] = torch.tensor(t)
        return full.to(self.device), torch.tensor(lengths, dtype=torch.long)


    @torch.no_grad()


    @torch.no_grad()
    def infer_epoch(self, epoch, loader, name="Valid"):
        self.model.eval()

        top1_hits, top5_hits = [], []


        debug_print = bool(getattr(self.cfg.eval, "debug_print", True))
        debug_n = int(getattr(self.cfg.eval, "debug_n", 5))

        for b_idx, batch in enumerate(loader):
            batch = batch.to(self.device)
            descr = self._batch_mol_desc(batch)
            cond_vec = self._batch_cond_vec(batch)
            sysid = batch.system_id.view(-1).to(self.device)

            gt_seq, gt_len = self._get_gt_seq(batch)
            B = gt_seq.size(0)


            data_list = batch.to_data_list()

            beam_size = int(getattr(self.cfg.eval, "beam_size", 5))
            temp = float(getattr(self.cfg.eval, "temperature", 1.0))

            cands = self._beam_search_topk(
                batch, descr,
                beam_size=beam_size,
                topk=5,
                cond_vec=cond_vec,
                sysid=sysid,
                temperature=temp
            )

            for i in range(B):
                L = int(gt_len[i].item())
                gt_smiles = canonical(self.tokenizer.decode(gt_seq[i, :L]))

                cand_pack = cands[i] if i < len(cands) else []
                top5_smiles = [x[1] for x in cand_pack[:5]]

                hit5 = (gt_smiles in top5_smiles)
                top5_hits.append(float(hit5))
                top1_hits.append(float(len(top5_smiles) > 0 and gt_smiles == top5_smiles[0]))


                if hit5:
                    import time
                    from torch_geometric.data import Batch

                    hit_k = top5_smiles.index(gt_smiles)
                    hit_seq = cand_pack[hit_k][2]
                    tgt_infer_in = hit_seq[:, :-1].to(self.device)


                    data_i = data_list[i]
                    sub_batch = Batch.from_data_list([data_i]).to(self.device)


                    reactant_smiles = data_i.reactant_smiles
                    product_smiles_gt = gt_smiles


                    from mol2graphinfo import ATOM_LST
                    atom_types = sub_batch.x[:, 0].cpu().numpy()
                    atom_symbols = [ATOM_LST[int(idx)] for idx in atom_types]


                    descr_i = descr[i:i + 1] if descr is not None else None
                    cond_i = cond_vec[i:i + 1] if cond_vec is not None else None
                    sysid_i = sysid[i:i + 1] if sysid is not None else None


                    logits2, attn = self.model(
                        sub_batch,
                        tgt_infer_in,
                        descr_i, cond_i, sysid_i,
                        return_attn=True,
                        return_enc_attn=True
                    )


                    layer_idx = -2
                    cross_list = attn["cross"]
                    li = layer_idx if len(cross_list) >= 2 else -1
                    cross_w = cross_list[li]

                    decself_list = attn["dec_self"]
                    li2 = layer_idx if len(decself_list) >= 2 else -1
                    decself_w = decself_list[li2]

                    enc_list = attn.get("enc_self", None)
                    enc_w = None
                    if enc_list is not None:
                        li3 = layer_idx if len(enc_list) >= 2 else -1
                        enc_w = enc_list[li3]


                    if cross_w.dim() == 4:
                        cross_ts = cross_w[0].detach().cpu().numpy()
                    else:
                        cross_ts = cross_w[0].detach().cpu().numpy()[None]

                    if decself_w.dim() == 4:
                        dec_self_ts = decself_w[0].detach().cpu().numpy()
                    else:
                        dec_self_ts = decself_w[0].detach().cpu().numpy()[None]

                    enc_ts = None
                    if enc_w is not None:
                        enc_ts = enc_w[0].detach().cpu().numpy()


                    _, mem_pad_mask, _ = self.model.encode(
                        sub_batch, descr_i, cond_i, sysid_i, return_enc_attn=True
                    )
                    valid = (~mem_pad_mask[0]).detach().cpu().numpy().astype(bool)

                    if enc_ts is not None:
                        if enc_ts.ndim == 2:
                            enc_ts = enc_ts[valid][:, valid]
                        elif enc_ts.ndim == 3:
                            enc_ts = enc_ts[:, valid, :][:, :, valid]

                    if cross_ts is not None:
                        if cross_ts.ndim == 2:
                            cross_ts = cross_ts[:, valid]
                        elif cross_ts.ndim == 3:
                            cross_ts = cross_ts[:, :, valid]

                    if dec_self_ts is not None:
                        T = tgt_infer_in.size(1)
                        if dec_self_ts.ndim == 2:
                            dec_self_ts = dec_self_ts[:T, :T]
                        elif dec_self_ts.ndim == 3:
                            dec_self_ts = dec_self_ts[:, :T, :T]


                    valid_atom_types = atom_types[valid]
                    valid_atom_symbols = [ATOM_LST[int(idx)] for idx in valid_atom_types]


                    out_dir = os.path.join(self.save_dir, "attn_dump", f"epoch_{epoch:03d}")
                    os.makedirs(out_dir, exist_ok=True)
                    tag = int(time.time() * 1000)


                    np.savez_compressed(
                        os.path.join(out_dir, f"top5hit_e{epoch:03d}_b{b_idx}_i{i}_k{hit_k}_{tag}.npz"),

                        gt_smiles=product_smiles_gt,
                        hit_rank=hit_k,
                        layer_used=int(li),
                        cross_attn=cross_ts,
                        dec_self_attn=dec_self_ts,
                        enc_self_attn=enc_ts,


                        reactant_smiles=reactant_smiles,
                        atom_types_raw=atom_types,
                        atom_symbols_raw=atom_symbols,
                        atom_types_valid=valid_atom_types,
                        atom_symbols_valid=valid_atom_symbols,
                        valid_mask=valid,


                        product_tokens=product_smiles_gt,
                        product_token_ids=tgt_infer_in[0].cpu().numpy(),


                        num_heads=cross_ts.shape[0],
                        num_tokens=cross_ts.shape[1],
                        num_nodes_raw=len(atom_symbols),
                        num_nodes_valid=len(valid_atom_symbols),


                        batch_idx=b_idx,
                        sample_idx=i,
                    )


                if debug_print and b_idx == 0 and i < debug_n:
                    def _flag(s):
                        try:
                            return "valid" if Chem.MolFromSmiles(s) is not None else "INVALID"
                        except:
                            return "INVALID"

                    pass
                    pass
                    pass
                    pass
                    for k, s in enumerate(top5_smiles):
                        hit = "<<< HIT" if s == gt_smiles else ""
                        pass

        top1 = float(np.mean(top1_hits)) if top1_hits else 0.0
        top5 = float(np.mean(top5_hits)) if top5_hits else 0.0

        logging.info(
            f"[{name}] Epoch {epoch} - INFER(top-k): Top-1={top1:.4f}  Top-5={top5:.4f}"
        )
        return top1, top5

    @torch.no_grad()
    def debug_samples(self, loader, num_batch=1, num_samples=5, tag="DEBUG"):
        self.model.eval()
        from itertools import islice

        for b_idx, batch in enumerate(islice(loader, num_batch)):
            batch = batch.to(self.device)
            cond = self._batch_cond_vec(batch)
            sysid = self._batch_system_id(batch)
            descr = self._batch_mol_desc(batch)


            gt_seq, gt_len = self._get_gt_seq(batch)

            pred_seq = self._greedy_decode(batch, descr,cond,sysid)

            B = gt_seq.size(0)
            for i in range(min(num_samples, B)):
                L = int(gt_len[i].item())
                gt_ids = gt_seq[i, :L]
                pred_ids = pred_seq[i, :L]

                gt_smiles = self.tokenizer.decode(gt_ids)
                pred_smiles = self.tokenizer.decode(pred_ids)

                logging.info(f"[{tag}] sample {i}  GT  : {gt_smiles}")
                logging.info(f"[{tag}] sample {i}  Pred: {pred_smiles}")

            if b_idx + 1 >= num_batch:
                break


    def run(self):

        if self.UNFREEZE_ENCODER:
            if self.is_resume:
                logging.info("Stage2: resume 模式，跳过 stage1_dir 初始化加载")


                logging.info("Setting fixed learning rates for continued training...")
                for i, param_group in enumerate(self.optimizer.param_groups):
                    if len(self.optimizer.param_groups) == 1:

                        param_group['lr'] = self.cfg.optimizer.learning_rate
                        target_lr = self.cfg.optimizer.learning_rate
                    else:

                        if i == 0:
                            param_group['lr'] = self.cfg.optimizer.encoder_lr
                            target_lr = self.cfg.optimizer.encoder_lr
                        else:
                            param_group['lr'] = self.cfg.optimizer.learning_rate
                            target_lr = self.cfg.optimizer.learning_rate
                    logging.info(f"  Param group {i}: LR set to {target_lr:.2e}")


                self.scheduler = None
                logging.info("Scheduler disabled - using fixed learning rates")


            else:

                stage1_dir = self.cfg.model.get("stage1_dir")
                if stage1_dir and os.path.exists(os.path.join(stage1_dir, "best_top5.pt")):
                    ckpt_path = os.path.join(stage1_dir, "best_top5.pt")
                    logging.info(f"Stage2：第一次启动，加载 Stage1 权重 → {ckpt_path}")
                    ckpt = torch.load(ckpt_path, map_location=self.device)
                    self.model.load_state_dict(ckpt["model"], strict=False)
                else:
                    logging.info("Stage2：未提供 stage1_dir，直接从零训练 Stage2（encoder + decoder 都随机初始化）")


            self._load_stage2_weights()

        else:
            logging.info("Stage 1：不加载权重，从头训练 decoder")

        for epoch in range(self.start_epoch, self.cfg.training.epoch + 1):
            logging.info(f"============= Epoch {epoch} =============")

            tr_loss, tr_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.eval_epoch(epoch, self.valid_loader)


            do_infer = True
            if do_infer:
                inf_top1, inf_top5 = self.infer_epoch(epoch, self.valid_loader)
            else:
                inf_top1, inf_top5 = 0.0, 0.0


            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                save_dict = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }

                if self.scheduler is not None:
                    save_dict["scheduler"] = self.scheduler.state_dict()

                torch.save(
                    save_dict,
                    os.path.join(self.save_dir, "model", "best_teacher.pt")
                )


            if inf_top5 > self.best_seq_acc:
                self.best_seq_acc = inf_top5
                save_dict = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                if self.scheduler is not None:
                    save_dict["scheduler"] = self.scheduler.state_dict()

                torch.save(
                    save_dict,
                    os.path.join(self.save_dir, "model", "best_top5.pt")
                )


            save_dict = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            if self.scheduler is not None:
                save_dict["scheduler"] = self.scheduler.state_dict()

            torch.save(
                save_dict,
                os.path.join(self.save_dir, "model", "last.pt")
            )


            self.writer.add_scalar("summary/train_token_acc", tr_acc, epoch)
            self.writer.add_scalar("summary/valid_token_acc_teacher", val_acc, epoch)
            self.writer.add_scalar("summary/valid_top1_infer", inf_top1, epoch)
            self.writer.add_scalar("summary/valid_top5_infer", inf_top5, epoch)

        # ===== Final Test: load best_top5.pt =====
        best_top5_path = os.path.join(
            self.save_dir,
            "model",
            "best_top5.pt"
        )

        if os.path.exists(best_top5_path):
            logging.info(
                f"[Final Test] Loading best_top5 checkpoint: {best_top5_path}"
            )

            ckpt = torch.load(
                best_top5_path,
                map_location=self.device
            )

            self.model.load_state_dict(
                ckpt["model"],
                strict=True
            )

            best_epoch = ckpt.get("epoch", "unknown")

            logging.info(
                f"[Final Test] Loaded best_top5.pt from epoch {best_epoch}"
            )

        else:
            logging.warning(
                "[Final Test] best_top5.pt not found. "
                "Using current model for test."
            )

        test_top1, test_top5 = self.test_model()

        logging.info(
            f"[FINAL TEST] Top-1={test_top1:.4f}, "
            f"Top-5={test_top5:.4f}"
        )

        self.writer.add_scalar(
            "summary/test_top1",
            test_top1,
            self.cfg.training.epoch
        )

        self.writer.add_scalar(
            "summary/test_top5",
            test_top5,
            self.cfg.training.epoch
        )

        self.writer.close()
        logging.info("Training finished.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to config.yaml")
    args = ap.parse_args()
    PFASSeqTrainer(args.config).run()

if __name__ == "__main__":
    main()
