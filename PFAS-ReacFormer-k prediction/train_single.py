import argparse
import datetime
import logging
import math
import os
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from addict import Dict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader

from model import GraphEncoder, Graph2VecEncoder, PFASGraphRegressor, TransformerEncoder


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


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss()

    def forward(self, pred, target):
        return self.loss_fn(pred.squeeze(), target)


class PFASSeqDataset(Dataset):
    def __init__(self, data_file):
        super().__init__(None, None, None)
        self.data, self.slices = torch.load(data_file, weights_only=False)
        self._num_graphs = self.slices["x"].shape[0] - 1

        pass
        for i in range(min(10, self._num_graphs)):
            pass

        all_y = torch.tensor([float(self.data.y[i]) for i in range(self._num_graphs)], dtype=torch.float)
        self.y_mean = all_y.mean()
        self.y_std = all_y.std(unbiased=False)
        if self.y_std < 1e-6:
            self.y_std = torch.tensor(1.0)

        pass

        self.mol_desc_mean = None
        self.mol_desc_std = None
        if hasattr(self.data, "mol_desc"):
            all_desc = []
            for i in range(self._num_graphs):
                desc = self._get_graph_attr("mol_desc", i)
                if desc is not None:
                    all_desc.append(desc.reshape(-1))
            if all_desc:
                all_desc_tensor = torch.stack(all_desc)
                self.mol_desc_mean = all_desc_tensor.mean(dim=0)
                self.mol_desc_std = all_desc_tensor.std(dim=0, unbiased=False)
                self.mol_desc_std = torch.where(
                    self.mol_desc_std < 1e-6,
                    torch.ones_like(self.mol_desc_std),
                    self.mol_desc_std,
                )

        self.cond_vec_mean = None
        self.cond_vec_std = None
        if hasattr(self.data, "cond_vec"):
            all_cond = []
            for i in range(self._num_graphs):
                cond = self._get_graph_attr("cond_vec", i)
                if cond is not None:
                    all_cond.append(cond.reshape(-1))
            if all_cond:
                all_cond_tensor = torch.stack(all_cond)
                self.cond_vec_mean = all_cond_tensor.mean(dim=0)
                self.cond_vec_std = all_cond_tensor.std(dim=0, unbiased=False)
                self.cond_vec_std = torch.where(
                    self.cond_vec_std < 1e-6,
                    torch.ones_like(self.cond_vec_std),
                    self.cond_vec_std,
                )

    def len(self):
        return self._num_graphs

    def _get_graph_attr(self, key, idx):
        if not hasattr(self.data, key) or key not in self.slices:
            return None

        s = int(self.slices[key][idx])
        e = int(self.slices[key][idx + 1])
        v = getattr(self.data, key)[s:e]

        if torch.is_tensor(v) and v.dim() == 0:
            v = v.view(1)

        return v

    def get(self, idx):
        d = Data()

        x_s, x_e = self.slices["x"][idx], self.slices["x"][idx + 1]
        e_s, e_e = self.slices["edge_index"][idx], self.slices["edge_index"][idx + 1]

        d.x = self.data.x[x_s:x_e]
        d.edge_index = self.data.edge_index[:, e_s:e_e]
        d.edge_attr = self.data.edge_attr[e_s:e_e]

        target_value = float(self.data.y[idx])
        d.y = torch.tensor([(target_value - self.y_mean) / self.y_std], dtype=torch.float)

        mol_desc = self._get_graph_attr("mol_desc", idx)
        if mol_desc is not None:
            mol_desc = mol_desc.reshape(-1)
            if self.mol_desc_mean is not None:
                mol_desc = (mol_desc - self.mol_desc_mean) / self.mol_desc_std
            d.mol_desc = mol_desc.unsqueeze(0)

        cond_vec = self._get_graph_attr("cond_vec", idx)
        if cond_vec is not None:
            cond_vec = cond_vec.reshape(-1)
            if self.cond_vec_mean is not None:
                cond_vec = (cond_vec - self.cond_vec_mean) / self.cond_vec_std
            d.cond_vec = cond_vec.unsqueeze(0)

        if hasattr(self.data, "system_id"):
            d.system_id = self.data.system_id[idx]

        return d


class PFASSeqTrainerCV:
    def __init__(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = Dict(yaml.safe_load(f))

        set_global_seed(self.cfg.data.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(self.cfg.model.save_dir, f"{time_tag}_5fold")
        os.makedirs(os.path.join(self.save_dir, "log"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "model"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "predictions"), exist_ok=True)

        self._set_logger()

        self.data = PFASSeqDataset(self.cfg.data.data_path)
        self.y_mean = float(self.data.y_mean)
        self.y_std = float(self.data.y_std) if float(self.data.y_std) >= 1e-6 else 1.0

        sample = self.data[0]
        if hasattr(sample, "cond_vec"):
            self.cond_dim = sample.cond_vec.view(-1).size(0)
        else:
            self.cond_dim = 0

        self.num_system = 0
        if hasattr(sample, "system_id"):
            all_sid = [int(self.data[i].system_id) for i in range(len(self.data))]
            self.num_system = max(all_sid) + 1

        logging.info(f"Dataset size={len(self.data)}, cond_dim={self.cond_dim}, num_system={self.num_system}")
        logging.info(f"log10(k) mean={self.y_mean:.6f}, std={self.y_std:.6f}")

    def _set_logger(self):
        log = logging.getLogger()
        log.setLevel(logging.INFO)
        [log.removeHandler(h) for h in log.handlers[:]]

        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(os.path.join(self.save_dir, "log", "train.log"), mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(sh)

    def _remap_and_filter_pretrained_state(self, src_state, model_state):
        mapped = {}
        skipped_prefix = []
        skipped_shape = []
        skipped_missing = []

        for k, v in src_state.items():
            new_k = None


            if k.startswith("graph_enc."):
                new_k = "encoder." + k


            elif k.startswith("trans_enc."):
                new_k = "encoder." + k


            else:
                skipped_prefix.append(k)
                continue

            if new_k not in model_state:
                skipped_missing.append((k, new_k))
                continue

            if model_state[new_k].shape != v.shape:
                skipped_shape.append((k, new_k, tuple(v.shape), tuple(model_state[new_k].shape)))
                continue

            mapped[new_k] = v

        return mapped, skipped_prefix, skipped_missing, skipped_shape

    def _build_model(self):
        d_descr = self.cfg.model.d_descr_in

        graph_enc = GraphEncoder(
            gnum_layer=self.cfg.model.gnn_num_layer,
            emb_dim=self.cfg.model.emb_dim,
            gnn_type=self.cfg.model.gnn_type,
            gnn_aggr=self.cfg.model.gnn_aggr,
            JK=self.cfg.model.gnn_jk,
            drop_ratio=self.cfg.model.drop_ratio,
            node_readout=self.cfg.model.node_readout,
            use_cont=self.cfg.model.use_cont,
            use_edge_head=self.cfg.model.use_edge_head,
            edge_attr_dim=self.cfg.model.edge_attr_dim,
            use_film=self.cfg.model.use_film,
            d_descr_in=d_descr,
            cond_dim=self.cond_dim,
            num_system=self.num_system,
            system_emb_dim=self.cfg.model.get("system_emb_dim", 16),
        )

        trans_enc = TransformerEncoder(
            num_layer=self.cfg.model.trans_num_layer,
            hidden_size=self.cfg.model.emb_dim,
            intermediate_size=self.cfg.model.trans_intermediate_size,
            num_heads=self.cfg.model.num_heads,
            hidden_dropout_prob=self.cfg.model.drop_ratio,
        ) if self.cfg.model.use_transformer else None

        encoder = Graph2VecEncoder(graph_enc, trans_enc)

        model = PFASGraphRegressor(
            encoder=encoder,
            d_descr_in=d_descr,
            d_model=self.cfg.model.emb_dim,
            use_mol_desc=self.cfg.model.use_mol_desc,
        ).to(self.device)

        pretrained_model_path = self.cfg.model.get("pretrained_model_path", None)
        logging.info(f"pretrained_model_path = {pretrained_model_path}")

        if pretrained_model_path:
            if os.path.exists(pretrained_model_path):
                ckpt = torch.load(pretrained_model_path, map_location=self.device)
                src_state = ckpt["model"]
                model_state = model.state_dict()

                mapped_state, skipped_prefix, skipped_missing, skipped_shape = \
                    self._remap_and_filter_pretrained_state(src_state, model_state)

                load_msg = model.load_state_dict(mapped_state, strict=False)

                logging.info(f"Loaded pretrained model from: {pretrained_model_path}")
                logging.info(f"Actually loaded keys: {len(mapped_state)}")
                logging.info(f"Missing keys: {load_msg.missing_keys[:20]} ... total={len(load_msg.missing_keys)}")
                logging.info(f"Unexpected keys: {load_msg.unexpected_keys[:20]} ... total={len(load_msg.unexpected_keys)}")
                logging.info(f"Skipped by prefix: {len(skipped_prefix)}")
                logging.info(f"Skipped because remapped key missing: {len(skipped_missing)}")
                logging.info(f"Skipped because shape mismatch: {len(skipped_shape)}")

                if len(mapped_state) == 0:
                    logging.warning("No pretrained weights were actually loaded. Check model structure / key mapping.")
            else:
                raise FileNotFoundError(f"Pretrained model not found: {pretrained_model_path}")
        else:
            logging.info("No pretrained model is used.")

        dec_lr = float(self.cfg.optimizer.learning_rate)
        wd = float(self.cfg.optimizer.weight_decay)
        enc_lr = float(self.cfg.optimizer.get("encoder_lr", 1e-5))
        unfreeze = bool(self.cfg.model.get("UNFREEZE_ENCODER", False))

        if not unfreeze:
            for p in model.encoder.parameters():
                p.requires_grad = False

            trainable = list(model.head.parameters())
            if model.cond_proj is not None:
                trainable += list(model.cond_proj.parameters())

            optimizer = optim.AdamW(trainable, lr=dec_lr, weight_decay=wd)
            logging.info("Stage 1: encoder FROZEN / head trainable")
        else:
            for p in model.parameters():
                p.requires_grad = True

            enc_param = list(model.encoder.parameters())
            dec_param = list(model.head.parameters())
            if model.cond_proj is not None:
                dec_param += list(model.cond_proj.parameters())

            optimizer = optim.AdamW(
                [{"params": enc_param, "lr": enc_lr},
                 {"params": dec_param, "lr": dec_lr}],
                weight_decay=wd,
            )
            logging.info("Stage 2: encoder FINETUNE / head continue")

        return model, optimizer

    def _build_scheduler(self, optimizer, train_loader_len):
        steps = math.ceil(train_loader_len / max(1, self.cfg.training.accum)) * self.cfg.training.epoch
        warmup = min(int(self.cfg.scheduler.warmup_step), max(1, steps - 1))

        def lr_lambda(s):
            if s < warmup:
                return float(s) / float(max(1, warmup))
            return max(0.0, float(steps - s) / float(max(1, steps - warmup)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
        c = torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        return c.to(self.device)

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        y_true = y_true.detach().view(-1).cpu().numpy()
        y_pred = y_pred.detach().view(-1).cpu().numpy()
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(mse ** 0.5)
        r2 = r2_score(y_true, y_pred)
        return float(r2), float(mse), rmse

    def prepare_or_load_splits(self):
        split_path = self.cfg.cv.get("split_path", None)
        if split_path is None:
            raise ValueError("请在 yaml 的 cv.split_path 中指定固定划分文件路径")

        if os.path.exists(split_path):
            logging.info(f"Loading fixed CV splits from: {split_path}")
            obj = torch.load(split_path)
            return obj["fold_splits"]

        logging.info(f"Creating new CV splits and saving to: {split_path}")

        n = len(self.data)
        indices = np.arange(n)

        kf = KFold(
            n_splits=int(self.cfg.cv.get("n_splits", 5)),
            shuffle=True,
            random_state=int(self.cfg.data.seed),
        )

        fold_splits = []

        for fold, (trainval_idx, test_idx) in enumerate(kf.split(indices), start=1):
            rng = np.random.default_rng(int(self.cfg.data.seed) + fold)
            trainval_idx = np.array(trainval_idx)
            rng.shuffle(trainval_idx)

            valid_ratio_within_train = float(self.cfg.cv.get("valid_ratio_within_train", 0.125))
            n_valid = max(1, int(len(trainval_idx) * valid_ratio_within_train))

            valid_idx = trainval_idx[:n_valid]
            train_idx = trainval_idx[n_valid:]
            test_idx = np.array(test_idx)

            fold_splits.append({
                "fold": fold,
                "train_idx": train_idx.tolist(),
                "valid_idx": valid_idx.tolist(),
                "test_idx": test_idx.tolist(),
            })

        torch.save(
            {
                "seed": int(self.cfg.data.seed),
                "n_splits": int(self.cfg.cv.get("n_splits", 5)),
                "valid_ratio_within_train": float(self.cfg.cv.get("valid_ratio_within_train", 0.125)),
                "fold_splits": fold_splits,
            },
            split_path,
        )

        logging.info(f"Saved fixed CV splits to: {split_path}")
        return fold_splits

    def run_one_epoch(self, model, loader, optimizer=None, scheduler=None, epoch=0, fold=0, name="Train"):
        is_train = optimizer is not None
        model.train() if is_train else model.eval()

        accum = max(1, self.cfg.training.accum)
        total_loss = 0.0
        all_pred = []
        all_true = []

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        loss_fn = CombinedLoss().to(self.device)

        for step, batch in enumerate(loader):
            batch = batch.to(self.device)
            descr = self._batch_mol_desc(batch)
            cond_vec = self._batch_cond_vec(batch)
            sysid = batch.system_id.view(-1).to(self.device) if hasattr(batch, "system_id") else None
            tgt_out = batch.y.to(self.device)

            if is_train:
                pred = model(batch, descr_vec=descr, cond_vec=cond_vec, system_id=sysid)
                loss = loss_fn(pred, tgt_out)
                (loss / accum).backward()

                if (step + 1) % accum == 0:
                    if self.cfg.training.clip_norm and self.cfg.training.clip_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), self.cfg.training.clip_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
            else:
                with torch.no_grad():
                    pred = model(batch, descr_vec=descr, cond_vec=cond_vec, system_id=sysid)
                    loss = loss_fn(pred, tgt_out)

            total_loss += loss.item()
            all_pred.append(pred.detach())
            all_true.append(tgt_out.detach())

        y_true = torch.cat(all_true, dim=0)
        y_pred = torch.cat(all_pred, dim=0)
        r2, mse, rmse = self.calculate_metrics(y_true, y_pred)
        avg_loss = total_loss / max(1, len(loader))

        logging.info(
            f"[Fold {fold}][{name}] Epoch {epoch} - "
            f"Loss={avg_loss:.4f}  R2={r2:.4f}  MSE={mse:.6f}  RMSE={rmse:.6f}"
        )
        return avg_loss, r2, mse, rmse

    @torch.no_grad()
    def predict_and_save_split(self, model, loader, fold, split_name, split_indices):
        model.eval()

        rows = []
        loss_fn = CombinedLoss().to(self.device)
        total_loss = 0.0
        all_pred = []
        all_true = []

        cursor = 0
        split_indices = np.array(split_indices)

        for batch in loader:
            batch = batch.to(self.device)
            descr = self._batch_mol_desc(batch)
            cond_vec = self._batch_cond_vec(batch)
            sysid = batch.system_id.view(-1).to(self.device) if hasattr(batch, "system_id") else None
            tgt_out = batch.y.to(self.device)

            pred = model(batch, descr_vec=descr, cond_vec=cond_vec, system_id=sysid)
            loss = loss_fn(pred, tgt_out)

            total_loss += loss.item()
            all_pred.append(pred.detach())
            all_true.append(tgt_out.detach())

            y_true = tgt_out.detach().view(-1).cpu()
            y_pred = pred.detach().view(-1).cpu()

            bs = y_true.size(0)
            for i in range(bs):
                dataset_index = int(split_indices[cursor + i])

                true_logk = y_true[i].item() * self.y_std + self.y_mean
                pred_logk = y_pred[i].item() * self.y_std + self.y_mean

                true_raw = 10 ** true_logk
                pred_raw = 10 ** pred_logk

                rows.append({
                    "dataset_index": dataset_index,
                    "true_k": true_raw,
                    "pred_k": pred_raw,
                    "true_logk": true_logk,
                    "pred_logk": pred_logk,
                })

            cursor += bs

        y_true_all = torch.cat(all_true, dim=0)
        y_pred_all = torch.cat(all_pred, dim=0)
        r2, mse, rmse = self.calculate_metrics(y_true_all, y_pred_all)
        avg_loss = total_loss / max(1, len(loader))

        out_csv = os.path.join(self.save_dir, "predictions", f"fold_{fold}_{split_name}_predictions.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)

        logging.info(
            f"[Fold {fold}][{split_name}] "
            f"Loss={avg_loss:.4f}  R2={r2:.4f}  MSE={mse:.6f}  RMSE={rmse:.6f}"
        )
        logging.info(f"[Fold {fold}][{split_name}] Saved predictions to: {out_csv}")

        pass
        for row in rows[:20]:
            pass

        return avg_loss, r2, mse, rmse

    def run_5fold_cv(self):
        fold_splits = self.prepare_or_load_splits()
        fold_results = []

        run_fold = int(self.cfg.cv.get("run_fold", 0))

        for split in fold_splits:
            fold = split["fold"]
            if run_fold > 0 and fold != run_fold:
                continue

            train_idx = np.array(split["train_idx"])
            valid_idx = np.array(split["valid_idx"])
            test_idx = np.array(split["test_idx"])

            logging.info("=" * 80)
            logging.info(f"Starting Fold {fold}")
            logging.info(
                f"[Fold {fold}] train={len(train_idx)}, valid={len(valid_idx)}, test={len(test_idx)}"
            )
            logging.info(
                f"[Fold {fold}] first train idx={train_idx[:5].tolist()}, "
                f"first valid idx={valid_idx[:5].tolist()}, "
                f"first test idx={test_idx[:5].tolist()}"
            )

            train_set = torch.utils.data.Subset(self.data, train_idx)
            valid_set = torch.utils.data.Subset(self.data, valid_idx)
            test_set = torch.utils.data.Subset(self.data, test_idx)

            nw = int(self.cfg.data.get("num_workers", 4))
            batch_size = int(self.cfg.data.batch_size)

            train_loader = PyGDataLoader(
                train_set, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True
            )
            valid_loader = PyGDataLoader(
                valid_set, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True
            )
            test_loader = PyGDataLoader(
                test_set, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True
            )

            train_loader_for_training = PyGDataLoader(
                train_set, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True
            )

            model, optimizer = self._build_model()
            scheduler = self._build_scheduler(optimizer, len(train_loader_for_training))

            best_valid_r2 = -float("inf")
            best_state = None

            for epoch in range(1, int(self.cfg.training.epoch) + 1):
                self.run_one_epoch(
                    model, train_loader_for_training,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    fold=fold,
                    name="Train",
                )
                _, valid_r2, _, _ = self.run_one_epoch(
                    model, valid_loader,
                    optimizer=None,
                    scheduler=None,
                    epoch=epoch,
                    fold=fold,
                    name="Valid",
                )

                if valid_r2 > best_valid_r2:
                    best_valid_r2 = valid_r2
                    best_state = {
                        "epoch": epoch,
                        "model": deepcopy(model.state_dict()),
                        "optimizer": deepcopy(optimizer.state_dict()),
                        "scheduler": deepcopy(scheduler.state_dict()),
                    }

            fold_model_path = os.path.join(self.save_dir, "model", f"fold_{fold}_best.pt")
            if best_state is not None:
                torch.save(best_state, fold_model_path)
                model.load_state_dict(best_state["model"], strict=False)
                logging.info(f"[Fold {fold}] Loaded best epoch={best_state['epoch']} for evaluation")

            train_loss_eval, train_r2_eval, train_mse_eval, train_rmse_eval = self.predict_and_save_split(
                model, train_loader, fold, "train", train_idx
            )
            valid_loss_eval, valid_r2_eval, valid_mse_eval, valid_rmse_eval = self.predict_and_save_split(
                model, valid_loader, fold, "valid", valid_idx
            )
            test_loss, test_r2, test_mse, test_rmse = self.predict_and_save_split(
                model, test_loader, fold, "test", test_idx
            )

            fold_results.append({
                "fold": fold,

                "train_loss": train_loss_eval,
                "train_r2": train_r2_eval,
                "train_mse": train_mse_eval,
                "train_rmse": train_rmse_eval,

                "valid_loss": valid_loss_eval,
                "valid_r2": valid_r2_eval,
                "valid_mse": valid_mse_eval,
                "valid_rmse": valid_rmse_eval,

                "test_loss": test_loss,
                "test_r2": test_r2,
                "test_mse": test_mse,
                "test_rmse": test_rmse,

                "best_valid_r2": best_valid_r2,
            })

        logging.info("=" * 80)
        logging.info("5-Fold CV Summary")

        mean_r2 = np.mean([x["test_r2"] for x in fold_results])
        std_r2 = np.std([x["test_r2"] for x in fold_results])
        mean_rmse = np.mean([x["test_rmse"] for x in fold_results])
        std_rmse = np.std([x["test_rmse"] for x in fold_results])
        mean_mse = np.mean([x["test_mse"] for x in fold_results])
        std_mse = np.std([x["test_mse"] for x in fold_results])

        summary_csv = os.path.join(self.save_dir, "predictions", "cv_summary.csv")
        pd.DataFrame(fold_results).to_csv(summary_csv, index=False)
        logging.info(f"Saved fold summary to: {summary_csv}")

        for res in fold_results:
            logging.info(
                f"Fold {res['fold']}: "
                f"Train R2={res['train_r2']:.4f}, Valid R2={res['valid_r2']:.4f}, Test R2={res['test_r2']:.4f} | "
                f"Train RMSE={res['train_rmse']:.4f}, Valid RMSE={res['valid_rmse']:.4f}, Test RMSE={res['test_rmse']:.4f} | "
                f"Best Valid R2={res['best_valid_r2']:.4f}"
            )

        logging.info(
            f"CV Mean ± Std | "
            f"R2={mean_r2:.4f} ± {std_r2:.4f}, "
            f"RMSE={mean_rmse:.4f} ± {std_rmse:.4f}, "
            f"MSE={mean_mse:.4f} ± {std_mse:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="PFAS k prediction with fixed 5-fold CV")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    trainer = PFASSeqTrainerCV(config_path=args.config)
    trainer.run_5fold_cv()

if __name__ == "__main__":
    main()
