import argparse
import datetime
import logging
import math
import os
import random
from copy import deepcopy
from typing import Dict as TypingDict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from addict import Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader

from model import (
    GraphEncoder,
    Graph2VecEncoder,
    PFASGraphRegressor,
    TransformerEncoder,
)


EPS = 1e-6


def set_global_seed(seed: int, deterministic: bool = False) -> None:
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

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred.view(-1), target.view(-1))


class PFASSeqDataset(Dataset):

    def __init__(self, data_file: str):
        super().__init__(None, None, None)
        loaded = torch.load(data_file, weights_only=False)
        if not isinstance(loaded, (tuple, list)) or len(loaded) != 2:
            raise ValueError(
                "数据文件应由 torch.save((data, slices), path) 保存，"
                "当前文件格式不是 (data, slices)。"
            )

        self.data, self.slices = loaded
        self._num_graphs = int(self.slices["x"].shape[0] - 1)

        all_y = torch.tensor(
            [float(self.data.y[i]) for i in range(self._num_graphs)],
            dtype=torch.float32,
        )
        if not torch.isfinite(all_y).all():
            raise ValueError("目标 y 中存在 NaN 或 Inf，请先检查数据。")

        pass
        for i in range(min(10, self._num_graphs)):
            pass
        pass

    def len(self) -> int:
        return self._num_graphs

    def _get_graph_attr(self, key: str, idx: int) -> Optional[torch.Tensor]:
        if not hasattr(self.data, key) or key not in self.slices:
            return None

        s = int(self.slices[key][idx])
        e = int(self.slices[key][idx + 1])
        value = getattr(self.data, key)[s:e]

        if torch.is_tensor(value) and value.dim() == 0:
            value = value.view(1)
        return value

    def get(self, idx: int) -> Data:
        d = Data()

        x_s = int(self.slices["x"][idx])
        x_e = int(self.slices["x"][idx + 1])
        e_s = int(self.slices["edge_index"][idx])
        e_e = int(self.slices["edge_index"][idx + 1])

        d.x = self.data.x[x_s:x_e]
        d.edge_index = self.data.edge_index[:, e_s:e_e]
        d.edge_attr = self.data.edge_attr[e_s:e_e]


        d.y = torch.tensor([float(self.data.y[idx])], dtype=torch.float32)

        mol_desc = self._get_graph_attr("mol_desc", idx)
        if mol_desc is not None:
            d.mol_desc = mol_desc.reshape(1, -1).to(torch.float32)

        cond_vec = self._get_graph_attr("cond_vec", idx)
        if cond_vec is not None:
            d.cond_vec = cond_vec.reshape(1, -1).to(torch.float32)

        if hasattr(self.data, "system_id"):
            sid = self.data.system_id[idx]
            if torch.is_tensor(sid):
                d.system_id = sid.clone().detach()
            else:
                d.system_id = torch.tensor(int(sid), dtype=torch.long)


        d.sample_id = torch.tensor([idx], dtype=torch.long)
        return d


def _finite_mean_std(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    matrix = matrix.to(torch.float32)
    finite = torch.isfinite(matrix)
    count = finite.sum(dim=0).clamp_min(1)

    safe_values = torch.where(finite, matrix, torch.zeros_like(matrix))
    mean = safe_values.sum(dim=0) / count

    centered = torch.where(finite, matrix - mean, torch.zeros_like(matrix))
    var = (centered ** 2).sum(dim=0) / count
    std = torch.sqrt(var)
    std = torch.where(std < EPS, torch.ones_like(std), std)
    return mean, std


def _normalize_with_train_stats(
    value: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    value = value.to(torch.float32).reshape(-1)
    mean = mean.to(value.device)
    std = std.to(value.device)


    value = torch.where(torch.isfinite(value), value, mean)
    return (value - mean) / std


def compute_fold_statistics(
    base_dataset: PFASSeqDataset,
    train_indices: Sequence[int],
) -> TypingDict[str, Optional[torch.Tensor]]:
    train_indices = [int(i) for i in train_indices]
    if len(train_indices) == 0:
        raise ValueError("训练集为空，无法计算标准化统计量。")

    raw_samples = [base_dataset[i] for i in train_indices]

    y_values = torch.stack([sample.y.view(-1)[0] for sample in raw_samples]).float()
    y_mean = y_values.mean()
    y_std = y_values.std(unbiased=False)
    if float(y_std) < EPS:
        y_std = torch.tensor(1.0, dtype=torch.float32)

    stats: TypingDict[str, Optional[torch.Tensor]] = {
        "y_mean": y_mean.detach().cpu(),
        "y_std": y_std.detach().cpu(),
        "mol_desc_mean": None,
        "mol_desc_std": None,
        "cond_vec_mean": None,
        "cond_vec_std": None,
    }

    desc_values = [s.mol_desc.reshape(-1) for s in raw_samples if hasattr(s, "mol_desc")]
    if desc_values:
        if len(desc_values) != len(raw_samples):
            raise ValueError("部分训练样本缺少 mol_desc，PyG 批处理可能不一致，请检查数据。")
        desc_matrix = torch.stack(desc_values)
        desc_mean, desc_std = _finite_mean_std(desc_matrix)
        stats["mol_desc_mean"] = desc_mean.cpu()
        stats["mol_desc_std"] = desc_std.cpu()

    cond_values = [s.cond_vec.reshape(-1) for s in raw_samples if hasattr(s, "cond_vec")]
    if cond_values:
        if len(cond_values) != len(raw_samples):
            raise ValueError("部分训练样本缺少 cond_vec，PyG 批处理可能不一致，请检查数据。")
        cond_matrix = torch.stack(cond_values)
        cond_mean, cond_std = _finite_mean_std(cond_matrix)
        stats["cond_vec_mean"] = cond_mean.cpu()
        stats["cond_vec_std"] = cond_std.cpu()

    return stats


class FoldNormalizedDataset(TorchDataset):

    def __init__(
        self,
        base_dataset: PFASSeqDataset,
        indices: Sequence[int],
        fold_stats: TypingDict[str, Optional[torch.Tensor]],
    ):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.stats = fold_stats

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> Data:
        dataset_index = int(self.indices[position])
        d = self.base_dataset[dataset_index]

        y_mean = self.stats["y_mean"]
        y_std = self.stats["y_std"]
        d.y = ((d.y.view(-1) - y_mean) / y_std).to(torch.float32)

        if hasattr(d, "mol_desc") and self.stats["mol_desc_mean"] is not None:
            normalized = _normalize_with_train_stats(
                d.mol_desc,
                self.stats["mol_desc_mean"],
                self.stats["mol_desc_std"],
            )
            d.mol_desc = normalized.unsqueeze(0)

        if hasattr(d, "cond_vec") and self.stats["cond_vec_mean"] is not None:
            normalized = _normalize_with_train_stats(
                d.cond_vec,
                self.stats["cond_vec_mean"],
                self.stats["cond_vec_std"],
            )
            d.cond_vec = normalized.unsqueeze(0)

        return d


class PFASSeqTrainerCV:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = Dict(yaml.safe_load(f))

        self.config_path = config_path
        self.split_seed = int(self.cfg.cv.get("split_seed", self.cfg.data.seed))
        self.train_seed = int(self.cfg.training.get("seed", self.cfg.data.seed))
        self.deterministic = bool(self.cfg.training.get("deterministic", False))
        set_global_seed(self.train_seed, deterministic=self.deterministic)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_splits = int(self.cfg.cv.get("n_splits", 5))

        time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(
            self.cfg.model.save_dir,
            f"{time_tag}_{self.n_splits}fold_fixed",
        )
        os.makedirs(os.path.join(self.save_dir, "log"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "model"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "predictions"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "fold_stats"), exist_ok=True)

        self._set_logger()
        self._save_config_copy()

        self.data = PFASSeqDataset(self.cfg.data.data_path)

        sample = self.data[0]
        self.cond_dim = sample.cond_vec.view(-1).size(0) if hasattr(sample, "cond_vec") else 0

        self.num_system = 0
        if hasattr(sample, "system_id"):
            all_sid = [int(self.data[i].system_id.view(-1)[0]) for i in range(len(self.data))]
            self.num_system = max(all_sid) + 1

        logging.info(
            "Dataset size=%d, cond_dim=%d, num_system=%d, device=%s",
            len(self.data),
            self.cond_dim,
            self.num_system,
            self.device,
        )
        logging.info(
            "split_seed=%d, train_seed=%d, deterministic=%s",
            self.split_seed,
            self.train_seed,
            self.deterministic,
        )

    def _set_logger(self) -> None:
        log = logging.getLogger()
        log.setLevel(logging.INFO)
        for handler in log.handlers[:]:
            log.removeHandler(handler)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(
            os.path.join(self.save_dir, "log", "train.log"),
            mode="w",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        log.addHandler(stream_handler)

    def _save_config_copy(self) -> None:
        output_path = os.path.join(self.save_dir, "used_config.yaml")
        with open(self.config_path, "r", encoding="utf-8") as src, open(
            output_path, "w", encoding="utf-8"
        ) as dst:
            dst.write(src.read())

    def _remap_and_filter_pretrained_state(self, src_state, model_state):
        mapped = {}
        skipped_prefix = []
        skipped_shape = []
        skipped_missing = []

        for key, value in src_state.items():
            new_key = None
            if key.startswith("graph_enc."):
                new_key = "encoder." + key
            elif key.startswith("trans_enc."):
                new_key = "encoder." + key
            else:
                skipped_prefix.append(key)
                continue

            if new_key not in model_state:
                skipped_missing.append((key, new_key))
                continue

            if model_state[new_key].shape != value.shape:
                skipped_shape.append(
                    (key, new_key, tuple(value.shape), tuple(model_state[new_key].shape))
                )
                continue

            mapped[new_key] = value

        return mapped, skipped_prefix, skipped_missing, skipped_shape

    def _build_model(self):
        d_descr = int(self.cfg.model.d_descr_in)

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

        trans_enc = None
        if self.cfg.model.use_transformer:
            trans_enc = TransformerEncoder(
                num_layer=self.cfg.model.trans_num_layer,
                hidden_size=self.cfg.model.emb_dim,
                intermediate_size=self.cfg.model.trans_intermediate_size,
                num_heads=self.cfg.model.num_heads,
                hidden_dropout_prob=self.cfg.model.drop_ratio,
            )

        encoder = Graph2VecEncoder(graph_enc, trans_enc)
        model = PFASGraphRegressor(
            encoder=encoder,
            d_descr_in=d_descr,
            d_model=self.cfg.model.emb_dim,
            use_mol_desc=self.cfg.model.use_mol_desc,
            head_hidden=int(self.cfg.model.get("head_hidden", 256)),
            head_dropout=float(self.cfg.model.get("head_dropout", 0.1)),
        ).to(self.device)

        pretrained_model_path = self.cfg.model.get("pretrained_model_path", None)
        logging.info("pretrained_model_path=%s", pretrained_model_path)

        if pretrained_model_path:
            if not os.path.exists(pretrained_model_path):
                raise FileNotFoundError(f"Pretrained model not found: {pretrained_model_path}")

            checkpoint = torch.load(
                pretrained_model_path,
                map_location=self.device,
                weights_only=False,
            )
            src_state = checkpoint["model"] if "model" in checkpoint else checkpoint
            model_state = model.state_dict()

            mapped_state, skipped_prefix, skipped_missing, skipped_shape = (
                self._remap_and_filter_pretrained_state(src_state, model_state)
            )
            load_msg = model.load_state_dict(mapped_state, strict=False)

            logging.info("Loaded pretrained model from: %s", pretrained_model_path)
            logging.info("Actually loaded keys: %d", len(mapped_state))
            logging.info("Missing keys total: %d", len(load_msg.missing_keys))
            logging.info("Unexpected keys total: %d", len(load_msg.unexpected_keys))
            logging.info("Skipped by prefix: %d", len(skipped_prefix))
            logging.info("Skipped because key missing: %d", len(skipped_missing))
            logging.info("Skipped because shape mismatch: %d", len(skipped_shape))

            if len(mapped_state) == 0:
                logging.warning(
                    "No pretrained weights were actually loaded. "
                    "Please check model structure and key mapping."
                )
        else:
            logging.info("No pretrained model is used.")

        decoder_lr = float(self.cfg.optimizer.learning_rate)
        weight_decay = float(self.cfg.optimizer.weight_decay)
        encoder_lr = float(self.cfg.optimizer.get("encoder_lr", 1e-5))
        unfreeze = bool(self.cfg.model.get("UNFREEZE_ENCODER", False))

        if not unfreeze:
            for parameter in model.encoder.parameters():
                parameter.requires_grad = False

            trainable = list(model.head.parameters())
            if model.cond_proj is not None:
                trainable += list(model.cond_proj.parameters())

            optimizer = optim.AdamW(
                trainable,
                lr=decoder_lr,
                weight_decay=weight_decay,
            )
            logging.info("Encoder FROZEN; regression head trainable.")
        else:
            for parameter in model.parameters():
                parameter.requires_grad = True

            encoder_parameters = list(model.encoder.parameters())
            decoder_parameters = list(model.head.parameters())
            if model.cond_proj is not None:
                decoder_parameters += list(model.cond_proj.parameters())

            optimizer = optim.AdamW(
                [
                    {"params": encoder_parameters, "lr": encoder_lr},
                    {"params": decoder_parameters, "lr": decoder_lr},
                ],
                weight_decay=weight_decay,
            )
            logging.info("Encoder FINETUNE; regression head trainable.")

        return model, optimizer

    def _build_scheduler(self, optimizer, train_loader_len: int):
        accum = max(1, int(self.cfg.training.accum))
        epochs = int(self.cfg.training.epoch)
        total_steps = math.ceil(train_loader_len / accum) * epochs
        total_steps = max(1, total_steps)
        warmup = min(
            int(self.cfg.scheduler.warmup_step),
            max(1, total_steps - 1),
        )

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step) / float(max(1, warmup))
            return max(
                0.0,
                float(total_steps - step) / float(max(1, total_steps - warmup)),
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _batch_mol_desc(self, batch):
        if not hasattr(batch, "mol_desc"):
            return None
        descr = batch.mol_desc
        if descr.dim() == 3 and descr.size(1) == 1:
            descr = descr.squeeze(1)
        descr = torch.nan_to_num(descr, nan=0.0, posinf=0.0, neginf=0.0)
        return descr.to(self.device)

    def _batch_cond_vec(self, batch):
        if not hasattr(batch, "cond_vec"):
            return None
        cond = batch.cond_vec
        if cond.dim() == 3 and cond.size(1) == 1:
            cond = cond.squeeze(1)
        cond = torch.nan_to_num(cond, nan=0.0, posinf=0.0, neginf=0.0)
        return cond.to(self.device)

    @staticmethod
    def _inverse_y(
        y_standardized: np.ndarray,
        fold_stats: TypingDict[str, Optional[torch.Tensor]],
    ) -> np.ndarray:
        y_mean = float(fold_stats["y_mean"])
        y_std = float(fold_stats["y_std"])
        return y_standardized * y_std + y_mean

    @staticmethod
    def calculate_metrics_numpy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> TypingDict[str, float]:
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

        if len(y_true) != len(y_pred):
            raise ValueError("y_true 与 y_pred 长度不一致。")
        if len(y_true) == 0:
            raise ValueError("没有可用于计算指标的样本。")

        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
        return {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae}

    def prepare_or_load_splits(self):
        split_path = self.cfg.cv.get("split_path", None)
        if split_path is None:
            raise ValueError("在 yaml 的 cv.split_path 中指定固定划分文件路径。")

        valid_ratio = float(self.cfg.cv.get("valid_ratio_within_train", 0.2))
        if not 0.0 < valid_ratio < 0.5:
            raise ValueError("cv.valid_ratio_within_train 应位于 (0, 0.5) 范围内。")

        expected_meta = {
            "seed": self.split_seed,
            "n_splits": self.n_splits,
            "valid_ratio_within_train": valid_ratio,
            "n_samples": len(self.data),
        }

        if os.path.exists(split_path):
            obj = torch.load(split_path, weights_only=False)
            metadata_matches = all(obj.get(k) == v for k, v in expected_meta.items())
            if metadata_matches and "fold_splits" in obj:
                logging.info("Loading fixed CV splits from: %s", split_path)
                return obj["fold_splits"]

            logging.warning(
                "Existing split file metadata does not match current configuration; "
                "it will be regenerated. Old metadata=%s, expected=%s",
                {k: obj.get(k) for k in expected_meta},
                expected_meta,
            )

        split_dir = os.path.dirname(os.path.abspath(split_path))
        os.makedirs(split_dir, exist_ok=True)
        logging.info("Creating CV splits and saving to: %s", split_path)

        indices = np.arange(len(self.data))
        kfold = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.split_seed,
        )

        fold_splits = []
        for fold, (trainval_idx, test_idx) in enumerate(kfold.split(indices), start=1):
            trainval_idx = np.asarray(trainval_idx, dtype=np.int64)
            test_idx = np.asarray(test_idx, dtype=np.int64)

            rng = np.random.default_rng(self.split_seed + fold)
            rng.shuffle(trainval_idx)

            n_valid = max(1, int(round(len(trainval_idx) * valid_ratio)))
            n_valid = min(n_valid, len(trainval_idx) - 1)

            valid_idx = trainval_idx[:n_valid]
            train_idx = trainval_idx[n_valid:]

            fold_splits.append(
                {
                    "fold": fold,
                    "train_idx": train_idx.tolist(),
                    "valid_idx": valid_idx.tolist(),
                    "test_idx": test_idx.tolist(),
                }
            )

        torch.save(
            {
                **expected_meta,
                "fold_splits": fold_splits,
            },
            split_path,
        )
        logging.info("Saved fixed CV splits to: %s", split_path)
        return fold_splits

    def _make_loaders(
        self,
        train_idx: np.ndarray,
        valid_idx: np.ndarray,
        test_idx: np.ndarray,
        fold_stats,
        fold_seed: int,
    ):
        train_set = FoldNormalizedDataset(self.data, train_idx, fold_stats)
        valid_set = FoldNormalizedDataset(self.data, valid_idx, fold_stats)
        test_set = FoldNormalizedDataset(self.data, test_idx, fold_stats)

        num_workers = int(self.cfg.data.get("num_workers", 4))
        batch_size = int(self.cfg.data.batch_size)
        pin_memory = self.device == "cuda"

        generator = torch.Generator()
        generator.manual_seed(fold_seed)

        common_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "worker_init_fn": seed_worker,
        }

        train_loader_for_training = PyGDataLoader(
            train_set,
            shuffle=True,
            generator=generator,
            **common_kwargs,
        )
        train_loader_eval = PyGDataLoader(train_set, shuffle=False, **common_kwargs)
        valid_loader = PyGDataLoader(valid_set, shuffle=False, **common_kwargs)
        test_loader = PyGDataLoader(test_set, shuffle=False, **common_kwargs)

        return train_loader_for_training, train_loader_eval, valid_loader, test_loader

    def run_one_epoch(
        self,
        model,
        loader,
        fold_stats,
        optimizer=None,
        scheduler=None,
        epoch=0,
        fold=0,
        name="Train",
    ):
        is_train = optimizer is not None
        model.train() if is_train else model.eval()

        accum = max(1, int(self.cfg.training.accum))
        total_loss = 0.0
        total_samples = 0
        all_pred_std = []
        all_true_std = []

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        loss_fn = CombinedLoss().to(self.device)
        loader_len = len(loader)

        for step, batch in enumerate(loader):
            batch = batch.to(self.device)
            descr = self._batch_mol_desc(batch)
            cond_vec = self._batch_cond_vec(batch)
            sysid = (
                batch.system_id.view(-1).to(self.device)
                if hasattr(batch, "system_id")
                else None
            )
            target_std = batch.y.view(-1).to(self.device)

            if is_train:
                pred_std = model(
                    batch,
                    descr_vec=descr,
                    cond_vec=cond_vec,
                    system_id=sysid,
                ).view(-1)
                loss = loss_fn(pred_std, target_std)
                (loss / accum).backward()

                should_step = ((step + 1) % accum == 0) or ((step + 1) == loader_len)
                if should_step:
                    clip_norm = float(self.cfg.training.get("clip_norm", 0.0) or 0.0)
                    if clip_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
            else:
                with torch.no_grad():
                    pred_std = model(
                        batch,
                        descr_vec=descr,
                        cond_vec=cond_vec,
                        system_id=sysid,
                    ).view(-1)
                    loss = loss_fn(pred_std, target_std)

            batch_size = int(target_std.numel())
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            all_pred_std.append(pred_std.detach().cpu())
            all_true_std.append(target_std.detach().cpu())

        if total_samples == 0:
            raise RuntimeError(f"Fold {fold} 的 {name} DataLoader 为空。")

        y_true_std = torch.cat(all_true_std).numpy()
        y_pred_std = torch.cat(all_pred_std).numpy()
        y_true_logk = self._inverse_y(y_true_std, fold_stats)
        y_pred_logk = self._inverse_y(y_pred_std, fold_stats)
        metrics = self.calculate_metrics_numpy(y_true_logk, y_pred_logk)
        avg_loss = total_loss / total_samples

        logging.info(
            "[Fold %d][%s] Epoch %d - StdLoss=%.6f  "
            "log10(k): R2=%.4f  MSE=%.6f  RMSE=%.6f  MAE=%.6f",
            fold,
            name,
            epoch,
            avg_loss,
            metrics["r2"],
            metrics["mse"],
            metrics["rmse"],
            metrics["mae"],
        )
        return {"loss": avg_loss, **metrics}

    @torch.no_grad()
    def predict_and_save_split(
        self,
        model,
        loader,
        fold_stats,
        fold: int,
        split_name: str,
    ):
        model.eval()

        loss_fn = CombinedLoss().to(self.device)
        total_loss = 0.0
        total_samples = 0
        rows = []

        for batch in loader:
            batch = batch.to(self.device)
            descr = self._batch_mol_desc(batch)
            cond_vec = self._batch_cond_vec(batch)
            sysid = (
                batch.system_id.view(-1).to(self.device)
                if hasattr(batch, "system_id")
                else None
            )

            target_std = batch.y.view(-1).to(self.device)
            pred_std = model(
                batch,
                descr_vec=descr,
                cond_vec=cond_vec,
                system_id=sysid,
            ).view(-1)
            loss = loss_fn(pred_std, target_std)

            batch_size = int(target_std.numel())
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            true_std_np = target_std.detach().cpu().numpy()
            pred_std_np = pred_std.detach().cpu().numpy()
            true_logk = self._inverse_y(true_std_np, fold_stats)
            pred_logk = self._inverse_y(pred_std_np, fold_stats)
            sample_ids = batch.sample_id.view(-1).detach().cpu().numpy()

            for sample_id, true_value, pred_value in zip(
                sample_ids,
                true_logk,
                pred_logk,
            ):

                true_k = float(np.power(10.0, true_value))
                pred_k = float(np.power(10.0, pred_value))
                rows.append(
                    {
                        "fold": fold,
                        "split": split_name,
                        "dataset_index": int(sample_id),
                        "true_logk": float(true_value),
                        "pred_logk": float(pred_value),
                        "residual_logk": float(true_value - pred_value),
                        "abs_error_logk": float(abs(true_value - pred_value)),
                        "true_k": true_k,
                        "pred_k": pred_k,
                    }
                )

        if total_samples == 0:
            raise RuntimeError(f"Fold {fold} 的 {split_name} DataLoader 为空。")

        result_df = pd.DataFrame(rows).sort_values("dataset_index").reset_index(drop=True)
        metrics = self.calculate_metrics_numpy(
            result_df["true_logk"].to_numpy(),
            result_df["pred_logk"].to_numpy(),
        )
        avg_loss = total_loss / total_samples

        output_csv = os.path.join(
            self.save_dir,
            "predictions",
            f"fold_{fold}_{split_name}_predictions.csv",
        )
        result_df.to_csv(output_csv, index=False)

        logging.info(
            "[Fold %d][%s final] StdLoss=%.6f  "
            "log10(k): R2=%.4f  MSE=%.6f  RMSE=%.6f  MAE=%.6f",
            fold,
            split_name,
            avg_loss,
            metrics["r2"],
            metrics["mse"],
            metrics["rmse"],
            metrics["mae"],
        )
        logging.info("Saved predictions to: %s", output_csv)

        return {"loss": avg_loss, **metrics}, result_df

    @staticmethod
    def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
        array = np.asarray(values, dtype=np.float64)
        if len(array) == 0:
            return float("nan"), float("nan")
        mean = float(np.mean(array))
        std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
        return mean, std

    def run_5fold_cv(self):
        fold_splits = self.prepare_or_load_splits()
        fold_results = []
        oof_frames = []

        run_fold = int(self.cfg.cv.get("run_fold", 0))
        if run_fold != 0:
            logging.warning(
                "cv.run_fold=%d：只运行单折时无法得到完整 OOF 指标。",
                run_fold,
            )

        max_epochs = int(self.cfg.training.epoch)
        patience = int(self.cfg.training.get("early_stopping_patience", 30))
        min_delta = float(self.cfg.training.get("early_stopping_min_delta", 0.0))

        for split in fold_splits:
            fold = int(split["fold"])
            if run_fold > 0 and fold != run_fold:
                continue

            train_idx = np.asarray(split["train_idx"], dtype=np.int64)
            valid_idx = np.asarray(split["valid_idx"], dtype=np.int64)
            test_idx = np.asarray(split["test_idx"], dtype=np.int64)


            fold_seed = self.train_seed
            set_global_seed(fold_seed, deterministic=self.deterministic)

            logging.info("=" * 90)
            logging.info("Starting Fold %d with fold_train_seed=%d", fold, fold_seed)
            logging.info(
                "[Fold %d] train=%d, valid=%d, test=%d",
                fold,
                len(train_idx),
                len(valid_idx),
                len(test_idx),
            )

            fold_stats = compute_fold_statistics(self.data, train_idx)
            y_mean = float(fold_stats["y_mean"])
            y_std = float(fold_stats["y_std"])
            logging.info(
                "[Fold %d] TRAIN-ONLY y stats: mean=%.6f, std=%.6f",
                fold,
                y_mean,
                y_std,
            )

            stats_path = os.path.join(
                self.save_dir,
                "fold_stats",
                f"fold_{fold}_normalization.pt",
            )
            torch.save(fold_stats, stats_path)

            (
                train_loader_for_training,
                train_loader_eval,
                valid_loader,
                test_loader,
            ) = self._make_loaders(
                train_idx,
                valid_idx,
                test_idx,
                fold_stats,
                fold_seed,
            )

            model, optimizer = self._build_model()
            scheduler = self._build_scheduler(optimizer, len(train_loader_for_training))

            best_valid_loss = float("inf")
            best_state = None
            epochs_without_improvement = 0

            for epoch in range(1, max_epochs + 1):
                self.run_one_epoch(
                    model,
                    train_loader_for_training,
                    fold_stats,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    fold=fold,
                    name="Train",
                )
                valid_metrics = self.run_one_epoch(
                    model,
                    valid_loader,
                    fold_stats,
                    optimizer=None,
                    scheduler=None,
                    epoch=epoch,
                    fold=fold,
                    name="Valid",
                )

                current_valid_loss = float(valid_metrics["loss"])
                improved = current_valid_loss < (best_valid_loss - min_delta)

                if improved:
                    best_valid_loss = current_valid_loss
                    epochs_without_improvement = 0
                    best_state = {
                        "epoch": epoch,
                        "model": deepcopy(model.state_dict()),
                        "optimizer": deepcopy(optimizer.state_dict()),
                        "scheduler": deepcopy(scheduler.state_dict()),
                        "fold_stats": deepcopy(fold_stats),
                        "fold": fold,
                        "split_seed": self.split_seed,
                        "train_seed": fold_seed,
                        "best_valid_loss": best_valid_loss,
                        "best_valid_metrics": deepcopy(valid_metrics),
                    }
                else:
                    epochs_without_improvement += 1

                if patience > 0 and epochs_without_improvement >= patience:
                    logging.info(
                        "[Fold %d] Early stopping at epoch %d; "
                        "best_valid_loss=%.6f, patience=%d",
                        fold,
                        epoch,
                        best_valid_loss,
                        patience,
                    )
                    break

            if best_state is None:
                raise RuntimeError(f"Fold {fold} 未保存任何最佳模型。")

            fold_model_path = os.path.join(
                self.save_dir,
                "model",
                f"fold_{fold}_best.pt",
            )
            torch.save(best_state, fold_model_path)
            model.load_state_dict(best_state["model"], strict=True)
            logging.info(
                "[Fold %d] Loaded best epoch=%d selected by minimum validation loss.",
                fold,
                best_state["epoch"],
            )

            train_metrics, _ = self.predict_and_save_split(
                model,
                train_loader_eval,
                fold_stats,
                fold,
                "train",
            )
            valid_metrics, _ = self.predict_and_save_split(
                model,
                valid_loader,
                fold_stats,
                fold,
                "valid",
            )
            test_metrics, test_df = self.predict_and_save_split(
                model,
                test_loader,
                fold_stats,
                fold,
                "test",
            )
            oof_frames.append(test_df)

            fold_results.append(
                {
                    "fold": fold,
                    "fold_train_seed": fold_seed,
                    "n_train": len(train_idx),
                    "n_valid": len(valid_idx),
                    "n_test": len(test_idx),
                    "best_epoch": int(best_state["epoch"]),
                    "train_y_mean": y_mean,
                    "train_y_std": y_std,
                    "best_valid_loss": best_valid_loss,
                    "train_loss": train_metrics["loss"],
                    "train_r2_logk": train_metrics["r2"],
                    "train_mse_logk": train_metrics["mse"],
                    "train_rmse_logk": train_metrics["rmse"],
                    "train_mae_logk": train_metrics["mae"],
                    "valid_loss": valid_metrics["loss"],
                    "valid_r2_logk": valid_metrics["r2"],
                    "valid_mse_logk": valid_metrics["mse"],
                    "valid_rmse_logk": valid_metrics["rmse"],
                    "valid_mae_logk": valid_metrics["mae"],
                    "test_loss": test_metrics["loss"],
                    "test_r2_logk": test_metrics["r2"],
                    "test_mse_logk": test_metrics["mse"],
                    "test_rmse_logk": test_metrics["rmse"],
                    "test_mae_logk": test_metrics["mae"],
                }
            )

        if not fold_results:
            raise RuntimeError("没有运行任何 fold，请检查 cv.run_fold 设置。")

        summary_df = pd.DataFrame(fold_results)
        summary_csv = os.path.join(
            self.save_dir,
            "predictions",
            "cv_fold_summary.csv",
        )
        summary_df.to_csv(summary_csv, index=False)
        logging.info("Saved fold summary to: %s", summary_csv)

        logging.info("=" * 90)
        logging.info("Cross-validation fold summary")
        for row in fold_results:
            logging.info(
                "Fold %d | best_epoch=%d | "
                "Train R2=%.4f RMSE=%.4f MAE=%.4f | "
                "Valid R2=%.4f RMSE=%.4f MAE=%.4f | "
                "Test R2=%.4f RMSE=%.4f MAE=%.4f",
                row["fold"],
                row["best_epoch"],
                row["train_r2_logk"],
                row["train_rmse_logk"],
                row["train_mae_logk"],
                row["valid_r2_logk"],
                row["valid_rmse_logk"],
                row["valid_mae_logk"],
                row["test_r2_logk"],
                row["test_rmse_logk"],
                row["test_mae_logk"],
            )

        fold_r2_mean, fold_r2_std = self._mean_std(summary_df["test_r2_logk"])
        fold_rmse_mean, fold_rmse_std = self._mean_std(summary_df["test_rmse_logk"])
        fold_mae_mean, fold_mae_std = self._mean_std(summary_df["test_mae_logk"])

        logging.info(
            "Fold Mean ± SD (secondary): R2=%.4f ± %.4f, "
            "RMSE=%.4f ± %.4f, MAE=%.4f ± %.4f",
            fold_r2_mean,
            fold_r2_std,
            fold_rmse_mean,
            fold_rmse_std,
            fold_mae_mean,
            fold_mae_std,
        )

        if run_fold == 0:
            oof_df = pd.concat(oof_frames, ignore_index=True)
            oof_df = oof_df.sort_values("dataset_index").reset_index(drop=True)

            expected_ids = set(range(len(self.data)))
            actual_ids = oof_df["dataset_index"].astype(int).tolist()
            actual_id_set = set(actual_ids)

            duplicate_ids = sorted(
                oof_df.loc[
                    oof_df["dataset_index"].duplicated(keep=False),
                    "dataset_index",
                ].astype(int).unique().tolist()
            )
            missing_ids = sorted(expected_ids - actual_id_set)
            unexpected_ids = sorted(actual_id_set - expected_ids)

            if len(oof_df) != len(self.data):
                raise RuntimeError(
                    f"OOF 行数={len(oof_df)}，但数据集样本数={len(self.data)}。"
                    f" 重复ID={duplicate_ids}，缺失ID={missing_ids}，异常ID={unexpected_ids}"
                )

            if duplicate_ids or missing_ids or unexpected_ids:
                raise RuntimeError(
                    "OOF 样本编号检查失败："
                    f"重复ID={duplicate_ids}，缺失ID={missing_ids}，异常ID={unexpected_ids}"
                )

            oof_metrics = self.calculate_metrics_numpy(
                oof_df["true_logk"].to_numpy(),
                oof_df["pred_logk"].to_numpy(),
            )

            oof_csv = os.path.join(
                self.save_dir,
                "predictions",
                "oof_predictions_all_samples.csv",
            )
            oof_df.to_csv(oof_csv, index=False)

            metrics_df = pd.DataFrame(
                [
                    {
                        "n_samples": len(oof_df),
                        "oof_r2_logk": oof_metrics["r2"],
                        "oof_mse_logk": oof_metrics["mse"],
                        "oof_rmse_logk": oof_metrics["rmse"],
                        "oof_mae_logk": oof_metrics["mae"],
                        "fold_test_r2_mean": fold_r2_mean,
                        "fold_test_r2_std": fold_r2_std,
                        "fold_test_rmse_mean": fold_rmse_mean,
                        "fold_test_rmse_std": fold_rmse_std,
                        "fold_test_mae_mean": fold_mae_mean,
                        "fold_test_mae_std": fold_mae_std,
                        "split_seed": self.split_seed,
                        "base_train_seed": self.train_seed,
                    }
                ]
            )
            metrics_csv = os.path.join(
                self.save_dir,
                "predictions",
                "oof_metrics.csv",
            )
            metrics_df.to_csv(metrics_csv, index=False)

            logging.info("=" * 90)
            logging.info(
                "FINAL OOF metrics on all %d unseen predictions, log10(k) scale: "
                "R2=%.4f, MSE=%.6f, RMSE=%.6f, MAE=%.6f",
                len(oof_df),
                oof_metrics["r2"],
                oof_metrics["mse"],
                oof_metrics["rmse"],
                oof_metrics["mae"],
            )
            logging.info("Saved OOF predictions to: %s", oof_csv)
            logging.info("Saved OOF metrics to: %s", metrics_csv)

        return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="PFAS log10(k) regression with leakage-free fixed K-fold CV"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file",
    )
    args = parser.parse_args()

    trainer = PFASSeqTrainerCV(config_path=args.config)
    trainer.run_5fold_cv()

if __name__ == "__main__":
    main()
