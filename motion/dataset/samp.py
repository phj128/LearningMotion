import copy
import os
import os.path as osp
import numpy as np
import torch
from motion.dataset.utils import load_norm_data

import motion.utils.utils as utils
from motion.utils.utils import pd_load, to_tensor, to_cpu_numpy
from motion.dataset.dataset import BaseDataset
from motion.dataset.builder import DATASETS


@DATASETS.register_module()
class SAMPData(BaseDataset):
    def _load_cfg(self, cfg):
        super()._load_cfg(cfg)
        self.data_dir = cfg.data_dir
        self.L = cfg.L
        self.state_dim = cfg.state_dim
        self.is_scheduled_sampling = cfg.is_scheduled_sampling

    def _load_data(self):
        print(f"Load data from {self.data_dir}")
        self.input_data = pd_load(
            osp.join(self.data_dir, self.split, "Input.txt")
        ).to_numpy()
        self.output_data = pd_load(
            osp.join(self.data_dir, self.split, "Output.txt")
        ).to_numpy()
        self.sequences = pd_load(
            osp.join(self.data_dir, self.split, "Sequences.txt")
        ).to_numpy()[:, 0]

        (
            self.input_mean,
            self.input_std,
            self.output_mean,
            self.output_std,
        ) = load_norm_data(self.data_dir)

        self.input_mean = to_tensor(self.input_mean)
        self.input_std = to_tensor(self.input_std)
        self.output_mean = to_tensor(self.output_mean)
        self.output_std = to_tensor(self.output_std)

    def _preprocess_dataset(self):
        super()._preprocess_dataset()
        if self.split == "train" and self.is_scheduled_sampling:
            self._process_train_data()
        else:
            pass

    def _process_train_data(self):
        N = self.input_data.shape[0]
        L = self.L
        self.input_data = torch.tensor(self.input_data, dtype=torch.float32).split(L)
        self.output_data = torch.tensor(self.output_data, dtype=torch.float32).split(L)
        self.sequences = torch.tensor(self.sequences, dtype=torch.float32).split(L)

        if N % self.L != 0:
            self.input_data = self.input_data[:-1]
            self.output_data = self.output_data[:-1]
            self.sequences = self.sequences[:-1]

        # Each rollout should contains frames from the same motion sequence only
        def _is_valid_data(x):
            valid = False
            if x[0] == x[-1]:
                valid = True
            return valid

        valid_ids = []
        for i, seq in enumerate(self.sequences):
            if _is_valid_data(seq):
                valid_ids.append(i)
        valid_ids = torch.tensor(valid_ids, dtype=torch.long)
        print(
            "Total num of rollouts {}, valid {}, invalid {} for {}".format(
                len(self.sequences),
                valid_ids.shape[0],
                len(self.sequences) - valid_ids.shape[0],
                self.split,
            )
        )

        self.input_data = [self.input_data[id] for id in valid_ids]
        self.output_data = [self.output_data[id] for id in valid_ids]
        self.sequences = [self.sequences[id] for id in valid_ids]

    def _load_data_instance(self, idx):
        x = to_tensor(self.input_data[idx])
        y = to_tensor(self.output_data[idx])
        ind = to_tensor(self.sequences[idx])
        data = {"x": x, "y": y, "ind": ind}
        return data

    def _load_statistics(self):
        return {
            "input_mean": self.input_mean,
            "input_std": self.input_std,
            "output_mean": self.output_mean,
            "output_std": self.output_std,
        }

    def _normalize_data(self, data):
        data["x"] = utils.normalize(data["x"], self.input_mean, self.input_std)
        data["y"] = utils.normalize(data["y"], self.output_mean, self.output_std)
        return data

    def _process_data(self, data):
        x, y = data["x"], data["y"]
        x1 = x[..., : self.state_dim]
        x2 = x[..., self.state_dim :]
        return_data = {"p_prev": x1, "I": x2, "y": y}
        return return_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        data = self._load_data_instance(idx)
        data = self._normalize_data(data)
        data = self._process_data(data)
        data.update(self._load_statistics())
        return data
