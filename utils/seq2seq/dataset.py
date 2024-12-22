import os
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.seq2seq.tools import pad_1D, pad_2D

class TrainDataset(Dataset):
    def __init__(self, filename, config, device, sort=False, drop_last=False):
        self.preprocessed_path = config["path"]["root_path"]
        self.nam_hubert_path = config["path"]["nam_hubert_path"]
        self.simulated_speech_hubert_path = config["path"]["simulated_speech_hubert_path"]
        self.batch_size = config["optimizer"]["batch_size"]
        self.device = device

        with open(os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.basename = [line.split()[0] for line in lines]

        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.basename)

    def __getitem__(self, idx):
        basename = self.basename[idx]

        # Read target simulated speech Hubert features
        target_hubert_path = os.path.join(self.simulated_speech_hubert_path, f"{basename}.npy")
        target_hubert_features = np.load(target_hubert_path).astype(np.float32)

        # Read input NAM Hubert features
        input_hubert_path = os.path.join(self.nam_hubert_path, f"{basename}.npy")
        input_hubert_features = np.load(input_hubert_path).astype(np.float32)

        # Read input CTC tokens
        ctc_path = os.path.join(self.preprocessed_path, "ASR_tokens_character", f"{basename}.npy")
        ctc_label = np.load(ctc_path).astype(np.int64)

        # Align lengths if necessary
        if input_hubert_features.shape[0] != target_hubert_features.shape[0]:
            min_len = min(input_hubert_features.shape[0], target_hubert_features.shape[0])
            input_hubert_features = input_hubert_features[:min_len]
            target_hubert_features = target_hubert_features[:min_len]

        sample = {
            "id": basename,
            "input": input_hubert_features,
            "target": target_hubert_features,
            "ctc_label": ctc_label
        }
        return sample


    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        inputs = [data[idx]["input"] for idx in idxs]
        targets = [data[idx]["target"] for idx in idxs]
        ctc_labels = [data[idx]["ctc_label"] for idx in idxs]

        input_lens = np.array([input.shape[0] for input in inputs])
        targets_lens = np.array([target.shape[0] for target in targets])
        ctc_labels_lens = np.array([ctc_label.shape[0] for ctc_label in ctc_labels])

        inputs = pad_2D(inputs)
        targets = pad_2D(targets)
        ctc_labels = pad_1D(ctc_labels)

        return (
            ids,
            inputs,
            input_lens,
            max(input_lens),
            targets,
            targets_lens,
            max(targets_lens),
            ctc_labels,
            ctc_labels_lens,
            max(ctc_labels_lens)
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = torch.tensor([d["input"].shape[0] for d in data], device=self.device)
            idx_arr = torch.argsort(-len_arr).cpu().numpy()
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size):]
        idx_arr = idx_arr[:len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = [self.reprocess(data, idx) for idx in idx_arr]

        return output

class InferDataset(Dataset):
    def __init__(self, filepath, config):
        self.preprocessed_path = config["path"]["root_path"]
        self.input_hubert_path = config["path"]["nam_hubert_path"]

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.basename = [line.split()[0] for line in lines]

    def __len__(self):
        return len(self.basename)

    def __getitem__(self, idx):
        basename = self.basename[idx]

        input_hubert_path = os.path.join(self.input_hubert_path, f"{basename}.npy")
        input_hubert_features = np.load(input_hubert_path)

        return basename, input_hubert_features

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        inputs = [d[1] for d in data]
        input_lens = np.array([input.shape[0] for input in inputs])

        inputs = pad_2D(inputs)

        return ids, inputs, input_lens, max(input_lens)
