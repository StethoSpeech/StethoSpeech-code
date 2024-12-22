import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.seq2seq.tools import to_device, log
from utils.seq2seq.model.loss import StethoSpeechLoss
from utils.seq2seq.dataset import TrainDataset

def evaluate(model, step, config, device, logger=None, vocoder=None):

    # Get dataset
    dataset = TrainDataset(
        "val.txt", config, device, sort=False, drop_last=False
    )
    batch_size = config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = StethoSpeechLoss().to(device)

    # Evaluation
    loss_sums = [0 for _ in range(6)]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                output = model(*(batch[1:]))

                # Cal Loss
                losses = Loss(step, batch, output)

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.6f}, MSE Loss: {:.4f}, CTC Loss: {:.4f}".format(
        *([step] + [l for l in loss_means])
    )
    log(logger, step, losses=loss_means)

    return message
