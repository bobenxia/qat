import collections
import os

import numpy as np
import random

import torch
from .build_dataset import build_train_dataset_and_sampler, build_valid_dataset_and_sampler

def build_train_loader(dataset_train, train_sampler, config, distributed):
    batch_size = config.DATALOADER.BATCH_SIZE
    num_workers = config.DATALOADER.NUM_WORKERS
    g = torch.Generator()
    g.manual_seed(0)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=batch_size,
        sampler=train_sampler, 
        num_workers=num_workers, 
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=True
    )

    return data_loader_train

def build_valid_loader(dataset_valid, valid_sampler, config, distributed):
    batch_size = config.DATALOADER.BATCH_SIZE
    num_workers = config.DATALOADER.NUM_WORKERS

    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, 
        batch_size=batch_size,
        sampler=valid_sampler, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True
    )

    return data_loader_valid

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)