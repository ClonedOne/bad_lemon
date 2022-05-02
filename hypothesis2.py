"""
This module contains the experiment to test the second hypothesis:
    Given a number of different classification models, trained on the same data,
    and having selected one as the victim, pairs of model with similar architectures
    will have lower Zest distances.
"""

import gc
import torch
import numpy as np

from tqdm.auto import tqdm

from zest import utils
from zest import model
from zest import train


# Experiment settings
dataset = 'CIFAR10'
base_model_path = '/home/giorgioseveri/projects/advml/lemon/cifar10_models/state_dicts/{}.pt'
victim_model = 'densenet121'
proxies = ['vgg11_bn', 'resnet18', 'resnet50', 'densenet161', 'googlenet']
b_size = 128
dist = ['1', '2', 'inf', 'cos']


# Load the victim model
victim_arch = eval(f"model.{victim_model}")
train_fn1 = train.TrainFn(batch_size=b_size, dataset=dataset, architecture=victim_arch)
print('Base model: {}'.format(victim_model))
train_fn1.load(base_model_path.format(victim_model))
train_fn1.lime()

# For each possible proxy model, load it and compute the distance
for p in tqdm(proxies, desc='Computing distances'):
    print('\nEvaluating distance with model {}'.format(p))
    arch_2 = eval(f"model.{p}")

    train_fn2 = train.TrainFn(batch_size=b_size, dataset=dataset, architecture=arch_2)
    train_fn2.load(base_model_path.format(p))
    train_fn2.lime()

    distance = np.array(utils.parameter_distance(train_fn1.lime_mask, train_fn2.lime_mask, order=dist, lime=True))
    print('Distance between {} and {}: {}'.format(victim_model, p, distance))

    del train_fn2
    gc.collect()
    torch.cuda.empty_cache()