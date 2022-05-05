"""
This module contains code to test if the third hypothesis holds:
    Given a number of different classification models, trained on the same data,
    and having selected one as the victim, there is negative correlation between the Zest
    distance and the transfer rate of the adversarial examples.
"""

import os
import gc
import torch
import numpy as np

from tqdm.auto import tqdm
from datetime import datetime
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescentPyTorch

from zest import utils
from zest import model
from zest import train


# Experiment settings
dataset = 'CIFAR10'
base_model_path = '/home/giorgioseveri/projects/advml/lemon/cifar10_models/state_dicts/{}.pt'
all_proxies = [
    'vgg11_bn',
    'vgg13_bn',
    'vgg16_bn',
    'vgg19_bn',
    'resnet18',
    'resnet34',
    'resnet50',
    'densenet121',
    'densenet161',
    'densenet169',
    'mobilenet_v2',
    'googlenet',
    'inception_v3'
]
b_size = 128
dist = ['1', '2', 'inf', 'cos']

# Setup
all_proxies = sorted(all_proxies)
os.makedirs('results', exist_ok=True)
distances_file_name = 'distances_all_{}.npy'.format('_'.join(all_proxies))
distances_file = os.path.join('results', distances_file_name)

distances = {}
train_fns = {}

for p in tqdm(all_proxies, desc='Computing representations'):
    p_arch = eval(f"model.{p}")
    train_fns[p] = train.TrainFn(batch_size=b_size, dataset=dataset, architecture=p_arch)
    train_fns[p].load(base_model_path.format(p))
    train_fns[p].lime()

for victim_model, train_fn1 in tqdm(train_fns.items(), desc='Evaluating target model'):
    distances[victim_model] = {}

    # For each possible proxy model, load it and compute the distance
    for p in tqdm(all_proxies, desc='Computing distances'):
        if p == victim_model:
            continue

        train_fn2 = train_fns[p]
        
        distance = np.array(utils.parameter_distance(
            train_fn1.lime_mask, train_fn2.lime_mask, order=dist, lime=True))
        print('Distance between {} and {}: {}'.format(victim_model, p, distance))
        distances[victim_model][p] = distance

        del train_fn2
        gc.collect()
        torch.cuda.empty_cache()

np.save(distances_file, distances)

