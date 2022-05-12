"""
This module contains code to find a fixed-size subset of the test set that all
13 pre-trained CIFAR-10 models are successfully classifying.
It will save a numpy array of the indices of CIFAR-10 test set images that are
successfully classified by all 13 models.
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
all_proxies = sorted(all_proxies)
b_size = 128
num_samples = 100

def load_model(m):
    arch = eval(f"model.{m}")
    net = arch()
    net.load_state_dict(torch.load(base_model_path.format(m)))
    net = net.eval()
    return net

# Load the test set
testset = utils.load_dataset(dataset, False, download=True)
tst_ld = torch.utils.data.DataLoader(testset, batch_size=b_size, shuffle=False, num_workers=32, pin_memory=True)

# Load all models
models = [load_model(m) for m in all_proxies]

# Select num_samples correctly classified samples from the test set
correct_idxs = []
tst_imgs = []
tst_labels = []
for i, batch in tqdm(enumerate(tst_ld, 0), desc='Selecting 100 samples'):
    b_x, b_y = batch
    net_correct = []

    for net in tqdm(models, desc='Classifying samples'):
        b_preds = torch.argmax(net(b_x), dim=1).numpy()

        b_correct = np.where(b_preds == b_y.numpy())[0]
        net_correct.append(set(b_correct))
        
    correct = set.intersection(*net_correct)
    correct = np.array(list(correct), dtype=int)

    tst_imgs.append(b_x[correct])
    tst_labels.append(b_y[correct])
    correct = [(i * b_size) + j for j in correct]
    correct_idxs += correct

    if len(correct_idxs) >= num_samples:
        break

correct_idxs = np.array(correct_idxs)[:num_samples]
tst_imgs = torch.cat(tst_imgs)[:100]
tst_labels = torch.cat(tst_labels)[:100]
tst_imgs = tst_imgs.numpy()
tst_labels = tst_labels.numpy()
print('Selected points: {}'.format(correct_idxs))
print('Selected points shapes: {} - {}'.format(tst_imgs.shape, tst_labels.shape))

out_file_name = os.path.join(
    'results/',
    'agree_{}_testset_{}_samples_{}.npy'.format(dataset, num_samples, '_'.join(all_proxies))
)
print('Saving to file: {}'.format(out_file_name))
np.save(
    out_file_name, 
    {
        'idxs': correct_idxs,
        'imgs': tst_imgs,
        'labels': tst_labels
    }
)

# Sanity check - should be all equal to num_samples
outdict = np.load(out_file_name, allow_pickle=True).item()
b_x = torch.tensor(outdict['imgs'])
b_y = torch.tensor(outdict['labels'])
for net in tqdm(models, desc='Classifying samples'):
    b_preds = torch.argmax(net(b_x), dim=1).numpy()

    b_correct = np.where(b_preds == b_y.numpy())[0]
    assert b_correct.shape[0] == num_samples