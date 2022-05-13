"""
This module contains code to compute adversarial examples for each of the proxies
over a subset of the test set.
Use `find_test_set.py` to find a fixed-size subset of the test set that all
13 pre-trained CIFAR-10 models are successfully classifying.
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
eps = 0.1
eps_step = 0.02
num_random_init = 5

# Change this path to the result file you would like to use
agree_file = 'results/agree_CIFAR10_testset_100_samples_densenet121_densenet161_densenet169_googlenet_inception_v3_mobilenet_v2_resnet18_resnet34_resnet50_vgg11_bn_vgg13_bn_vgg16_bn_vgg19_bn.npy'

# Load data points to attack
agree_data = np.load(agree_file, allow_pickle=True).item()
b_x = agree_data['imgs']
b_y = agree_data['labels']

# Utility to load a model from disk
def load_model(m):
    arch = eval(f"model.{m}")
    net = arch()
    net.load_state_dict(torch.load(base_model_path.format(m)))
    net = net.eval()
    return net

# Utility to wrap a model in a ART PyTorchClassifier
def wrap_model(m, tst_imgs):
    criterion = torch.nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(
        model=m,
        loss=criterion,
        clip_values=(np.min(tst_imgs), np.max(tst_imgs)),
        input_shape=tst_imgs.shape[1:],
        nb_classes=10,  # CIFAR10
    )
    return classifier

# Compute the adversarial examples for each model
adv_exs = {}
for p in all_proxies:
    print('\nGenerating adv examples using: {}'.format(p))
    net = load_model(p)
    classifier = wrap_model(net, b_x)

    # Attack
    attack = ProjectedGradientDescentPyTorch(
        classifier, 
        eps=eps, 
        eps_step=eps_step, 
        num_random_init=num_random_init,
        batch_size=b_size,
    )
    x_adv = attack.generate(x=b_x)
    adv_exs[p] = x_adv

    # Evaluate success rate
    x_adv_preds = np.argmax(classifier.predict(x_adv), axis=-1)
    assert len(b_x) == len(x_adv_preds)
    success_rate = 1 - np.sum(x_adv_preds == b_y) / len(x_adv_preds)
    print('Local success rate: {}'.format(success_rate))

    # Cleanup
    del net, classifier
    gc.collect()
    torch.cuda.empty_cache()

# Save the results
advex_file_name = 'advex_{}.npy'.format('_'.join(all_proxies))
advex_file = os.path.join('results', advex_file_name)
np.save(advex_file, adv_exs)

transfer = {}
# Check transfer success rate
for victim in all_proxies:
    print('\nGenerating adv examples using: {}'.format(victim))
    net = load_model(victim)
    classifier = wrap_model(net, b_x)

    successes = {}
    preds = {}
    for p in all_proxies:
        adv = adv_exs[p]
        print('Evaluating samples generated on: {}'.format(p))
        adv_preds = np.argmax(classifier.predict(adv), axis=-1)
        assert len(b_y) == len(adv_preds)
        preds[p] = adv_preds

        success_rate = 1 - np.sum(adv_preds == b_y) / len(b_y)
        print('Evasion success rate: {}'.format(success_rate))
        successes[p] = success_rate

    transfer[victim] = {
        'successes': successes,
        'preds': preds
    }

# Save the results
transfer_file_name = 'transfer_{}.npy'.format('_'.join(all_proxies))
transfer_file = os.path.join('results', transfer_file_name)
np.save(transfer_file, transfer)
