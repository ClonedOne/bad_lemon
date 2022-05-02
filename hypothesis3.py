"""
This module contains code to test if the third hypothesis holds:
    Given a number of different classification models, trained on the same data,
    and having selected one as the victim, there is negative correlation between the Zest
    distance and the transfer rate of the adversarial examples.
"""

import os
import gc
from socket import AF_X25
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
victim_model = 'densenet161'
proxies = [
    'vgg11_bn',
    'vgg13_bn',
    'vgg16_bn',
    'vgg19_bn',
    'resnet18',
    'resnet34',
    'resnet50',
    'densenet121',
    'densenet169',
    'mobilenet_v2',
    'googlenet',
    'inception_v3'
]
b_size = 128
dist = ['1', '2', 'inf', 'cos']

# Setup
proxies = sorted(proxies)
os.makedirs('results', exist_ok=True)
distances_file_name = 'distances_v_{}_p_{}.npy'.format(victim_model, '_'.join(proxies))
distances_file = os.path.join('results', distances_file_name)

# Load the victim model
victim_arch = eval(f"model.{victim_model}")
train_fn1 = train.TrainFn(
    batch_size=b_size, dataset=dataset, architecture=victim_arch)
print('Base model: {}'.format(victim_model))
train_fn1.load(base_model_path.format(victim_model))


# ZEST DISTANCE ESTIMATION

# Check if the distances have already been computed
if os.path.exists(distances_file):
    print('Distances already computed. Loading them...')
    distances = np.load(distances_file, allow_pickle=True).item()

else:
    train_fn1.lime()
    distances = {}

    # For each possible proxy model, load it and compute the distance
    for p in tqdm(proxies, desc='Computing distances'):
        print('\nEvaluating distance with model {}'.format(p))
        arch_2 = eval(f"model.{p}")

        train_fn2 = train.TrainFn(
            batch_size=b_size, dataset=dataset, architecture=arch_2)
        train_fn2.load(base_model_path.format(p))
        train_fn2.lime()

        distance = np.array(utils.parameter_distance(
            train_fn1.lime_mask, train_fn2.lime_mask, order=dist, lime=True))
        print('Distance between {} and {}: {}'.format(victim_model, p, distance))
        distances[p] = distance

        del train_fn2
        gc.collect()
        torch.cuda.empty_cache()

    np.save(distances_file, distances)


# PREDICT EXPECTED BEST PROXY MODEL

dists = []
for p in proxies:
    if p in distances:
        dists.append(distances[p])
dists = np.stack(dists)
print('\nDistances: {}'.format(dists))

min_idx = np.argmin(dists[:, -1]) # Cosine distance
predicted = proxies[min_idx]
print('\nPredicted best proxy model: {}'.format(predicted))


# COMPUTE ADVERSARIAL EXAMPLES

# Utility to load a model
def load_model(m):
    arch = eval(f"model.{m}")
    net = arch()
    net.load_state_dict(torch.load(base_model_path.format(m)))
    net = net.eval()
    return net

testset = utils.load_dataset(dataset, False, download=True)
tst_ld = torch.utils.data.DataLoader(testset, batch_size=b_size, shuffle=False, num_workers=32, pin_memory=True)

# Load victim model
net = load_model(victim_model)

# Select 100 correctly classified samples from the test set
correct_idxs = []
tst_imgs = []
tst_labels = []
for i, batch in tqdm(enumerate(tst_ld, 0), desc='Selecting 100 samples'):
    b_x, b_y = batch
    b_preds = torch.argmax(net(b_x), dim=1).numpy()

    b_correct = np.where(b_preds == b_y.numpy())[0]
    tst_imgs.append(b_x[b_correct])
    tst_labels.append(b_y[b_correct])

    b_correct = [(i * b_size) + j for j in b_correct]
    correct_idxs += b_correct

    if len(correct_idxs) >= 100:
        break

correct_idxs =  np.array(correct_idxs)[:100]
tst_imgs = torch.cat(tst_imgs)[:100]
tst_labels = torch.cat(tst_labels)[:100]
tst_imgs = tst_imgs.numpy()
tst_labels = tst_labels.numpy()
print('Selected points: {} - {}'.format(tst_imgs.shape, tst_labels.shape))

# Cleanup
del net
gc.collect()
torch.cuda.empty_cache()


# Utility to wrap a model in a PyTorchClassifier
def wrap_model(m):
    criterion = torch.nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(
        model=m,
        loss=criterion,
        clip_values=(np.min(tst_imgs), np.max(tst_imgs)),
        input_shape=tst_imgs.shape[1:],
        nb_classes=10,  # CIFAR10
    )
    return classifier

# Check if advex file is already present
advex_file_name = 'advex_v_{}_p_{}.npy'.format(victim_model, '_'.join(proxies))
advex_file = os.path.join('results', advex_file_name)

if os.path.exists(advex_file):
    print('Adversarial examples already computed. Loading them...')
    res_dict = np.load(advex_file, allow_pickle=True).item()
    adv_exs = res_dict['adv_exs']

else:
    adv_exs = {}

    for p in proxies:
        print('\nGenerating adv examples using: {}'.format(p))
        net = load_model(p)
        classifier = wrap_model(net)  # ART wrapper

        # Attack
        attack = ProjectedGradientDescentPyTorch(classifier, eps=0.1, eps_step=0.02, num_random_init=5)
        x_adv = attack.generate(x=tst_imgs)
        adv_exs[p] = x_adv

        # Evaluate success rate
        x_adv_preds = np.argmax(classifier.predict(x_adv), axis=-1)
        assert len(tst_labels) == len(x_adv_preds)
        success_rate = 1 - np.sum(x_adv_preds == tst_labels) / len(x_adv_preds)
        print('Approximate success rate: {}'.format(success_rate))

        del net, classifier
        gc.collect()
        torch.cuda.empty_cache()

    res_dict = {
        'victim': victim_model,
        'dataset': dataset,
        'proxies': proxies,
        'adv_exs': adv_exs,
        'test_images': correct_idxs,
    }

    # Save results
    np.save(advex_file, res_dict)


# Find the best proxy model

print('\nEvaluating generated samples on the original victim model')
net = load_model(victim_model)
classifier = wrap_model(net)

successes = []
for p in proxies:
    adv = adv_exs[p]
    print('Evaluating samples generated on: {}'.format(p))
    adv_preds = np.argmax(classifier.predict(adv), axis=-1)
    assert len(tst_labels) == len(adv_preds)
    success_rate = 1 - np.sum(adv_preds == tst_labels) / len(tst_labels)
    print('Evasion success rate: {}'.format(success_rate))
    successes.append(success_rate)

best_idx = np.argmax(successes)
best_proxy = proxies[best_idx]
print('\nBest proxy model: {}'.format(best_proxy))
