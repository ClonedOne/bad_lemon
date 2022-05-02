"""
This module contains the experiment to test the first hypothesis:
    Given a number of different classification models trained on the same data,
    and having selected one as the victim, the adversarial examples computed
    on the other models will transfer to the victim model with different 
    success rates. In particular, we expect the models of a similar architecture
    to produce adversarial examples that transfer more easily.
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


# Experiment settings
dataset = 'CIFAR10'
base_model_path = '/home/giorgioseveri/projects/advml/lemon/cifar10_models/state_dicts/{}.pt'
victim_model = 'densenet121'
proxies = ['vgg11_bn', 'resnet18', 'resnet50', 'densenet161', 'googlenet']
b_size = 32

# Utility to load a model
def load_model(m):
    arch = eval(f"model.{m}")
    net = arch()
    net.load_state_dict(torch.load(base_model_path.format(m)))
    net = net.eval()
    return net


# Load data
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


# For each proxy:
#  - Load the model
#  - Compute the adversarial examples using projected gradient descent
#  - Compute the success rate
#  - Store the adversarial images
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


# Evaluate the generated samples on the original victim model
print('\nEvaluating generated samples on the original victim model')
net = load_model(victim_model)
classifier = wrap_model(net)

res_dict = {
    'victim': victim_model,
    'dataset': dataset,
    'proxies': proxies,
    'adv_exs': adv_exs,
    'success_rate': {},
    'test_images': correct_idxs,
}

for p, adv in adv_exs.items():
    print('Evaluating samples generated on: {}'.format(p))
    adv_preds = np.argmax(classifier.predict(adv), axis=-1)
    assert len(tst_labels) == len(adv_preds)
    success_rate = 1 - np.sum(adv_preds == tst_labels) / len(tst_labels)
    print('Evasion success rate: {}'.format(success_rate))
    res_dict['success_rate'][p] = success_rate


# Save results
ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
os.makedirs('results', exist_ok=True)
np.save('results/{}_{}_{}.npy'.format(victim_model, dataset, ts), res_dict)