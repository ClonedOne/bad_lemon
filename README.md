# Bad Lemon

Experiments with faster black box evasion attacks.

This set of experiments relies on code from the following repositories:
- [Zest Model Distance](https://github.com/cleverhans-lab/Zest-Model-Distance). This is the repository of the paper "A Zest of LIME: Towards Architecture-Independent Model Distances" by Jia et al. published at ICLR 2022.
- [PyTorch CIFAR-10](https://github.com/huyvnphan/PyTorch_CIFAR10). This repository provides implementations and pre-trained models for multiple CIFAR-10 classifiers.


## Objectives

The model similarity technique shown in Jia et al. 2022 provides an architecture agnostic, gray-box --only requires access to some training points and the output logits of the victim model-- method to find a proxy model that is similar to a target deployed classifier. 

This information can be used to significantly reduce the operational costs of generating adversarial examples for the victim model. An adversary could acquire a relatively large number of potential proxy candidates (pre-trained models from model hubs, or self trained models), and find the closest one to the victim. They could then craft evasive samples on this classifier using first order, white-box, attacks, and have a much higher expectation that the generated adversarial examples will transfer.


## Usage

