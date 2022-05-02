# Bad Lemon

Experiments with faster black box evasion attacks.

This set of experiments relies on code from the following repositories:
- [Zest Model Distance](https://github.com/cleverhans-lab/Zest-Model-Distance). This is the repository of the paper "A Zest of LIME: Towards Architecture-Independent Model Distances" by Jia et al. published at ICLR 2022.
- [PyTorch CIFAR-10](https://github.com/huyvnphan/PyTorch_CIFAR10). This repository provides implementations and pre-trained models for multiple CIFAR-10 classifiers.


## Objectives

The model similarity technique shown in Jia et al. 2022 provides an architecture agnostic, gray-box --only requires access to some training points and the output logits of the victim model-- method to find a proxy model that is similar to a target deployed classifier. 

This information can be used to significantly reduce the operational costs of generating adversarial examples for the victim model. An adversary could acquire a relatively large number of potential proxy candidates (pre-trained models from model hubs, or self trained models), and find the closest one to the victim. They could then craft evasive samples on this classifier using first order, white-box, attacks, and have a much higher expectation that the generated adversarial examples will transfer.


## Usage

### Hypothesis 1 

There is a noticeable advantage in trying to transfer adversarial samples computed on similar architectures.

To show this `hypothesis1.py` will slect 100 correct classified test data points for a victim model (`densenet121`) and then generate evasive samples on 5 different proxies, with PGD.

Run the script with `ptyhon hypothesis1.py` and it will produce an output similar to this:
```
Evaluating generated samples on the original victim model
Evaluating samples generated on: vgg11_bn
Evasion success rate: 0.29000000000000004
Evaluating samples generated on: resnet18
Evasion success rate: 0.31000000000000005
Evaluating samples generated on: resnet50
Evasion success rate: 0.41000000000000003
Evaluating samples generated on: densenet161
Evasion success rate: 0.51
Evaluating samples generated on: googlenet
Evasion success rate: 0.10999999999999999
```

Samples generated on `densenet161` are, as expected, significantly easier to transfer than those generated on different architectures.

### Hypothesis 2

Models of a similar architecture have smaller Zest distances.

Run the script with `ptyhon hypothesis2.py` and it will produce an output similar to this:
```
Distance between densenet121 and vgg11_bn: [4.10939111e+03 4.79590988e+01 4.05846262e+00 2.84967363e-01]
...
Distance between densenet121 and resnet18: [2.55898926e+03 3.33659401e+01 3.14345694e+00 1.79761827e-01]
...
Distance between densenet121 and resnet50: [2.26656641e+03 3.10252571e+01 2.91031361e+00 1.45045757e-01]
...
Distance between densenet121 and densenet161: [1.96612476e+03 2.82449207e+01 2.52590132e+00 1.40515983e-01]
...
Distance between densenet121 and googlenet: [2.93570166e+03 3.95697975e+01 3.66321969e+00 2.62254894e-01]
```

### Hypothesis 3

Smaller Zest distances are correlated with higher adversarial transferability.

