# GAN (Generative Adversarial Nets)

An implementation of Generative Adversarial Nets

## requirements
* python version : 3.5.6
* tensorflow version : 1.10.0

# Examples

![output](./images/samples55.png)

# Discussion



* When optimizing to become balanced of generator and discriminator, the learning rate between generator and discriminator could set differently by user configuration. But it is quite difficult to make balanced between generator and discriminator.


* We can find mode collapsing problem on the example shown above. The model generated 0 in most cases. During several tests, the model generated no digits other than 0 and 8 in most cases.