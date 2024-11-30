**17. Generative Adversarial Networks:**

1.1 **Adversarial Training**
* Generative models and deep learning: The subchapter introduces the concept of generative models and how deep learning has significantly improved their performance, particularly in complex real-world applications.
* Adversarial training concept: The subchapter explains the core idea of GANs, involving a generator and a discriminator network in a zero-sum game scenario.
* Unsupervised learning as supervised learning:  The adversarial training approach transforms the unsupervised density modelling problem into a form of supervised learning by using the discriminator network to provide a training signal.


1.1.1 **Loss Function**
* Binary target variable: The subchapter defines a binary target variable (t) to represent real (t=1) or synthetic (t=0) data.
* Discriminator probability and cross-entropy: The discriminator network outputs the probability of data being real using a logistic-sigmoid function, and the training utilizes the standard cross-entropy error function.
* GAN error function: The subchapter describes the specific form of the GAN error function (17.6), which involves separate terms for real and synthetic data.
* Adversarial optimization: The subchapter emphasizes the unique aspect of GAN training where the error is minimized with respect to the discriminator's parameters (φ) but maximized with respect to the generator's parameters (w).


1.1.2 **GAN Training in Practice**
* Practical training challenges: The subchapter discusses the difficulties in training GANs, such as the lack of a progress metric and the phenomenon of mode collapse.
* Optimal discriminator and gradient vanishing: The subchapter explains how the optimal discriminator function can have vanishing gradients near data points, leading to slow learning for the generator network.
* Smoothing techniques: The subchapter presents methods for smoothing the discriminator function, including least-squares GAN, instance noise, and modified error functions (17.10).
* Wasserstein distance: The subchapter introduces the Wasserstein distance as a measure of the difference between distributions and its application in Wasserstein GAN (WGAN) and gradient penalty Wasserstein GAN (WGAN-GP).


1.2 **Image GANs**
* Convolutional networks in GANs: The subchapter discusses the advantages of using convolutional networks for image generation, particularly for the generator and discriminator networks.
* Transpose convolutions: The use of transpose convolutions in the generator network for mapping a lower-dimensional latent space to a high-resolution image is highlighted.
* Progressive growing of GANs: The subchapter explains the technique of progressively growing GAN architectures for generating high-resolution images efficiently.
* BigGAN architecture: The subchapter introduces BigGAN, a complex GAN architecture for class-conditional image generation.


1.2.1 **CycleGAN**
* Image-to-image translation: The subchapter describes CycleGAN, a GAN architecture for image-to-image translation between different domains (e.g., photographs and paintings).
* Bijective mappings and conditional generators: CycleGAN's use of two conditional generators and two discriminators to learn bijective mappings between domains is explained.
* Cycle consistency error: The subchapter introduces the cycle consistency error, an additional term in the loss function to ensure consistent image translations between domains and back.
* Representation learning with GANs: The subchapter discusses how GANs can be used for representation learning, revealing semantically meaningful structure in data.
* Disentangled representations: The concept of disentangled representations in GANs, allowing for generating images with specific attributes, is explained.