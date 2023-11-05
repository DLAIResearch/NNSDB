# NNSDB: Stealthy dynamic backdoor attack against neural networks for image classification
Our approach utilizes deep learning steganography to create manipulated images, capitalizing on the unique sensitivity of neural networks to minute perturbations. The network is then trained de novo with these manipulated images, creating a backdoor-infused model. 
![Image image](https://github.com/DLAIResearch/NNSDB/blob/main/NNSDB.png)
<br/>
## Pre-requisites
- Pytorch 1.5 - Please install [PyTorch](https://pytorch.org/get-started/locally/) and CUDA if you don't have it installed.
- ## Datasets
- [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)
- [GTSRB](https://benchmark.ini.rub.de/)
- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  list_attributes = [18, 31, 21]
- The above datasets can be downloaded directly using Pytorch.
- [ImageNet](https://www.image-net.org/download.php) We adopt 10 subclasses of Imagenet (n01530575, n02123045, n01978287, n02085620, n01440764, n01675722, n01728572, n01770081, n01664065, n02114367).

- ## Neural Network Steganography
-  [Fixed neural network steganography: Train the images, not the network](https://github.com/varshakishore/FNNS)
- ## Defense methods
- 1. Fine-pruning: Defending against backdooring attacks on deep neural networks
- 2. Neural cleanse: Identifying and mitigating backdoor attacks in neural networks
- 3. STRIP: a defence against trojan attacks on deep neural networks
- 4. Grad-cam: Visual explanations from deep networks via gradient-based localization
<br>
