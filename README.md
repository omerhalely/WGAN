# WGAN
Implementation of a WGAN.

## Training
For training a model use the following line:
```bash
python main.py --model "model" --model-name "model-name" --data "data-type" --epochs 100 --train True
```
model is the model which will be trained (WGAN/DCGAN)

model-name is the name of the model which will be saved to ./models/model-name.

data is the type of the data which will be used in the training process (CIFAR10/FashionMNIST).

For example, training a WGAN over the CIFAR10 dataset:
```bash
python main.py --model "WGAN" --model-name "WGAN_CIFAR10" --data "CIFAR10" --epochs 100 --train True
```

## Testing
```bash
python main.py --model "model" --model-name "model-name" --data "data-type"
```
model is the model which will be trained (WGAN/DCGAN)

model-name is the name of the model which will be saved to ./models/model-name.

data is the type of the data which will be used in the training process (CIFAR10/FashionMNIST).

For example, testing the WGAN which was trained on the CIFAR10 dataset:
```bash
python main.py --model "WGAN" --model-name "WGAN_CIFAR10" --data "CIFAR10"
```

Samples of the output images of the WGAN will eb saved to ./models/model-name
