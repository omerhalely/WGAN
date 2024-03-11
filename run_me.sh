# Training WGAN on CIFAR10 dataset.
python main.py --model "WGAN" --model-name "WGAN_CIFAR10" --data "CIFAR10" --epochs 100 --train True
# Testing WGAN_CIFAR10.
python main.py --model "WGAN" --model-name "WGAN_CIFAR10" --data "CIFAR10"

# Training WGAN on FashionMNIST dataset.
python main.py --model "WGAN" --model-name "WGAN_FashionMNIST" --data "FashionMNIST" --epochs 100 --train True
# Testing WGAN_FashionMNIST.
python main.py --model "WGAN" --model-name "WGAN_FashionMNIST" --data "FashionMNIST"


# Training DCGAN on CIFAR10 dataset.
python main.py --model "DCGAN" --model-name "DCGAN_CIFAR10" --data "CIFAR10" --epochs 30 --train True
# Testing DCGAN_CIFAR10.
python main.py --model "DCGAN" --model-name "DCGAN_CIFAR10" --data "CIFAR10"

# Training DCGAN on FashionMNIST dataset.
python main.py --model "DCGAN" --model-name "DCGAN_FashionMNIST" --data "FashionMNIST" --epochs 30 --train True
# Testing DCGAN_FashionMNIST.
python main.py --model "DCGAN" --model-name "DCGAN_FashionMNIST" --data "FashionMNIST"
