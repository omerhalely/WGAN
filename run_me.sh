# Training WGAN on CIFAR10 dataset.
python3 main.py --model "WGAN" --model-name "WGAN_CIFAR10" --data "CIFAR10" --epochs 100 --train true
# Testing WGAN_CIFAR10.
python3 main.py --model "WGAN" --model-name "WGAN_CIFAR10" --data "CIFAR10"

# Training WGAN on FashionMNIST dataset.
python3 main.py --model "WGAN" --model-name "WGAN_FashionMNIST" --data "FashionMNIST" --epochs 100 --train True
# Testing WGAN_FashionMNIST.
python3 main.py --model "WGAN" --model-name "WGAN_FashionMNIST" --data "FashionMNIST"


# Training DCGAN on CIFAR10 dataset.
python3 main.py --model "DCGAN" --model-name "DCAN_CIFAR10" --data "CIFAR10" --epochs 30 --train True
# Testing WGAN_CIFAR10.
python3 main.py --model "DCGAN" --model-name "DCGAN_CIFAR10" --data "CIFAR10"

# Training WGAN on FashionMNIST dataset.
python3 main.py --model "DCGAN" --model-name "DCGAN_FashionMNIST" --data "FashionMNIST" --epochs 30 --train True
# Testing WGAN_FashionMNIST.
python3 main.py --model "DCGAN" --model-name "DCGAN_FashionMNIST" --data "FashionMNIST"



