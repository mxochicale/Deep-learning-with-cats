Installation of Pytorth and tensorflow
---



Installing pytorch and tensorflow by following 
[1](https://hackernoon.com/up-and-running-with-ubuntu-nvidia-cuda-cudnn-tensorflow-and-pytorch-a54ec2ec907d) and [2](https://github.com/mxochicale/tensorflow/blob/master/installation/yadav_installation.md).


# Prepare computer

```
sudo apt-get update
sudo apt-get -y upgrade 
```


```
sudo apt-get install -y build-essential cmake g++ gfortran
sudo apt-get install -y git pkg-config python-dev
sudo apt-get install -y build-essential libssl-dev libffi-dev python3-dev #for python3
sudo apt-get install -y software-properties-common wget
sudo apt-get -y autoremove
sudo rm -rf /var/lib/apt/lists/*
```




# GPU

```
sudo lshw -C display | grep product
       product: GM206 [GeForce GTX 960]
```


GPU: GeForce GTX 960
Compute Capability: 5.2

https://developer.nvidia.com/cuda-gpus




# Nvidia Drivers


```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
```

Installing nvidia-390 drivers can cause some issues [more](https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-390/+bug/1773113).
Therefore, let's install nvidia-375 which were tested [on Ubuntu 16.04 with GTX960](https://github.com/mxochicale/tensorflow/blob/master/installation/yadav_installation.md)


```
sudo apt-get install nvidia-375
```





