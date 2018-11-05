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

Installing nvidia-390 drivers can cause some issues 
in Ubuntu 18.04 (See [bugs](https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-390/+bug/1773113) ).
Therefore, let's install nvidia-375 which were tested [on Ubuntu 16.04 with GTX960](https://github.com/mxochicale/tensorflow/blob/master/installation/yadav_installation.md)


```
sudo apt-get -y install nvidia-375
```


verify nvidia driver installation

```
cat /proc/driver/nvidia/version
nvidia-smi
```

> NB. It can be noted that nvidia-375 were installed
but the installed version is 390. (See below)

## output

```

$ cat /proc/driver/nvidia/version
NVRM version: NVIDIA UNIX x86_64 Kernel Module  390.87  Tue Aug 21 12:33:05 PDT 2018
GCC version:  gcc version 7.3.0 (Ubuntu 7.3.0-27ubuntu1~18.04) 



$ nvidia-smi
Fri Nov  2 10:51:26 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.87                 Driver Version: 390.87                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 960     Off  | 00000000:01:00.0  On |                  N/A |
| 42%   31C    P8    11W / 130W |    373MiB /  4035MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1034      G   /usr/lib/xorg/Xorg                            15MiB |
|    0      1099      G   /usr/bin/gnome-shell                          56MiB |
|    0      1345      G   /usr/lib/xorg/Xorg                           141MiB |
|    0      1481      G   /usr/bin/gnome-shell                          76MiB |
|    0      1884      G   ...uest-channel-token=15116432904883338830    79MiB |
+-----------------------------------------------------------------------------+

```



Getting GPU support to work requires a symphony of different hardware and software. 
To ensure compatibility we are going to ask Ubuntu to not update certain software. 
You can remove this hold at any time. Run:

```
sudo apt-mark hold nvidia-driver-390
```



# Installing CUDA 9.0

## kernel headers


```
Kernel *headers* are the header files used to compile the kernel - and other
applications which depend on the symbols / structures defined in these
header files, like kernel modules. An example can be graphic card drivers;
if the driver does not have a binary matching the running kernel and needs
to be compiled.
```


```
sudo apt install linux-headers-$(uname -r)
```

## download cuda and install it

https://developer.nvidia.com/cuda-90-download-archive


```
cd Downloads
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda-repo-ubuntu1704-9-0-local_9.0.176-1_amd64-deb
mv cuda-repo-ubuntu1704-9-0-local_9.0.176-1_amd64-deb cuda-repo-ubuntu1704-9-0-local_9.0.176-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1704-9-0-local_9.0.176-1_amd64.deb
sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
rm cuda-repo-ubuntu1704-9-0-local_9.0.176-1_amd64.deb
```

## hold this cuda package from updates:

```
sudo apt-mark hold cuda
```

## source cuda in bashrc

Lastly you need your PATH variable in your .bashrc file. 
This hidden file is located in your home directory. 
You need to add the following line to the end of your .bashrc file:
`export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}`
You can use any text editor you like, or nano or vim. but open your .bashrc file 
and add that to the end of the file save it and close. 
You will need to start a new terminal session (close and reopen your terminal) 
for the .bashrc file changes to get loaded.

```
cd #navigate back to home
vim .bashrc
```
add 
```
# cuda
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
```

source .bashrc	






## test cuda

```
cd #navigate back to home
cat /proc/driver/nvidia/version
```
output
```
NVRM version: NVIDIA UNIX x86_64 Kernel Module  390.87  Tue Aug 21 12:33:05 PDT 2018
GCC version:  gcc version 7.3.0 (Ubuntu 7.3.0-27ubuntu1~18.04) 
```

```
nvcc --version
```
output
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2017 NVIDIA Corporation
Built on Fri_Sep__1_21:08:03_CDT_2017
Cuda compilation tools, release 9.0, V9.0.176
```




# cuDNN

NVIDIA cuDNN is a GPU-accelerated library of primitives for deep neural networks.
(You have to sign up to download cuDNN).
It has been choosen  cuDNN version 7.0.5 over 7.1.4 based on what TensorFlow 
suggested for optimal compatibility at the time.



## download libraries and install them

Download the following packages and follow [cuDNN instructions](https://hackernoon.com/up-and-running-with-ubuntu-nvidia-cuda-cudnn-tensorflow-and-pytorch-a54ec2ec907d)


* cuDNN v7.0.5 Runtime Library for Ubuntu16.04 (Deb)
* cuDNN v7.0.5 Developer Library for Ubuntu16.04 (Deb)
* cuDNN v7.0.5 Code Samples and User Guide for Ubuntu16.04 (Deb)


After you have downloaded them, run these next commands:


```
cd Downloads/ #navigate to where you downloaded them
sudo dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-doc_7.0.5.15-1+cuda9.0_amd64.deb
```

output

```

$ sudo dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
[sudo] password for robot: 
Selecting previously unselected package libcudnn7.
(Reading database ... 192141 files and directories currently installed.)
Preparing to unpack libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb ...
Unpacking libcudnn7 (7.0.5.15-1+cuda9.0) ...
Setting up libcudnn7 (7.0.5.15-1+cuda9.0) ...
Processing triggers for libc-bin (2.27-3ubuntu1) ...

$ sudo dpkg -i libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
Selecting previously unselected package libcudnn7-dev.
(Reading database ... 192148 files and directories currently installed.)
Preparing to unpack libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb ...
Unpacking libcudnn7-dev (7.0.5.15-1+cuda9.0) ...
Setting up libcudnn7-dev (7.0.5.15-1+cuda9.0) ...
update-alternatives: using /usr/include/x86_64-linux-gnu/cudnn_v7.h to provide /usr/include/cudnn.h (libcudnn) in auto mode

$ sudo dpkg -i libcudnn7-doc_7.0.5.15-1+cuda9.0_amd64.deb
Selecting previously unselected package libcudnn7-doc.
(Reading database ... 192154 files and directories currently installed.)
Preparing to unpack libcudnn7-doc_7.0.5.15-1+cuda9.0_amd64.deb ...
Unpacking libcudnn7-doc (7.0.5.15-1+cuda9.0) ...
Setting up libcudnn7-doc (7.0.5.15-1+cuda9.0) ...

```




## test cuDNN installation


```
cd #back to home
cp -r /usr/src/cudnn_samples_v7/ $HOME #copy the test mnist files over to home.
cd $HOME/cudnn_samples_v7/mnistCUDNN
make clean && make
```

output

```
$ make clean && make
rm -rf *o
rm -rf mnistCUDNN
/usr/local/cuda/bin/nvcc -ccbin g++ -I/usr/local/cuda/include -IFreeImage/include  -m64    -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_53,code=compute_53 -o fp16_dev.o -c fp16_dev.cu
In file included from /usr/local/cuda/include/host_config.h:50:0,
                 from /usr/local/cuda/include/cuda_runtime.h:78,
                 from <command-line>:0:
/usr/local/cuda/include/crt/host_config.h:119:2: error: #error -- unsupported GNU version! gcc versions later than 6 are not supported!
 #error -- unsupported GNU version! gcc versions later than 6 are not supported!
  ^~~~~
Makefile:203: recipe for target 'fp16_dev.o' failed
make: *** [fp16_dev.o] Error 1

```

## gcc 6.x and g++ 6.x

Lets install gcc 6.x and g++ 6.x.
Then we can create a symlink to tell cuDNN to look for version 6. 
We can leave version 7 installed as well.


```
sudo apt -y install gcc-6 g++-6
    # ln -s makes a symbolic link so cuda looks for gcc version 6 where we tell it to look.
sudo ln -s /usr/bin/gcc-6 /usr/local/cuda/bin/gcc
sudo ln -s /usr/bin/g++-6 /usr/local/cuda/bin/g++
```

## test cuda with gcc6 and g++6 


```
make clean && make
```
output
```
$ make clean && make
rm -rf *o
rm -rf mnistCUDNN
/usr/local/cuda/bin/nvcc -ccbin g++ -I/usr/local/cuda/include -IFreeImage/include  -m64    -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_53,code=compute_53 -o fp16_dev.o -c fp16_dev.cu
g++ -I/usr/local/cuda/include -IFreeImage/include   -o fp16_emu.o -c fp16_emu.cpp
g++ -I/usr/local/cuda/include -IFreeImage/include   -o mnistCUDNN.o -c mnistCUDNN.cpp
/usr/local/cuda/bin/nvcc -ccbin g++   -m64      -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_53,code=compute_53 -o mnistCUDNN fp16_dev.o fp16_emu.o mnistCUDNN.o  -LFreeImage/lib/linux/x86_64 -LFreeImage/lib/linux -lcudart -lcublas -lcudnn -lfreeimage -lstdc++ -lm

```



cuDNN test

```
./mnistcuDNN
```
output


```
$ ./mnistCUDNN 
cudnnGetVersion() : 7005 , CUDNN_VERSION from cudnn.h : 7005 (7.0.5)
Host compiler version : GCC 6.4.0
There are 1 CUDA capable devices on your machine :
device 0 : sms  8  Capabilities 5.2, SmClock 1253.0 Mhz, MemSize (Mb) 4035, MemClock 3505.0 Mhz, Ecc=0, boardGroupID=0
Using device 0

Testing single precision
Loading image data/one_28x28.pgm
Performing forward propagation ...
Testing cudnnGetConvolutionForwardAlgorithm ...
Fastest algorithm is Algo 1
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.039040 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.043776 time requiring 3464 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.062464 time requiring 57600 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: 0.173984 time requiring 2057744 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 0.258240 time requiring 203008 memory
Resulting weights from Softmax:
0.0000000 0.9999399 0.0000000 0.0000000 0.0000561 0.0000000 0.0000012 0.0000017 0.0000010 0.0000000 
Loading image data/three_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000000 0.0000000 0.9999288 0.0000000 0.0000711 0.0000000 0.0000000 0.0000000 0.0000000 
Loading image data/five_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000008 0.0000000 0.0000002 0.0000000 0.9999820 0.0000154 0.0000000 0.0000012 0.0000006 

Result of classification: 1 3 5

Test passed!

Testing half precision (math in single precision)
Loading image data/one_28x28.pgm
Performing forward propagation ...
Testing cudnnGetConvolutionForwardAlgorithm ...
Fastest algorithm is Algo 1
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.032480 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.040800 time requiring 3464 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.048032 time requiring 28800 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: 0.170208 time requiring 2057744 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 0.274240 time requiring 203008 memory
Resulting weights from Softmax:
0.0000001 1.0000000 0.0000001 0.0000000 0.0000563 0.0000001 0.0000012 0.0000017 0.0000010 0.0000001 
Loading image data/three_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000000 0.0000000 1.0000000 0.0000000 0.0000714 0.0000000 0.0000000 0.0000000 0.0000000 
Loading image data/five_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000008 0.0000000 0.0000002 0.0000000 1.0000000 0.0000154 0.0000000 0.0000012 0.0000006 

Result of classification: 1 3 5

Test passed!
```


## hold packages delete files 
Hold these packages from being updated and remove 
(you can delete the .deb files as well in your Downloads/) 
the copied samples directory we used for the previous MNIST test:

```
sudo apt-mark hold libcudnn7 libcudnn7-dev libcudnn7-doc #These are the three .deb files we installed
rm -r cudnn_samples_v7/
cd Downloads
rm *.deb
```


# Deep Learning Frameworks


## conda


Check the lastest version of Anaconda at https://repo.anaconda.com/archive/

(See more [ [1] (sep2018) ](https://www.ceos3c.com/open-source/install-anaconda-ubuntu-18-04/) 
[ [2] ](https://linuxhint.com/install_anaconda_python_ubuntu_1804/) ).


```

cd /tmp/
wget https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh #04 November 2018
md5sum Anaconda3*.sh
bash Anaconda3*.sh #ENTER/yes/yes/no
source ~/.bashrc
rm Anaconda3*.sh
```


testing conda

```
conda --version
conda list
```


## tensorflow

Initialize the env with pip and not installing pip after the env has been created. 
There used to be (might still be) an insidious bug where if you installed pip 
after you created an env pip installs would install packages globally 
regardless if you were in an activated env. This is highly undesirable behavior. 


```
cd
conda create -n tensorflow python=3.6 pip numpy #remember to initialize the env with pip here.
conda activate tensorflow
```


```
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.8.0-cp36-cp36m-linux_x86_64.whl
conda list | grep tensor
```


output

```
$ conda list | grep tensor
# packages in environment at /tmp/yes/envs/tensorflow:
tensorboard               1.8.0                     <pip>
tensorflow-gpu            1.8.0                     <pip>
```


* test tensorflow


```
python
```
```
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
exit()
```

* now deactivate your tensorflow conda env 


```
conda deactivate
conda deactivate
```




## pytorch

* create conda env for pytorch

```

conda create -n pytorch python=3.6 pip numpy
conda activate pytorch

```
* install pytorch

```
conda install pytorch torchvision cuda90 -c pytorch
conda list | grep torch
```
output
```
$ conda list | grep torch
# packages in environment at /tmp/yes/envs/pytorch:
cuda90                    1.0                  h6433d27_0    pytorch
pytorch                   0.4.1           py36_py35_py27__9.0.176_7.1.2_2    pytorch
torchvision               0.2.1                    py36_1    pytorch
```





* test pytorch

```
python #start the python interpreter to test pytorch for GPU support
```


```
import torch
torch.cuda.current_device()
torch.cuda.device(0)
torch.cuda.get_device_name(0)
exit()
```
output
```
>>> import torch
>>> torch.cuda.current_device()
0
>>> torch.cuda.device(0)
<torch.cuda.device object at 0x7f9f19f80b38>
>>> torch.cuda.get_device_name(0)
'GeForce GTX 960'
>>> exit()
```

```
conda deactivate
conda deactivate
```


Thanks Kyle for such helpful tutorial [1](https://hackernoon.com/up-and-running-with-ubuntu-nvidia-cuda-cudnn-tensorflow-and-pytorch-a54ec2ec907d)




