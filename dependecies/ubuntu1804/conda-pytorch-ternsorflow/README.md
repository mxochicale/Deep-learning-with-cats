Deep Learning Frameworks
---

## conda
Check the lastest version of Anaconda at https://repo.anaconda.com/archive/ (See more [ [1] (sep2018) ](https://www.ceos3c.com/open-source/install-anaconda-ubuntu-18-04/), [ [2] ](https://linuxhint.com/install_anaconda_python_ubuntu_1804/)).

```
cd ~/Downloads
wget https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh #04 November 2018 ## Length: 667822837 (637M) 
md5sum Anaconda3*.sh
bash Anaconda3*.sh #ENTER/yes/yes/no
source ~/.bashrc
#rm Anaconda3*.sh #don't delete in case of reinstallation
```

* verify installation 
```
conda info
conda --version
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
# packages in environment at /home/robot/anaconda3/envs/tensorflow:
tensorboard               1.8.0                     <pip>
tensorflow-gpu            1.8.0                     <pip>
```


* testing tensorflow
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
conda install pytorch torchvision cuda90 -c pytorch # pytorch-0.4.1 | 471.7 MB (November 2018)
conda list | grep torch
```
output
```
$ conda list | grep torch
# packages in environment at /home/robot/anaconda3/envs/pytorch:
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




