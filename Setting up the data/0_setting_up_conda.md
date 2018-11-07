Install conda dependencies
---

```
conda create -n gancats python=3.6 pip numpy #remember to initialize the env with pip here.  
conda activate gancats
```

#install opencv-python
```
conda install --channel https://conda.anaconda.org/menpo opencv3 #opencv3-3.1.0  py36_0 37.4 MB  menpo
```

#install tensorflow-gpu
```
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.8.0-cp36-cp36m-linux_x86_64.whl
pip install tensorboard_logger
```

#install pytorch
```
conda install pytorch torchvision cuda90 --channel pytorch # pytorch-0.4.1 | 471.7 MB (November 2018)
```

#install ipython
```
conda install ipython
```


#Cheking 
```
conda list | grep tensor
conda list | grep torch
conda list | grep ipython
```


#deactivate
```
conda deactivate
conda deactivate
```



