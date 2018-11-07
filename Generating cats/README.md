Runnig scripts
---

# instructions:
```
mkdir -p $HOME/datasets/output_cats_bigger_than_64x64
cd $HOME/github/Deep-learning-with-cats/Generating\ cats/
conda activate gancats
python DCGAN.py --input_folder $HOME/datasets/cats_bigger_than_64x64 --output_folder $HOME/datasets/output_cats_bigger_than_64x64
conda deactivate
conda deactivate
```


# To get TensorBoard output, use the python command: 
```
conda activate gancats
tensorboard --logdir $HOME/datasets/output_cats_bigger_than_64x64
conda deactivate
conda deactivate
```

* open a internet webbrowers, e.g.

```
brave http://machine:6006
```







# terminal output

```
$ python DCGAN.py --input_folder $HOME/datasets/cats_bigger_than_64x64/ --output_folder $HOME/datasets/output_cats_bigger_than_64x64/
Namespace(D_h_size=128, D_load='', G_h_size=128, G_load='', SELU=False, batch_size=64, beta1=0.5, cuda=True, gen_extra_images=0, image_size=64, input_folder='/home/robot/datasets/cats_bigger_than_64x64/', lr_D=5e-05, lr_G=0.0002, n_colors=3, n_epoch=1000, n_gpu=1, n_workers=2, output_folder='/home/robot/datasets/output_cats_bigger_than_64x64/', seed=None, weight_decay=0, z_size=100)
Random Seed: 200
/home/robot/anaconda3/envs/gancats/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/transforms/transforms.py:188: UserWarning: The use of the transforms.Scale transform is deprecated, please use transforms.Resize instead.
DCGAN_G(
  (main): Sequential(
    (Start-ConvTranspose2d): ConvTranspose2d(100, 1024, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (Start-BatchNorm2d): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (Start-ReLU): ReLU()
    (Middle-ConvTranspose2d [1]): ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (Middle-BatchNorm2d [1]): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (Middle-ReLU [1]): ReLU()
    (Middle-ConvTranspose2d [2]): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (Middle-BatchNorm2d [2]): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (Middle-ReLU [2]): ReLU()
    (Middle-ConvTranspose2d [3]): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (Middle-BatchNorm2d [3]): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (Middle-ReLU [3]): ReLU()
    (End-ConvTranspose2d): ConvTranspose2d(128, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (End-Tanh): Tanh()
  )
)
DCGAN_D(
  (main): Sequential(
    (Start-Conv2d): Conv2d(3, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (Start-LeakyReLU): LeakyReLU(negative_slope=0.2, inplace)
    (Middle-Conv2d [0]): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (Middle-BatchNorm2d [0]): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (Middle-LeakyReLU [0]): LeakyReLU(negative_slope=0.2, inplace)
    (Middle-Conv2d [1]): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (Middle-BatchNorm2d [1]): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (Middle-LeakyReLU [1]): LeakyReLU(negative_slope=0.2, inplace)
    (Middle-Conv2d [2]): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (Middle-BatchNorm2d [2]): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (Middle-LeakyReLU [2]): LeakyReLU(negative_slope=0.2, inplace)
    (End-Conv2d): Conv2d(1024, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (End-Sigmoid): Sigmoid()
  )
)
DCGAN.py:338: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  log_value('errD', errD.data[0], current_step)
DCGAN.py:339: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  log_value('errG', errG.data[0], current_step)
DCGAN.py:344: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  s = fmt % (epoch, param.n_epoch, i, len(dataset), errD.data[0], errG.data[0], D_real, D_fake, D_G, end - start)
[0/1000][0/146] Loss_D: 1.7780 Loss_G: 3.9959 D(x): 0.5644 D(G(z)): 0.5435 / 0.0305 time:15.8111
[0/1000][50/146] Loss_D: 3.6139 Loss_G: 13.0106 D(x): 0.4558 D(G(z)): 0.6005 / 0.0000 time:41.6406
[0/1000][100/146] Loss_D: 1.9251 Loss_G: 10.4111 D(x): 0.5641 D(G(z)): 0.2702 / 0.0001 time:69.4411
[1/1000][0/146] Loss_D: 0.4253 Loss_G: 7.7632 D(x): 0.8581 D(G(z)): 0.0736 / 0.0009 time:100.7441
[1/1000][50/146] Loss_D: 1.8167 Loss_G: 2.1257 D(x): 0.4721 D(G(z)): 0.5102 / 0.1432 time:123.2778
[1/1000][100/146] Loss_D: 1.3257 Loss_G: 1.7375 D(x): 0.6117 D(G(z)): 0.4869 / 0.2012 time:145.8626


.
.
.

[49/1000][0/146] Loss_D: 1.3333 Loss_G: 1.1614 D(x): 0.5417 D(G(z)): 0.4853 / 0.3365 time:3307.4411
[49/1000][50/146] Loss_D: 1.2821 Loss_G: 1.0511 D(x): 0.5204 D(G(z)): 0.4439 / 0.3602 time:3330.2704
[49/1000][100/146] Loss_D: 1.4208 Loss_G: 0.9038 D(x): 0.4847 D(G(z)): 0.4838 / 0.4148 time:3353.1273
[50/1000][0/146] Loss_D: 1.3587 Loss_G: 1.0792 D(x): 0.5611 D(G(z)): 0.5181 / 0.3548 time:3374.1982
[50/1000][50/146] Loss_D: 1.2139 Loss_G: 1.0916 D(x): 0.5957 D(G(z)): 0.4770 / 0.3554 time:3397.1825
[50/1000][100/146] Loss_D: 1.4452 Loss_G: 0.9927 D(x): 0.4841 D(G(z)): 0.4799 / 0.4063 time:3419.9759

.
.
.

```

## output path


```
$ tree -h
.
└── [4.0K]  run-0
    ├── [4.0K]  images
    │   ├── [737K]  fake_samples_epoch000.png
    │   ├── [747K]  fake_samples_epoch001.png
    │   ├── [642K]  fake_samples_epoch002.png
    │   ├── [631K]  fake_samples_epoch003.png

.
.
.

    │   ├── [550K]  fake_samples_epoch049.png
    │   ├── [551K]  fake_samples_epoch050.png
    │   └── [543K]  fake_samples_epoch051.png
    ├── [4.0K]  logs
    │   ├── [626K]  events.out.tfevents.1541599710.machine
    │   └── [ 18K]  log.txt
    └── [4.0K]  models
        ├── [ 42M]  D_epoch_0.pth
        ├── [ 42M]  D_epoch_25.pth
        ├── [ 42M]  D_epoch_50.pth
        ├── [ 48M]  G_epoch_0.pth
        ├── [ 48M]  G_epoch_25.pth
        └── [ 48M]  G_epoch_50.pth

4 directories, 60 files
```



