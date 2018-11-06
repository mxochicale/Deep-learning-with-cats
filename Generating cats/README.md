Runnig scripts
---












# issues


## * [ ] `no images found` DCGAN.py

```
RuntimeError: Found 0 files in subfolders of: /home/robot/datasets/cats_bigger_than_64x64
Supported extensions are: .jpg,.jpeg,.png,.ppm,.bmp,.pgm,.tif
```

instructions:
```
mkdir -p $HOME/datasets/output_cats_bigger_than_64x64
cd $HOME/github/Deep-learning-with-cats/Generating\ cats/
conda activate gancats
python DCGAN.py --input_folder $HOME/datasets/cats_bigger_than_64x64 --output_folder $HOME/datasets/output_cats_bigger_than_64x64
conda deactivate
conda deactivate
```

ADDED: Tue  6 Nov 23:32:44 GMT 2018

