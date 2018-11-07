issues
---

# todo 



# sorted


* [x] `no images found` DCGAN.py

```
RuntimeError: Found 0 files in subfolders of: /home/robot/datasets/cats_bigger_than_64x64
Supported extensions are: .jpg,.jpeg,.png,.ppm,.bmp,.pgm,.tif
```

SOLUTION 1


If your images are directly in data/ try to move everything into
data/1/ and run again? [ref](https://github.com/pytorch/examples/issues/236#issuecomment-337488192)

```
cd $HOME/datasets/cats_bigger_than_64x64/ 
mkdir images
mv *.jpg images/
```

`It seems like the program tends to recursively read in files, that is convenient in some cases.`
[ref](https://github.com/pytorch/examples/issues/236#issuecomment-432697252)



instructions:
```
mkdir -p $HOME/datasets/output_cats_bigger_than_64x64
cd $HOME/datasets/output_cats_bigger_than_64x64
cd $HOME/github/Deep-learning-with-cats/Generating\ cats/
conda activate gancats
python DCGAN.py --input_folder $HOME/datasets/cats_bigger_than_64x64/ --output_folder $HOME/datasets/output_cats_bigger_than_64x64/
conda deactivate
conda deactivate
```

ADDED: Tue  6 Nov 23:32:44 GMT 2018
SORTED: Wed  7 Nov 14:10:48 GMT 2018



