## Usage for b_setting_up_script.sh

# conda activate gancats
# sh b_setting_up_script.sh
# conda deactivate


## Go to datasets path
cd
mkdir -p datasets
cd datasets


## Preprocessing and putting in folders for different image sizes
mkdir cats_bigger_than_64x64
mkdir cats_bigger_than_128x128
wget -nc https://raw.githubusercontent.com/mxochicale/Deep-learning-with-cats/master/Setting%20up%20the%20data/preprocess_cat_dataset.py
python preprocess_cat_dataset.py
rm preprocess_cat_dataset.py


## Removing cat_dataset
rm -r cat_dataset



## Move to an images path
cd $HOME/datasets/cats_bigger_than_64x64/ 
mkdir images
mv *.jpg images/

cd $HOME/datasets/cats_bigger_than_128x128
mkdir images
mv *.jpg images/


## Move to your favorite place
#mv cats_bigger_than_64x64 /home/alexia/Datasets/Meow_64x64
#mv cats_bigger_than_128x128 /home/alexia/Datasets/Meow_128x128

