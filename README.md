# just-GAN-it
## Overview
This project generates labeled images datasets by different GANs.
Anyone can add his own dataset and train GAN model.

Following these steps and you'll have your own generated dataset.

## How does it work?
- Step 1:
Insert your dataset images in dataset/train_images with number folder for each label.
For example if you have 3 labels, label 1 will be called 1 and this folder will contain all the images from label 1.
- Step 2:
Choose your relevant arguments:
m = model (string) DCGAN or WGAN
e = epochs (int) not required
gi = generated iterations (int)  not required
bs = batch_size (int)
oi = output_items (int) - The number of items that you want to get from the generator.
nor = number_of_rows (int) - How many rows that you want for the output images.
ims = images_size (int) - How to normalize the input images.
cb = create_batches (string) - Set False if you don't want to create new batches.
t = train (string) - Set False if you just want to generate new images by exist models.


##Running

Running training of DCGAN model 

python main.py --model DCGAN \
               --bs 200 \
               --oi 25 \
               --nor 5 \
               --ims 64 \
               --epochs 30 \
               --t True \
               --cb True \
               --e True
               
               
##Example of training process

#Simpson training:

![](Simpson.gif)

#Anime training:

![](Anime.gif)

#Football Team training:

![](Football Team.gif)