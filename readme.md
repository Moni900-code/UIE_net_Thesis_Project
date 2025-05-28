# EnahnceNet: LiteEnhanceNet: A Lightweight Network for Real-time Single Underwater Image Enhancement 


### image restoration with PyTorch:
python friendly open-source deep learning framework for research purpose. 

### Dataset: EUVP: 1. paired: dark, imagenet, scenes..(MSE matric) 2. unpaired (real time underwater image enhancement:- UIQM, SSIM, PSNR)

## compare among matrics: ssim, psnr, uiqm,
### ssim (structure similarity index measure): 
To check the structure similarity( texture,luminance etc) between ground truth and enhanced image.. 0-1 range. need increase value.

# value: 0.84

### psnr (point signal to noise ratio): 
To check the noise level of image. increase value is good. 40+ is high quality.

# value: 27.43

### uiqm (underwater image quality measure): 
To check overall visual quality improve. important matric for underwater image enhancement

# value: 2.90

### mse (mean squred error): 
To check difference between GT and output image

# value: 0.15



1. Data folder:  output: enhanced image 

2. snapshots: weight save checkpoint of the 

3. utils folder: utility function/ helper code:

   1. data_utils.py: training and validation dataset augmnetation and transformation/normalization

   2. imqual_utils.py: image quality check code----- psnr and ssim matrics calculation process

   3. plot_utils.py: enhance image generation, loss value plot fo generator and discriminator, 

   4. ssm_psnr_utils.py: same as imqual_utils.py

   5. uqim_utils.py: UICM (Underwater imae colorfulness measure), UISM (Underwater Image Sharpness Measure), UIConM (Underwater Image Contrast Measure)

4. wandb folder: Help to model training, validation loss, accuracy visualizaion using weight and biasess library 

5. combined_loss.py: add all loss for total loss

6. dataloader.py: For dataset load and tranfer data for training and testing.

7. metrics_calculation.py: After training, calculate all metrics for generated enhanced image.

8. model.py: model architecture

9. ssim_loss.py: Calculate ssim loss during trianing the model 

10. vgg_loss.py: to calculate loss based on human perception-based similarity. better quality = small value of vgg loss. we use vgg19 pretrain model.

11. test.py:
12. training.py:

13. uiqm_utils.py: same as upper uqim_utils.py file with window size=8 and First handles division-by-zero edge  case.

# note: window_size = 10, a 100x100 image will be divided into 10x10 = 100 blocks. Each block is analyzed for quality (calculates contrast, sharpness, or enhancement).



## Train the Model
python training.py

## Test the Model
python test.py

## Environment
python=3.8
pytorch=1.11
cudnn=8.2
numpy=1.22






# Additional note:

Tensor: multi-dimentional array like NumPy array.. For Deep learning purpose all images are converted into tensor array for easy numeric calculation.

Pixel loss (MSE or L1): claculate prixel by pixel difference of images. 
VGG based perceptual loss: to check high level feature ( texture, edges, patterns)

ssim loss:  small value.... calculate for enhanced image

luminance:  value of brightness (calculate brightness of pixel by pixel)
contrast: difference of brightness (calculate brightness of group of pixel)