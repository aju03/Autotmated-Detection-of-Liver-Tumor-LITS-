'''
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
'''
import nibabel as nib
from glob import glob
import keras.backend as K
import numpy as np
from keras.models import load_model
from skimage import io
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
import os
from skimage.util import img_as_ubyte
import warnings
warnings.filterwarnings('ignore')

img_rows = int(512/2)
img_cols = int(512/2)
smooth=1
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def create_test_data(img_path):
    print('-'*30)
    print('Creating Test data...')
    print('-'*30)
    imgs_test = []
    
    for i in range(len(img_path)):
        img = nib.load(img_path[i])
        for k in range(img.shape[2]):  
            img_2d = np.array(img.get_data()[::2, ::2, k])
            imgs_test.append(img_2d)
                      
    imgst = np.ndarray((len(imgs_test), img_rows, img_cols), dtype=np.uint8)
    for index, img in enumerate(imgs_test):
        imgst[index, :, :] = img
        
    return imgst
def preprocess(img_path,mask_path):
    images=[]
    a=[]
    print('\nLoading Volumes...')
    print('-'*30)
    for i in range(len(img_path)):
      a=nib.load(img_path[i]).get_data()
      print("image--%d--loaded"%(i))
      a=np.resize(a,(a.shape[0],256,256))
      a=a[:,:,:]   	
      for j in range(a.shape[0]):
       	images.append((a[j,:,:]))
        
    images=np.asarray(images)
        
    masks=[]
    b=[]
    print('\nLoading Masks...')
    print('-'*30)
    for i in range(len(mask_path)):
       	b=nib.load(mask_path[i]).get_data()
       	print("mask--%d--loaded"%(i))
       	b=np.resize(b,(b.shape[0],256,256))
       	b=b[:,:,:]
       	for j in range(b.shape[0]):
       		masks.append((b[j,:,:]))
        
    masks=np.asarray(masks)
    
    img = images.reshape(-1,256,256,1)
    mask=masks.reshape(-1,256,256,1)
    
    return img,mask

def preproces(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

#from google.colab import drive
#drive.mount("/content/drive")


print('Loading Test data')
print('-'*30)
#img_path = glob("/content/drive/My Drive/image1/volume-0.nii")
img_path = glob("D:/S8_CS2_MASS/Dataset/Testing/test-volume-0.nii")
imgs_test = create_test_data(img_path)
print('-'*30)
print('Pre-Processing Test data')
print('-'*30)
imgs_test = preproces(imgs_test)
imgs_test = imgs_test.astype('float32')
new_model=load_model("D:/S8_CS2_MASS/Code/project.h5",custom_objects={'dice_coef':dice_coef,'dice_coef_loss':dice_coef_loss})
img_val = glob("D:/S8_CS2_MASS/Dataset/Training Batch1/train/volume-*.nii")
mask_val = glob("D:/S8_CS2_MASS/Dataset/Training Batch1/seg/segmentation-*.nii")
print('-'*30)
print('Validation')
print('-'*30)
val_X,val_ground=preprocess(img_val,mask_val)
scores = new_model.evaluate(val_X, val_ground, verbose=1)
print("Evaluation %s: %.2f%%" % (new_model.metrics_names[1], scores[1]*100))

print('-'*30)
print('Predicting on Test data')
print('-'*30)
imgs_mask_test = new_model.predict(imgs_test, verbose=1)
#np.save('imgs_mask_test.npy', imgs_mask_test)
imgs_mask_test -= imgs_mask_test.min()
imgs_mask_test /= imgs_mask_test.max()
#pre_dir = '/content/drive/My Drive/preds'
pre_dir='D:/S8_CS2_MASS/preds'
for k in range(len(imgs_mask_test)):
    a=rescale_intensity(imgs_test[k][:,:,0],out_range=(-1,1))
    b=(imgs_mask_test[k][:,:,0]).astype(np.uint8)
    io.imsave(os.path.join(pre_dir, str(k)+'_pre.png'),mark_boundaries(a,b))
    
print('Saved!')


from PIL import Image
from glob import glob 

frames = []
imgs = glob("D:/S8_CS2_MASS/preds/*.png")
for i in imgs:
    new_frame = Image.open(i).transpose(Image.ROTATE_90).rotate(180)
    frames.append(new_frame)

frames[0].save('D:/S8_CS2_MASS/png_to_gif.gif', format='GIF',append_images=frames[1:],save_all=True,duration=500, loop=0)

