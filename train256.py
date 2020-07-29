from keras.optimizers import Adam
import tensorflow as tf
import nibabel as nib
from glob import glob
import keras.backend as K
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint,EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from model import u_net
#from keras.callbacks import History
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
import warnings
warnings.filterwarnings('ignore')

smooth=1
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

input_shape = [256,256,1]
dropout_rate = 0.3
l2_lambda = 0.0002

model = u_net(input_shape, dropout_rate, l2_lambda)
model.summary()

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

img_path = glob("D:/S8_CS2_MASS/Dataset/Training Batch1/Image0-29/volume-*.nii")
mask_path = glob("D:/S8_CS2_MASS/Dataset/Training Batch1/Mask0-29/segmentation-*.nii")

img_ex = glob("D:/S8_CS2_MASS/Dataset/Training Batch1/img30-59/volume-*.nii")
mask_ex = glob("D:/S8_CS2_MASS/Dataset/Training Batch1/mask30-59/segmentation-*.nii")

img_val = glob("D:/S8_CS2_MASS/Dataset/Training Batch1/train/volume-*.nii")
mask_val = glob("D:/S8_CS2_MASS/Dataset/Training Batch1/seg/segmentation-*.nii")

adam = Adam(lr = 0.0001)
model.compile(optimizer = adam, loss =dice_coef_loss, metrics = [dice_coef])
EarlyStop = EarlyStopping(monitor='val_dice_coef', patience=10, verbose=1, mode='min')
checkpoint = ModelCheckpoint('project.h5', monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max')
callbacks_list = [EarlyStop,checkpoint]

print('-'*30)
print('Training Batch 1')
print('-'*30)
data,mask=preprocess(img_path,mask_path)
train_X,valid_X,train_ground,valid_ground = train_test_split(data,mask,test_size=0.2,random_state=13)
history= model.fit(train_X,train_ground, batch_size=8,epochs=25,verbose=1,validation_data=(valid_X,valid_ground),validation_split=0.2,callbacks=callbacks_list)
model.save('project.h5')
model.save_weights('new_weights.hdf5')
print("Model Saved.")

print('-'*30)
print('Training Batch 2')
print('-'*30)
vol,seg=preprocess(img_ex,mask_ex)
tra_2,valid_2,tra_gr2,valid_gr2 = train_test_split(vol,seg,test_size=0.2,random_state=13)
history=model.fit(tra_2,tra_gr2, batch_size=8,epochs=25,verbose=1,validation_data=(valid_2,valid_gr2),validation_split=0.2,callbacks=callbacks_list)
model.save('project.h5',overwrite=True)
model.save_weights('new_weights.hdf5',overwrite=True)
print("Model Saved.")

print('-'*30)
print('Validation')
print('-'*30)
val_X,val_ground=preprocess(img_val,mask_val)
scores = model.evaluate(val_X, val_ground, verbose=1)
print("Evaluation %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('2D Unet Model')
plt.ylabel('Dice coeff')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

