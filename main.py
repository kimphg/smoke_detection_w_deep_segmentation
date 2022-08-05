from sklearn.model_selection import train_test_split
import os
import glob
import cv2
import numpy as np # linear algebra
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.util import montage
montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
data_dir = '/content/drive/MyDrive/Smoke-detecttion-image-processing/dataset'
train_image_dir = os.path.join(data_dir, 'D:/dataset/smoke/data_unet/train')
test_image_dir = os.path.join(data_dir, 'D:/dataset/smoke/data_unet/test')
import gc; gc.enable() # memory is tight
from skimage.morphology import label

BATCH_SIZE = 4
EDGE_CROP = 2
NB_EPOCHS = 100
GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'DECONV'
# downsampling inside the network
NET_SCALING = None
# downsampling in preprocessing
IMG_SCALING = (1, 1)
# number of validation images to use
VALID_IMG_COUNT = 1600
# maximum number of steps_per_epoch in training
MAX_TRAIN_STEPS = 200
AUGMENT_BRIGHTNESS = False

def get_all_imgs():
    img_path = os.path.join(train_image_dir,'images')
    images = glob.glob(os.path.join(img_path,'*.*'))
    return [os.path.basename(image) for image in images]

TRAIN_IMGS, TEST_IMGS = train_test_split(get_all_imgs(),test_size=0.2,random_state=20)

print(len(TRAIN_IMGS), len(TEST_IMGS))

import random
def cv2_brightness_augment(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    v = hsv[:,:,2]
    seed = random.uniform(0.5,1.2)
    v = (( v/255.0 ) * seed)*255.0
    hsv[:,:,2] = np.array(np.clip(v,0,255),dtype=np.uint8)
    rgb_final = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return rgb_final

def make_image_gen(img_file_list=TRAIN_IMGS, batch_size = BATCH_SIZE):
    all_batches = img_file_list
    out_rgb = []
    out_mask = []
    img_path = os.path.join(train_image_dir,'images')
    mask_path = os.path.join(train_image_dir,'masks')
    while True:
        np.random.shuffle(all_batches)
        for c_img_id in all_batches:
            c_img = imread(os.path.join(img_path,c_img_id))
            c_img = cv2_brightness_augment(c_img)
            c_mask = imread(os.path.join(mask_path,c_img_id))
            if IMG_SCALING is not None:
                c_img = cv2.resize(c_img,(256,256),interpolation = cv2.INTER_AREA)
                c_mask = cv2.resize(c_mask,(256,256),interpolation = cv2.INTER_AREA)
            c_mask = np.reshape(c_mask,(c_mask.shape[0],c_mask.shape[1],-1))
            c_mask = c_mask > 0
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []

train_gen = make_image_gen()
train_x, train_y = next(train_gen)
valid_x, valid_y = next(make_image_gen(TEST_IMGS,len(TEST_IMGS)))
from keras.preprocessing.image import ImageDataGenerator
dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
                  rotation_range = 15, 
                  width_shift_range = 0.1, 
                  height_shift_range = 0.1, 
                  shear_range = 0.01,
                  zoom_range = [0.9, 1.25],  
                  horizontal_flip = True, 
                  vertical_flip = False,
                  fill_mode = 'reflect',
                   data_format = 'channels_last')
# brightness can be problematic since it seems to change the labels differently from the images 
if AUGMENT_BRIGHTNESS:
    dg_args[' brightness_range'] = [0.5, 1.5]
image_gen = ImageDataGenerator(**dg_args)

if AUGMENT_BRIGHTNESS:
    dg_args.pop('brightness_range')
label_gen = ImageDataGenerator(**dg_args)

def create_aug_gen(in_gen, seed = None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(255*in_x, 
                             batch_size = in_x.shape[0], 
                             seed = seed, 
                             shuffle=True)
        g_y = label_gen.flow(in_y, 
                             batch_size = in_x.shape[0], 
                             seed = seed, 
                             shuffle=True)

        yield next(g_x)/255.0, next(g_y)

cur_gen = create_aug_gen(train_gen)
t_x, t_y = next(cur_gen)
print('x', t_x.shape, t_x.dtype, t_x.min(), t_x.max())
print('y', t_y.shape, t_y.dtype, t_y.min(), t_y.max())
# only keep first 9 samples to examine in detail
t_x = t_x[:9]
t_y = t_y[:9]

gc.collect()

from keras import models, layers

def upsample_conv(filters, kernel_size, strides, padding):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)
def upsample_simple(filters, kernel_size, strides, padding):
    return layers.UpSampling2D(strides)

if UPSAMPLE_MODE=='DECONV':
    upsample=upsample_conv
else:
    upsample=upsample_simple
    
input_img = layers.Input(t_x.shape[1:], name = 'RGB_Input')
pp_in_layer = input_img
if NET_SCALING is not None:
    pp_in_layer = layers.AvgPool2D(NET_SCALING)(pp_in_layer)
    
pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer)
pp_in_layer = layers.BatchNormalization()(pp_in_layer)

c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (pp_in_layer)
c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c1)
p1 = layers.MaxPooling2D((2, 2)) (c1)

c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (p1)
c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c2)
p2 = layers.MaxPooling2D((2, 2)) (c2)

c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (p2)
c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c3)
p3 = layers.MaxPooling2D((2, 2)) (c3)

c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (p3)
c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (c4)
p4 = layers.MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same') (p4)
c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same') (c5)
p5 = layers.MaxPooling2D(pool_size=(2, 2)) (c5)

c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same') (p5)
c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same') (c6)
p6 = layers.MaxPooling2D(pool_size=(2, 2)) (c6)

c7 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same') (p6)
c7 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same') (c7)
p7 = layers.MaxPooling2D(pool_size=(2, 2)) (c7)

c8 = layers.Conv2D(2048, (3, 3), activation='relu', padding='same') (p7)
c8 = layers.Conv2D(2048, (3, 3), activation='relu', padding='same') (c8)

u9 = upsample(1024, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = layers.concatenate([u9, c7])
c9 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same') (u9)
c9 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same') (c9)

u10 = upsample(512, (2, 2), strides=(2, 2), padding='same') (c9)
u10 = layers.concatenate([u10, c6])
c10 = layers.Conv2D(512, (3, 3), activation='relu', padding='same') (u10)
c10 = layers.Conv2D(512, (3, 3), activation='relu', padding='same') (c10)

u11 = upsample(256, (2, 2), strides=(2, 2), padding='same') (c10)
u11 = layers.concatenate([u11, c5])
c11 = layers.Conv2D(256, (3, 3), activation='relu', padding='same') (u11)
c11 = layers.Conv2D(256, (3, 3), activation='relu', padding='same') (c11)

u12 = upsample(128, (2, 2), strides=(2, 2), padding='same') (c11)
u12 = layers.concatenate([u12, c4], axis=3)
c12 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (u12)
c12 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (c12)

u13 = upsample(64, (2, 2), strides=(2, 2), padding='same') (c12)
u13 = layers.concatenate([u13, c3], axis=3)
c13 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (u13)
c13 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c13)

u14 = upsample(32, (2, 2), strides=(2, 2), padding='same') (c13)
u14 = layers.concatenate([u14, c2], axis=3)
c14 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (u14)
c14 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c14)

u15 = upsample(16, (2, 2), strides=(2, 2), padding='same') (c14)
u15 = layers.concatenate([u15, c1], axis=3)
c15 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (u15)
c15 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c15)

d = layers.Conv2D(1, (1, 1), activation='sigmoid') (c15)
d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)
if NET_SCALING is not None:
    d = layers.UpSampling2D(NET_SCALING)(d)

seg_model = models.Model(inputs=[input_img], outputs=[d])
seg_model.summary()


import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
def dice_coef(y_true, y_pred, smooth=1):
    print(y_true,y_pred)
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
def dice_p_bce(in_gt, in_pred):
    return 1e-3*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)
def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)
seg_model.compile(optimizer=Adam(1e-4, decay=1e-6), loss=dice_p_bce, metrics=[dice_coef, 'binary_accuracy', true_positive_rate])

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('seg_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1, 
                             save_best_only=True, mode='max', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, 
                                   patience=3, 
                                   verbose=1, mode='max', epsilon=0.0001, cooldown=2, min_lr=1e-6)
early = EarlyStopping(monitor="val_dice_coef", 
                      mode="max", 
                      patience=15) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]

step_count = min(MAX_TRAIN_STEPS, len(TRAIN_IMGS)//BATCH_SIZE)
aug_gen = create_aug_gen(make_image_gen())
val_gen = make_image_gen(TEST_IMGS, len(TEST_IMGS)//BATCH_SIZE)
loss_history = [seg_model.fit_generator(aug_gen, 
                             steps_per_epoch=step_count, 
                             epochs=NB_EPOCHS, 
                             validation_data=val_gen,
                             validation_steps=len(TEST_IMGS)//BATCH_SIZE,
                             callbacks=callbacks_list,
                            workers=1 # the generator is not very thread safe
                                       )]

