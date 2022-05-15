# %%
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, LeakyReLU,BatchNormalization, UpSampling2D, ReLU
import numpy as np

# Definitions of our architecture and blocks
# %%
def bottle_block(x,units=64,kernel=3):
    out = conv_block(x,units,kernel,1)
    out = Conv2D(units,kernel,1,padding="same")(out)
    out = BatchNormalization()(out)
    out = out + x
    out = ReLU()(out)
    return out

def conv_block(x,units,kernel=3,stride=1,activation=ReLU):
    out = Conv2D(units,kernel,stride,padding="same")(x)
    out = BatchNormalization()(out)
    out = activation()(out)
    return out

def up_block(x,units=64,kernel=3):
    out = UpSampling2D(2,interpolation="bilinear")(x)
    out = conv_block(out,units,kernel,1)
    return out

def down_block(x,units=64,kernel=3):
    out = conv_block(x,units,kernel,2)
    out = conv_block(out,units,kernel,1)
    return out

# %%
def generator(input_shape,units=64,layers=1,num_blocks=6,out_channels=6):
    x = keras.layers.Input(input_shape)
    g2 = conv_block(x,units,7,1)
    temp = units
    skip = []
    for i in range(0,layers):
        g2 = down_block(g2,temp*2,3)
        skip.append(g2)
        temp *=2
    
    #code for local enhancer network
    #not sure if needed to access local network output for training
    local = None
    for i in range(0,layers):
        g2 = skip[-1-i]
        local = g2
        for j in range(0,num_blocks):
            local = bottle_block(local,temp)
        local = g2 + local
        temp = temp//2
        local = up_block(local,temp)
    
    g2 = local
    for i in range(0,num_blocks):
        g2 = bottle_block(g2,units)
    g2 = conv_block(g2,out_channels,7)
    g2 = tf.keras.activations.tanh(g2)
    return keras.Model(x,g2)

# %%
def unet(input_shape,n_units,levels,classes):
    inputs = keras.layers.Input(shape=(input_shape))
    output = inputs
    units = n_units
    #create array to store skip tensors
    level = []
    #Create contracting path
    for i in range(0,levels):
        output = keras.layers.Conv2D(units,3,strides=1,padding='same',activation='relu')(output)
        output = keras.layers.Conv2D(units,3,strides=1,padding='same',activation='relu')(output)
        #output = keras.layers.Dropout(0.1)(output)
        level.append(output)
        output = keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="valid")(output)
        units = units * 2
        
    #Create expansion path    
    for i in range(0,levels):
        output = keras.layers.Conv2D(units,3,strides=1,padding='same',activation='relu')(output)
        output = keras.layers.Conv2D(units,3,strides=1,padding='same',activation='relu')(output)
        output = keras.layers.UpSampling2D(size =(2,2),interpolation="bilinear")(output)
        units = units//2
        output = keras.layers.Conv2D(units,2,padding='same')(output)
        output = tf.concat([level[-1-i],output],axis=3)
    
    #outputting segmentation map
    output = keras.layers.Conv2D(units,3,strides=1,padding='same',activation='relu')(output)
    output = keras.layers.Conv2D(units,3,strides=1,padding='same',activation='relu')(output)
    output = keras.layers.Conv2D(classes,1,padding='same',activation='sigmoid')(output)
    model = keras.Model(inputs,output)
    return model

# %%
def discriminator(x,n_units=64,layers=3):
    temp = n_units
    out = Conv2D(temp,4,2,padding="same")(x)
    out = LeakyReLU()(out)
    for i in range(1,layers):
        temp = n_units*2
        out = conv_block(out,temp,4,2,activation=LeakyReLU)
    temp = n_units*2
    out = conv_block(out,temp,4,1,activation=LeakyReLU)
    out = Conv2D(1,4,1,padding="same")(out)
    out = keras.layers.Flatten()(out)
    out = keras.activations.sigmoid(out)
    return keras.Model(x,out)

# %%
def multiscale_discriminator(input_shape):
    inputs = keras.layers.Input(input_shape)
    out = None
    return keras.Model(inputs,out)
# VGG19 used for perceptual loss
# %%
vgg = keras.applications.vgg19.VGG19(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'
)
# Weights for training
pixel_weight = 0.1
perceptual_weight = 1
adversarial_weight = 0.001
segmentation_weight = 0.1
reconstruction_weight = 1
stepwise_weight = 1

# SOme of the loss functions we defined
# %%
# Using the VGG19 pretrained net, get the facial feature map and compare the two images' facial features
def perceptual_loss(y_true, y_pred):
    x = tf.image.resize(y_true,(224,224))
    y = tf.image.resize(y_pred,(224,224))
    x_vgg = vgg.predict(x)
    y_vgg = vgg.predict(y)
    loss = 0
    loss +=  keras.losses.MeanAbsoluteError(x_vgg,y_vgg)
    return loss
def pixel_loss(x, y):
    return keras.losses.MeanAbsoluteError(x,y)
    
def reconstruction_loss(x, y):
    loss = 1*perceptual_loss(x,y) + 0.1*pixel_loss(x,y)
    return loss
def adversarial_loss(y_true, y_pred):
    pass
def poisson_blending_loss(y_true, y_pred):
    return reconstruction_loss()
# Getting the test images into a dataset
# %%
dataset = tf.keras.utils.image_dataset_from_directory(
    'dataset/CelebA-HQ-img/',
    labels=None,
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=10000,
    image_size=(128, 128),
    shuffle=False,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False
)
# Getting the masks
# %%
mask = np.zeros((10000,128,128,2))
for i in range(0,10000):
    s = f'{i:05}'
    try:
        mask1=tf.keras.utils.load_img('./dataset/CelebAMask-HQ-mask-anno/hair/'+ s +"_hair.png", color_mode= "grayscale", target_size=(128,128), interpolation="nearest")
        mask1= tf.keras.preprocessing.image.img_to_array(mask1)

    except:
        mask1=tf.zeros((128,128,1))
    mask2=tf.keras.utils.load_img('./dataset/CelebAMask-HQ-mask-anno/skin/'+ s +"_skin.png", color_mode= "grayscale", target_size=(128,128), interpolation="nearest")
    mask2= tf.keras.preprocessing.image.img_to_array(mask2)
    mask[i] = tf.concat([mask1,mask2], -1)
mask = mask/255


