import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os

#Number of Emotion Classes
num_classes = 7
#Each image dimention is 48*48 in fer2013 dataset
img_rows, img_cols = 48, 48
#Number of images we want to give in a batch to CNN for training
batch_size = 16

#Training dataset path
train_data_dir = r'C:\Users\amit1\Desktop\Emotion Recognition System\train'
#Validation dataset path
validation_data_dir = r'C:\Users\amit1\Desktop\Emotion Recognition System\validation'

#Using ImageDataGenerator() function we generate 7-8 modified frames of each image for model to detect and train on specific features
train_datagen = ImageDataGenerator(
					rescale=1./255, #Divide each pixel in image by 255 to make its value between the range (0.0 to 1.0) to scale down the value
					rotation_range=30,
					shear_range=0.3,
					zoom_range=0.3,
					width_shift_range=0.4,
					height_shift_range=0.4,
					horizontal_flip=True,
					fill_mode='nearest')

#For Validating no need to rotate, shear, zoom or flip the image. Just normalize it by dividing each piecl by 255
validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
					train_data_dir, #Training dataset path
					color_mode='grayscale',
					target_size=(img_rows,img_cols), #Image dimention 48*48
					batch_size=batch_size, #Number of images we want to give in a batch to CNN for training i.e 16
					class_mode='categorical', #Various emotion categories i.e. ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
					shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir, #Validation dataset path
							color_mode='grayscale',
							target_size=(img_rows,img_cols), #Image dimention 48*48
							batch_size=batch_size, #Number of images we want to give in a batch to CNN for training i.e 16
							class_mode='categorical', #Various emotion categories i.e. ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
							shuffle=True)


#From here on we will construct the CNN by setting , adding the input layer, hidden layers and output layers in model

model = Sequential()


#-------------------------------------------------------------------------------------------------------------------------------------------

# Block-1

#Activation Function : ReLU, GeLU, ELU work the same way but with just a little difference while handling negative values


model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-2 

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-3

model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-4 

model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#-------------------------------------------------------------------------------------------------------------------------------------------

# Block-5

model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-6

model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-7

model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax')) 

print(model.summary())

#Softmax Activation Function : Use when you have to classify between more than 2 sample values
#Sigmoid Activation Function : Use when you have to classify between 2 sample values
#Note : For binary classification of sample values both activation function work the same way


from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

'''
verbose = 0, means silent
verbose = 1, which includes both progress bar and one line per epoch
verbose = 2, one line per epoch i.e. epoch no./total no. of epochs
'''

#In checkpoint I've saved the model iteration with the best accuracy.
checkpoint = ModelCheckpoint(r'C:\Users\amit1\Desktop\Emotion Recognition System\emotion_recognition_vgg_cnn.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

#If validation loss i.e. val_loss is not improving for 3 iterations i.e. patience=3 then stop training model early
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

#If validation loss i.e. val_loss is not improving for 3 iterations i.e. patience=3 then reduce the learning rate by 0.0001
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])

nb_train_samples = 28821        #24176
nb_validation_samples = 7066    #3006
epochs=25

history=model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples//batch_size)




'''
CNN Design :

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 48, 48, 32)        320
_________________________________________________________________
activation (Activation)      (None, 48, 48, 32)        0
_________________________________________________________________
batch_normalization (BatchNo (None, 48, 48, 32)        128
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 48, 48, 32)        9248
_________________________________________________________________
activation_1 (Activation)    (None, 48, 48, 32)        0
_________________________________________________________________
batch_normalization_1 (Batch (None, 48, 48, 32)        128
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 24, 24, 32)        0
_________________________________________________________________
dropout (Dropout)            (None, 24, 24, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 24, 24, 64)        18496
_________________________________________________________________
activation_2 (Activation)    (None, 24, 24, 64)        0
_________________________________________________________________
batch_normalization_2 (Batch (None, 24, 24, 64)        256
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 24, 24, 64)        36928
_________________________________________________________________
activation_3 (Activation)    (None, 24, 24, 64)        0
_________________________________________________________________
batch_normalization_3 (Batch (None, 24, 24, 64)        256
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 12, 64)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 12, 12, 64)        0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 12, 12, 128)       73856
_________________________________________________________________
activation_4 (Activation)    (None, 12, 12, 128)       0
_________________________________________________________________
batch_normalization_4 (Batch (None, 12, 12, 128)       512
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 12, 12, 128)       147584
_________________________________________________________________
activation_5 (Activation)    (None, 12, 12, 128)       0
_________________________________________________________________
batch_normalization_5 (Batch (None, 12, 12, 128)       512
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 6, 6, 128)         0
_________________________________________________________________
dropout_2 (Dropout)          (None, 6, 6, 128)         0
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 6, 6, 256)         295168
_________________________________________________________________
activation_6 (Activation)    (None, 6, 6, 256)         0
_________________________________________________________________
batch_normalization_6 (Batch (None, 6, 6, 256)         1024
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 6, 6, 256)         590080
_________________________________________________________________
activation_7 (Activation)    (None, 6, 6, 256)         0
_________________________________________________________________
batch_normalization_7 (Batch (None, 6, 6, 256)         1024
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 3, 3, 256)         0
_________________________________________________________________
dropout_3 (Dropout)          (None, 3, 3, 256)         0
_________________________________________________________________
flatten (Flatten)            (None, 2304)              0
_________________________________________________________________
dense (Dense)                (None, 64)                147520
_________________________________________________________________
activation_8 (Activation)    (None, 64)                0
_________________________________________________________________
batch_normalization_8 (Batch (None, 64)                256
_________________________________________________________________
dropout_4 (Dropout)          (None, 64)                0
_________________________________________________________________
dense_1 (Dense)              (None, 64)                4160
_________________________________________________________________
activation_9 (Activation)    (None, 64)                0
_________________________________________________________________
batch_normalization_9 (Batch (None, 64)                256
_________________________________________________________________
dropout_5 (Dropout)          (None, 64)                0
_________________________________________________________________
dense_2 (Dense)              (None, 7)                 455
_________________________________________________________________
activation_10 (Activation)   (None, 7)                 0
=================================================================
Total params: 1,328,167
Trainable params: 1,325,991
Non-trainable params: 2,176
_________________________________________________________________
None

'''

r'''
C:\Users\amit1\AppData\Local\Programs\Python\Python38\lib\site-packages\tensorflow\python\keras\engine\training.py:1844: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.
  warnings.warn('`Model.fit_generator` is deprecated and '
2021-03-18 10:27:58.138018: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
Epoch 1/25
1801/1801 [==============================] - 319s 176ms/step - loss: 2.3143 - accuracy: 0.1871 - val_loss: 1.7781 - val_accuracy: 0.2589

Epoch 00001: val_loss improved from inf to 1.77812, saving model to C:\Users\amit1\Desktop\Emotion Recognition System\emotion_recognition_vgg_cnn.h5
Epoch 2/25
1801/1801 [==============================] - 315s 175ms/step - loss: 1.8082 - accuracy: 0.2409 - val_loss: 1.7769 - val_accuracy: 0.2541

Epoch 00002: val_loss improved from 1.77812 to 1.77689, saving model to C:\Users\amit1\Desktop\Emotion Recognition System\emotion_recognition_vgg_cnn.h5
Epoch 3/25
1801/1801 [==============================] - 319s 177ms/step - loss: 1.7872 - accuracy: 0.2542 - val_loss: 1.9918 - val_accuracy: 0.2358

Epoch 00003: val_loss did not improve from 1.77689
Epoch 4/25
1801/1801 [==============================] - 316s 176ms/step - loss: 1.7570 - accuracy: 0.2745 - val_loss: 1.5570 - val_accuracy: 0.3882

Epoch 00004: val_loss improved from 1.77689 to 1.55703, saving model to C:\Users\amit1\Desktop\Emotion Recognition System\emotion_recognition_vgg_cnn.h5
Epoch 5/25
1801/1801 [==============================] - 317s 176ms/step - loss: 1.6801 - accuracy: 0.3217 - val_loss: 1.4803 - val_accuracy: 0.4392

Epoch 00005: val_loss improved from 1.55703 to 1.48027, saving model to C:\Users\amit1\Desktop\Emotion Recognition System\emotion_recognition_vgg_cnn.h5
Epoch 6/25
1801/1801 [==============================] - 317s 176ms/step - loss: 1.5865 - accuracy: 0.3784 - val_loss: 1.3042 - val_accuracy: 0.5051

Epoch 00006: val_loss improved from 1.48027 to 1.30423, saving model to C:\Users\amit1\Desktop\Emotion Recognition System\emotion_recognition_vgg_cnn.h5
Epoch 7/25
1801/1801 [==============================] - 319s 177ms/step - loss: 1.5416 - accuracy: 0.3982 - val_loss: 1.2404 - val_accuracy: 0.5278

Epoch 00007: val_loss improved from 1.30423 to 1.24038, saving model to C:\Users\amit1\Desktop\Emotion Recognition System\emotion_recognition_vgg_cnn.h5
Epoch 8/25
1801/1801 [==============================] - 318s 177ms/step - loss: 1.4884 - accuracy: 0.4275 - val_loss: 1.2924 - val_accuracy: 0.5082

Epoch 00008: val_loss did not improve from 1.24038
Epoch 9/25
1801/1801 [==============================] - 321s 178ms/step - loss: 1.4685 - accuracy: 0.4364 - val_loss: 1.2569 - val_accuracy: 0.5286

Epoch 00009: val_loss did not improve from 1.24038
Epoch 10/25
1801/1801 [==============================] - 320s 178ms/step - loss: 1.4504 - accuracy: 0.4450 - val_loss: 1.2372 - val_accuracy: 0.5252

Epoch 00010: val_loss improved from 1.24038 to 1.23719, saving model to C:\Users\amit1\Desktop\Emotion Recognition System\emotion_recognition_vgg_cnn.h5
Epoch 11/25
1801/1801 [==============================] - 318s 176ms/step - loss: 1.4169 - accuracy: 0.4577 - val_loss: 1.1676 - val_accuracy: 0.5541

Epoch 00011: val_loss improved from 1.23719 to 1.16763, saving model to C:\Users\amit1\Desktop\Emotion Recognition System\emotion_recognition_vgg_cnn.h5
Epoch 12/25
1801/1801 [==============================] - 315s 175ms/step - loss: 1.4046 - accuracy: 0.4684 - val_loss: 1.1704 - val_accuracy: 0.5489

Epoch 00012: val_loss did not improve from 1.16763
Epoch 13/25
1801/1801 [==============================] - 316s 176ms/step - loss: 1.4011 - accuracy: 0.4684 - val_loss: 1.1526 - val_accuracy: 0.5580

Epoch 00013: val_loss improved from 1.16763 to 1.15260, saving model to C:\Users\amit1\Desktop\Emotion Recognition System\emotion_recognition_vgg_cnn.h5
Epoch 14/25
1801/1801 [==============================] - 316s 175ms/step - loss: 1.3797 - accuracy: 0.4726 - val_loss: 1.1320 - val_accuracy: 0.5677

Epoch 00014: val_loss improved from 1.15260 to 1.13204, saving model to C:\Users\amit1\Desktop\Emotion Recognition System\emotion_recognition_vgg_cnn.h5
Epoch 15/25
1801/1801 [==============================] - 318s 177ms/step - loss: 1.3553 - accuracy: 0.4837 - val_loss: 1.1357 - val_accuracy: 0.5690

Epoch 00015: val_loss did not improve from 1.13204
Epoch 16/25
1801/1801 [==============================] - 316s 175ms/step - loss: 1.3556 - accuracy: 0.4871 - val_loss: 1.1625 - val_accuracy: 0.5554

Epoch 00016: val_loss did not improve from 1.13204
Epoch 17/25
1801/1801 [==============================] - 316s 176ms/step - loss: 1.3464 - accuracy: 0.4942 - val_loss: 1.1891 - val_accuracy: 0.5472
Restoring model weights from the end of the best epoch.

Epoch 00017: val_loss did not improve from 1.13204

Epoch 00017: ReduceLROnPlateau reducing learning rate to 0.00020000000949949026.
Epoch 00017: early stopping
2021-03-18 11:57:57.698431: W tensorflow/core/kernels/data/generator_dataset_op.cc:107] Error occurred when finalizing GeneratorDataset iterator: Failed precondition: Python interpreter state is not initialized. The process may be terminated.
         [[{{node PyFunc}}]]
'''
