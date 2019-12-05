# %%
import time
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab
from tqdm import tqdm
import shutil  # for folder removal.
import keras
import winsound
from keras_segmentation.models.unet import vgg_unet
%matplotlib inline

# %%
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# %% Functions
def notification():
    winsound.Beep(1000,1000)

def save_model(model,model_folder_savings):
  """
  Saves the model given as parameter.
  """
  os.chdir(model_folder_savings)
  date = datetime.datetime.today()
  time_stamp = date.strftime("day %d month %m year %y at %H %M %S")
  model.save('model {}.h5'.format(time_stamp))

def train(model, epoch, call_id,model_folder_savings, callbacks):
  """
  Trains the given model from the current epoch (left because of an issue)
  to the end.
  """
  try:
    history = model.train(
        train_images =  images_folder_train,
        train_annotations = annotations_images_folder_train,
        val_images =  images_folder_val,
        val_annotations = annotations_images_folder_val,
        epochs=epoch,
        batch_size=22,
        val_batch_size=22,
        steps_per_epoch= 5300,
        verify_dataset=False,
        n_classes=134,
        load_weights=None,
        optimizer_name='adadelta',
        callbacks=[tensorboard_callback],
        validate=True,
    )
    # notification for end of epoch.
    notification()
    # saves the model
    save_model(model,ep,model_folder_savings)
  except Exception as e:
    notification()
    time.sleep(1)
    notification()
    save_model(model,ep,model_folder_savings)
    print("Error in one training session : ")
    print(e)


# %% Folders creation of the training
# images
images_folder_val = "D:\\Dataset\\coco2017\\val2017\\val2017\\"
images_folder_train = "D:\\Dataset\\coco2017\\train2017\\train2017\\"
# annotations
annotations_images_folder_val = "D:\\Dataset\\new_seg\\semantic_validation\\semantic_validation\\"
annotations_images_folder_train = "D:\\Dataset\\new_seg\\semantic_train\\semantic_train\\"
# model folder for storage on the hard drive
model_folder_savings = "D:\\Dataset\\models\\"
# creation of a folder for the training.
os.chdir(model_folder_savings)
date = datetime.datetime.today()
time_stamp = date.strftime("day %d month %m year %y at %H %M %S")
model_folder_name="model {}\\".format(time_stamp)
os.mkdir(model_folder_name)
model_folder_savings += model_folder_name


# %% tensorboard callback
os.chdir(model_folder_savings)
logdir = "\\tensorboard" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.mkdir(logdir)
logdir = model_folder_savings + logdir
batch_size = 22
tensorboard_callback = keras.callbacks.TensorBoard(
log_dir=logdir,
batch_size=batch_size,
update_freq="batch")


# %% Checkpoints callbacks
check_path=model_folder_savings+"weights.hdf5"
checkpointer = keras.callbacks.ModelCheckpoint(
filepath=check_path,
monitor = 'val_acc',
verbose=1,
save_best_only=True)

# %% Training parameters
kwargs = {"train_images":images_folder_train,
"train_annotations": annotations_images_folder_train,
"val_images":images_folder_val,
"val_annotations": annotations_images_folder_val,
"epochs": 20,
"batch_size":batch_size,
"val_batch_size": 22,
"steps_per_epoch":5300,
"verify_dataset":False,
"n_classes":134,
"load_weights":None,
"callbacks": [tensorboard_callback,checkpointer],
"validate":True,
"optimizer_name":'adadelta'}

# %% Model declaration
model = vgg_unet(n_classes=kwargs["n_classes"] ,  input_height=224, input_width=224)
winsound.Beep(1000,1000)

# %% loads a model
model.load_weights("D:\\Dataset\models\\model day 27 month 11 year 19 at 12 45 25\\weights.hdf5")

# %% Training
try:
  history = model.train(**kwargs)
  # notification for end of epoch.
  notification()
except Exception as e:
  notification()
  time.sleep(1)
  notification()
  save_model(model,model_folder_savings)
  print("Error in one training session : ")
  print(e)




# %% Save the model
save_model(model,model_folder_savings)

# %% Test
# validation image
os.chdir(images_folder_val)
img_file="000000000285.jpg"
img = mpimg.imread(img_file)
# annotation of the image
os.chdir(annotations_images_folder_val)
semantic_file = img_file[:-3]+'png'
img_semantic = plt.imread(semantic_file)
# prediction
os.chdir(images_folder_val)
out = model.predict_segmentation(
    inp=img_file,
    out_fname="out.png"
)
# display
plt.figure()
plt.subplot(1,3,1)
plt.axis("off")
plt.imshow(img)
plt.subplot(1,3,2)
plt.axis("off")
plt.imshow(out)
plt.subplot(1,3,3)
plt.axis("off")
plt.imshow(img_semantic)


# %% Visualization : actiavtion model
################################################################
###   Visualization
################################################################
layer_outputs = [layer.output for layer in model.layers][1:]
activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs) 


# %% Test with an input
os.chdir(images_folder_val)
img_file="000000000285.jpg"
# Conversion to tensor
img = keras.preprocessing.image.load_img(img_file, target_size=(224, 224))
img_tensor = keras.preprocessing.image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
plt.imshow(img_tensor[0])
plt.show()
# Prediction
activations = activation_model.predict(img_tensor) 

# %%
first_layer_activation = activations[0]
print(first_layer_activation.shape)

# %%
plt.matshow(first_layer_activation[0, :, :, 10], cmap='viridis')


# %% Displays all the activation maps
depth = 7
start_layer=27
end_layer=start_layer+depth
layer_names = [layer.name for layer in model.layers[start_layer:end_layer]]

# %%    
images_per_row = 10
for layer_name, layer_activation in zip(layer_names, activations[start_layer:end_layer]): # Displays the feature maps
  n_features = layer_activation.shape[-1] # Number of features in the feature map
  size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
  n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
  print("Processing layer {}, of shape {}".format(layer_name,layer_activation.shape))
  display_grid = np.zeros((size * n_cols, images_per_row * size))
  for col in range(n_cols): # Tiles each filter into a big horizontal grid
      for row in range(images_per_row):
          channel_image = layer_activation[0,:, :,col * images_per_row + row]
          channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
          channel_image /= channel_image.std()
          channel_image *= 64
          channel_image += 128
          channel_image = np.clip(channel_image, 0, 255).astype('uint8')
          display_grid[col * size : (col + 1) * size, # Displays the grid
                        row * size : (row + 1) * size] = channel_image
  scale = 1. / size
  plt.figure(figsize=(scale * display_grid.shape[1],scale * display_grid.shape[0]))
  plt.title(layer_name)
  plt.grid(False)
  cmap='jet'
  plt.imshow(display_grid, aspect='auto', cmap=cmap)

# %%
def visualize_activation_map(model,img_path,layer_name=[]):
  # building the activation model
  layer_outputs = [layer.output for layer in model.layers][1:]
  activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs) 
  # Convertion to tensor
  img = keras.preprocessing.image.load_img(img_path, target_size=(model.input_width, model.input_height))
  img_tensor = keras.preprocessing.image.img_to_array(img)
  img_tensor = np.expand_dims(img_tensor, axis=0)
  img_tensor /= 255.
  # Prediction
  activations = activation_model.predict(img_tensor) 
  # Visualization
  images_per_row = 10
  # Layers to visualize
  layers = [layer for layer in model.layers if layer.name in layer_name]
  # Visualization loop
  for layer in layers:
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    print("Processing layer {}, of shape {}".format(layer_name,layer_activation.shape))
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,:, :,col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                          row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    cmap='jet'
    plt.imshow(display_grid, aspect='auto', cmap=cmap)

# %%
visualize_activation_map(model,images_folder_val+"000000000285.jpg")

# %%
