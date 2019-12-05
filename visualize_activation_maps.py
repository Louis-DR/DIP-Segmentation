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


# %% Folders creation of the training
# images
images_folder_val = "D:\\Dataset\\coco2017\\val2017\\val2017\\"
images_folder_train = "D:\\Dataset\\coco2017\\train2017\\train2017\\"
# annotations
annotations_images_folder_val = "D:\\Dataset\\new_seg\\semantic_validation\\semantic_validation\\"
annotations_images_folder_train = "D:\\Dataset\\new_seg\\semantic_train\\semantic_train\\"
# Altered images
dist_images_path = "C:\\Users\\Arnaud\\Google Drive\\shared\\ECE6258\\SAMPLE"
# model folder for storage on the hard drive
model_folder_savings = "D:\\Dataset\\models\\"

# %% Model declaration
model = vgg_unet(n_classes=134 ,  input_height=224, input_width=224)
winsound.Beep(1000,1000)

# %% loads a model
model.load_weights("D:\\Dataset\models\\model day 27 month 11 year 19 at 12 45 25\\weights.hdf5")


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


# %%
def visualize_activation_map(model,img_path,layer_names=[],save=False):
  # building the activation model,; the input is removed since it
  # it is not a layer
  layer_outputs = [layer.output for layer in model.layers if not layer.name.startswith('input')]
  activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
  # Convertion to tensor
  img = keras.preprocessing.image.load_img(img_path, target_size=(model.input_width, model.input_height))
  img_tensor = keras.preprocessing.image.img_to_array(img)
  img_tensor = np.expand_dims(img_tensor, axis=0)
  img_tensor /= 255.
  # Prediction
  activations = activation_model.predict(img_tensor)
  assert(len(layer_outputs)==len(activations))
  # Visualization
  images_per_row = 10
  # Layers to visualize
  layers = [(layer.name, activation) for layer,activation in zip(model.layers[1:],activations) if layer.name in layer_names]
  # Visualization loop
  for layer in layers:
    layer_name, layer_activation = layer
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    print("Processing layer {}, of shape {}".format(layer_name,layer_activation.shape))
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,:, :,col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()  + 1e-5
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
    plt.axis('off')
    plt.imshow(display_grid, aspect='auto', cmap=cmap)
    if save:
        os.chdir(img_path[:-16])
        plt.savefig('{}_{}.svg'.format(img_path.split('\\')[-1][:-3],layer_name), format='svg')

# %%

images=[
images_folder_val+"000000000285.jpg",
images_folder_val+"000000000785.jpg",
dist_images_path + "\\sat08\\" + "000000000285.jpg",
dist_images_path + "\\sat05\\" + "000000000285.jpg",
dist_images_path + "\\sat01\\" + "000000000285.jpg",
dist_images_path + "\\rain025\\" + "000000000285.jpg",
dist_images_path + "\\rain05\\" + "000000000285.jpg",
dist_images_path + "\\noisesnp\\" + "000000000285.jpg",
dist_images_path + "\\noisegaus\\" + "000000000285.jpg",
dist_images_path + "\\hue25\\" + "000000000285.jpg",
dist_images_path + "\\hue10\\" + "000000000285.jpg",
dist_images_path + "\\hue05\\" + "000000000285.jpg",
dist_images_path + "\\flare10\\" + "000000000285.jpg",
dist_images_path + "\\flare08\\" + "000000000285.jpg",
dist_images_path + "\\flare05\\" + "000000000285.jpg",
dist_images_path + "\\chrab1\\" + "000000000285.jpg",
dist_images_path + "\\chrab2\\" + "000000000285.jpg",
dist_images_path + "\\chrab3\\" + "000000000285.jpg",
dist_images_path + "\\blurmed\\" + "000000000285.jpg",
dist_images_path + "\\blurgaus\\" + "000000000285.jpg",
#images_folder_val+"000000000285.jpg",
images_folder_val+"000000000724.jpg",
images_folder_val+"000000036660.jpg"
]
for img_path in images:
    visualize_activation_map(
        model,
        img_path,
        layer_names=["block1_conv2","block2_conv2"],
        save=True
    )


# %%
def visualize_activation_map_single(model,img_path,layer_names=[]):
  # building the activation model,; the input is removed since it
  # it is not a layer
  layer_outputs = [layer.output for layer in model.layers if not layer.name.startswith('input')]
  activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
  # Convertion to tensor
  img = keras.preprocessing.image.load_img(img_path, target_size=(model.input_width, model.input_height))
  img_tensor = keras.preprocessing.image.img_to_array(img)
  img_tensor = np.expand_dims(img_tensor, axis=0)
  img_tensor /= 255.
  # Prediction
  activations = activation_model.predict(img_tensor)
  assert(len(layer_outputs)==len(activations))
  # Layers to visualize
  layers = [(layer.name, activation) for layer,activation in zip(model.layers[1:],activations) if layer.name in layer_names]
  # Visualization loop
  for layer in layers:
    layer_name, layer_activation = layer
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    print("Processing layer {}, of shape {}".format(layer_name,layer_activation.shape))
    feature = 60
    channel_image = layer_activation[0,:,:,feature]
    plt.figure()
    plt.title(layer_name)
    plt.grid(False)
    cmap='jet'
    plt.axis('off')
    plt.imshow(channel_image, aspect='auto', cmap=cmap)
    #plt.savefig('{}{}.eps'.format(img_path.split('\\')[-1],layer_name), format='eps')

# %%

images=[
images_folder_val+"000000000785.jpg",
images_folder_val+"000000000285.jpg",
images_folder_val+"000000000724.jpg",
images_folder_val+"000000036660.jpg"
]
for img_path in images:
    visualize_activation_map_single(
        model,
        img_path,
        layer_names=["conv2d_4"]
    )


# %%


# %%
