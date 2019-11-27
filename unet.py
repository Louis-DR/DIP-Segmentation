#%%
from keras_segmentation.models.unet import vgg_unet
import ipykernel # to fix a bug with Keras verbose
from keras_tqdm import TQDMNotebookCallback

#%%
model = vgg_unet(n_classes=201 ,  input_height=640, input_width=640)

model.train(
    train_images =  "D:/processed/validation",
    train_annotations = "D:/processed/semantic_validation",
    epochs=10, steps_per_epoch=64
)

#%%
out = model.predict_segmentation(
    inp="dataset1/images_prepped_test/0016E5_07965.png",
    out_fname="/tmp/out.png"
)

import matplotlib.pyplot as plt
plt.imshow(out)

# evaluating the model 
print(model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )