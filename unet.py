#%%
from keras_segmentation.models.unet import vgg_unet

#%%
model = vgg_unet(n_classes=91 ,  input_height=640, input_width=640)

model.train(
    train_images =  "D:/val2017/",
    train_annotations = "D:/DIP-Segmentation/annoted",
    epochs=5
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