
#%%
"""Imports"""
import matplotlib.pyplot as plt
%matplotlib inline
import skimage.io as io
import png
import numpy as np
import time
import datetime
import os
from keras_segmentation.models.unet import vgg_unet
import cv2
from tqdm import tqdm
import json

from random import random
from matplotlib.colors import ListedColormap
randcolors = [(random(),random(),random()) for _ in range(255)]
rcmap = ListedColormap(randcolors)

ε = 1e-12

# path to COCO-dataset directory
sourcedir = 'E:/Cours/DIP/Coco'

#%%
"""Loading """
f = open(f'{sourcedir}/annotations/panoptic_train2017.json')
data = json.load(f)
f.close()
cats = {cat['id']:cat['name'] for cat in data['categories']}
cats[0] = 'nothing'


#%%
"""Model"""
n_classes = 201

model = vgg_unet(n_classes=n_classes ,  input_height=224, input_width=224)

images_folder_val = f"{sourcedir}/processed/validation"
images_folder_train = f"{sourcedir}/processed/training"
annotations_images_folder_val = f"{sourcedir}/processed/semantic_validation"
annotations_images_folder_train = f"{sourcedir}/processed/semantic_train"


#%%
"""Loading"""
# model.load_weights("D:/processed/model19-11-19-(01_21_16)-epoch_1.h5")
model.load_weights(f"{sourcedir}/modelday 22 month 11 year 19 at 00 46 12 epoch 2.h5")


#%%
"""Assess prediction on one sample"""
# images_folder_val = f"{sourcedir}/processed/flare08"
images_folder_val = f"{sourcedir}/processed/validation"
img_name = "000000000724"
img_in = "{}/{}.jpg".format(images_folder_val,img_name)
img_out = "{}/{}.png".format(annotations_images_folder_val,img_name)
image = io.imread(img_in)
truth = io.imread(img_out)
pred = model.predict_segmentation(
  inp=img_in
)
pred = cv2.resize(pred, truth.shape[::-1], interpolation=cv2.INTER_NEAREST)

plt.rcParams['figure.figsize'] = [15, 8]
plt.subplot(1,3,1)
plt.axis('off')
plt.imshow(image)
plt.title("Image")
plt.subplot(1,3,2)
plt.axis('off')
plt.imshow(truth, cmap=rcmap, vmin=0, vmax=200)
plt.title("Ground truth")
plt.subplot(1,3,3)
plt.axis('off')
plt.imshow(pred, cmap=rcmap, vmin=0, vmax=200)
plt.title("Prediction")
plt.show()
plt.rcParams['figure.figsize'] = [6.4, 4.8]

occurances_truth = np.zeros(n_classes)
occurances_pred = np.zeros(n_classes)
precision = np.zeros(n_classes)
recall = np.zeros(n_classes)
iou = np.zeros(n_classes)
dice = np.zeros(n_classes)
accuracy = 0
for cl in range(n_classes):
  occurances_truth[cl] = np.sum(truth == cl)
  occurances_pred[cl] = np.sum(pred == cl)
  intersection = np.sum(( truth == cl )*( pred == cl ))
  union = np.sum(np.maximum( ( truth == cl ) , ( pred == cl ) ))
  precision[cl] = intersection/(occurances_truth[cl]+ε)
  recall[cl] = intersection/(occurances_pred[cl]+ε)
  iou[cl] = intersection/(union+ε)
  dice[cl] = 2*intersection/(occurances_truth[cl]+occurances_pred[cl]+ε)
  accuracy += intersection
accuracy /= (truth.shape[0]*truth.shape[1])
uiou = sum([iou[cl] for cl in range(n_classes)])/len(set(truth.flatten()))
wiou = sum([iou[cl]*np.sum(truth == cl) for cl in range(n_classes)])/(truth.shape[0]*truth.shape[1])

hist = [(cats[catid], occurances_truth[catid], occurances_pred[catid], precision[catid], recall[catid], iou[catid], dice[catid]) for catid in cats.keys()]
hist = [histel for histel in hist if histel[1]+histel[2]>5]
hist.sort(reverse=True, key=lambda tup: tup[1])  # sorts in place
cl,ot,op,precision,recall,iou,dice = zip(*hist)
r1 = range(len(cl))
plt.barh([x+0.2 for x in r1], precision, height=0.4, align='center', color='Goldenrod', label='Precision')
plt.barh([x-0.2 for x in r1], recall, height=0.4, align='center', color='IndianRed', label='Recall')
plt.yticks(range(len(cl)),cl)
plt.legend()
plt.xlim(0.0,1.0)
plt.show()
plt.barh([x+0.2 for x in r1], iou, height=0.4, align='center', color='DarkSeaGreen', label='IoU')
plt.barh([x-0.2 for x in r1], dice, height=0.4, align='center', color='LightSeaGreen', label='Dice')
plt.yticks(range(len(cl)),cl)
plt.axvline(x=accuracy, label="Accuracy", color='Gray', alpha=0.5, dashes=[1,1], linewidth=2)
plt.axvline(x=uiou, label="Uniform IoU", color='DarkSeaGreen', alpha=0.5, dashes=[4,2], linewidth=2)
plt.axvline(x=wiou, label="Weighted IoU", color='DarkSeaGreen', alpha=0.5, dashes=[2,2], linewidth=2)
plt.legend()
plt.xlim(0.0,1.0)
plt.show()
plt.barh([x+0.2 for x in r1], ot, height=0.4, align='center', color='RosyBrown', label='Ground truth')
plt.barh([x-0.2 for x in r1], op, height=0.4, align='center', color='MediumPurple', label='Prediction')
plt.yticks(range(len(cl)),cl)
plt.legend()
plt.xlabel('Number of pixels')
plt.show()

# cv2.imwrite("out_pred_resized.png", pred)
# cv2.imwrite("outbnwref.png", truth)


#%%
"""Assess prediction on validation dataset"""
uious = [] 
wious = []
for img_path in tqdm(os.listdir(images_folder_val)):
  img_name = img_path[:-4]
  img_in = "{}/{}.jpg".format(images_folder_val,img_name)
  img_out = "{}/{}.png".format(annotations_images_folder_val,img_name)
  img_ref = io.imread(img_out)
  pred = model.predict_segmentation(inp=img_in)
  pred_resized = cv2.resize(pred, img_ref.shape[::-1], interpolation=cv2.INTER_NEAREST)
  ious = np.zeros(n_classes)
  for cl in range(n_classes):
    intersection = np.sum(( img_ref == cl )*( pred_resized == cl ))
    union = np.sum(np.maximum( ( img_ref == cl ) , ( pred_resized == cl ) ))
    ious[cl] = intersection/(union+ε)
  uiou = sum([ious[cl] for cl in range(n_classes)])/len(set(img_ref.flatten()))
  wiou = sum([ious[cl]*np.sum(img_ref == cl) for cl in range(n_classes)])/(img_ref.shape[0]*img_ref.shape[1])
  uious.append(uiou)
  wious.append(wiou)
  # print("uiou = {:0.3f}  ;  wiou = {:0.3f}".format(uiou, wiou))
avg_uiou = sum(uious)/len(uious)
avg_wiou = sum(wious)/len(wious)
print("mean uiou = {:0.3f}  ;  mean wiou = {:0.3f}".format(avg_uiou, avg_wiou))
plt.plot(range(len(uious)), sorted(uious), label="uIoU", color="GoldenRod", linewidth=3)
plt.plot(range(len(wious)), sorted(wious), label="iIoU", color="MediumPurple", linewidth=3)
plt.axhline(y=avg_uiou, label="Mean uIoU", color='GoldenRod', alpha=0.5, dashes=[4,2], linewidth=2)
plt.axhline(y=avg_wiou, label="Mean iIoU", color='MediumPurple', alpha=0.5, dashes=[4,2], linewidth=2)
plt.legend()
plt.xlabel("Image rank")
plt.show()



#%%
"""Assess robustness on altered dataset"""
muious = []
mwious = []
minuious = []
minwious = []
maxuious = []
maxwious = []
maxnbr = 1000
# runs = [
#   {
#     'folder': 'validation',
#     'label': "Orignal",
#     'alpha': 1.0
#   },
#   {
#     'folder': 'chrab1',
#     'label': "Chom. aber. 1",
#     'alpha': 0.8
#   },
#   {
#     'folder': 'chrab2',
#     'label': "Chom. aber. 2",
#     'alpha': 0.6
#   },
#   {
#     'folder': 'chrab3',
#     'label': "Chom. aber. 3",
#     'alpha': 0.4
#   }
# ]
# runs = [
#   {
#     'folder': 'validation',
#     'label': "Orignal",
#     'alpha': 1.0
#   },
#   {
#     'folder': 'hue05',
#     'label': "+5 hue",
#     'alpha': 0.8
#   },
#   {
#     'folder': 'hue10',
#     'label': "+10 hue",
#     'alpha': 0.6
#   },
#   {
#     'folder': 'hue25',
#     'label': "+25 hue",
#     'alpha': 0.4
#   }
# ]
# runs = [
#   {
#     'folder': 'validation',
#     'label': "Orignal",
#     'alpha': 1.0
#   },
#   {
#     'folder': 'rain025',
#     'label': "25% rain",
#     'alpha': 0.7
#   },
#   {
#     'folder': 'rain05',
#     'label': "50% rain",
#     'alpha': 0.4
#   }
# ]
runs = [
  {
    'folder': 'validation',
    'label': "Orignal",
    'alpha': 1.0
  },
  {
    'folder': 'blur2',
    'label': "Blur 2",
    'alpha': 0.8
  },
  {
    'folder': 'blur4',
    'label': "Blur 4",
    'alpha': 0.6
  },
  {
    'folder': 'blur8',
    'label': "Blur 8",
    'alpha': 0.4
  }
]
# runs = [
#   {
#     'folder': 'validation',
#     'label': "Orignal",
#     'alpha': 1.0
#   },
#   {
#     'folder': 'jpeg50',
#     'label': "JPEG 50%",
#     'alpha': 0.8
#   },
#   {
#     'folder': 'jpeg25',
#     'label': "JPEG 25%",
#     'alpha': 0.6
#   },
#   {
#     'folder': 'jpeg10',
#     'label': "JPEG 10%",
#     'alpha': 0.4
#   },
#   {
#     'folder': 'jpeg05',
#     'label': "JPEG 5%",
#     'alpha': 0.2
#   }
# ]
# runs = [
#   {
#     'folder': 'validation',
#     'label': "Orignal",
#     'alpha': 1.0
#   },
#   {
#     'folder': 'sat08',
#     'label': "80% Saturation",
#     'alpha': 0.8
#   },
#   {
#     'folder': 'sat05',
#     'label': "50% Saturation",
#     'alpha': 0.6
#   },
#   {
#     'folder': 'sat025',
#     'label': "25% Saturation",
#     'alpha': 0.4
#   },
#   {
#     'folder': 'sat010',
#     'label': "10% Saturation",
#     'alpha': 0.2
#   }
# ]
# runs = [
#   {
#     'folder': 'validation',
#     'label': "Orignal",
#     'alpha': 1.0
#   },
#   {
#     'folder': 'flare05',
#     'label': "50% flare",
#     'alpha': 0.8
#   },
#   {
#     'folder': 'flare08',
#     'label': "80% flare",
#     'alpha': 0.6
#   },
#   {
#     'folder': 'flare10',
#     'label': "100% flare",
#     'alpha': 0.4
#   }
# ]
for run in runs:
  images_folder_val = f"{sourcedir}/processed/{run['folder']}"
  uious = []
  wious = []
  nbr = 0
  for img_path in tqdm(os.listdir(images_folder_val)):
    nbr += 1
    img_name = img_path[:-4]
    img_in = "{}/{}.jpg".format(images_folder_val,img_name)
    img_out = "{}/{}.png".format(annotations_images_folder_val,img_name)
    img_ref = io.imread(img_out)
    pred = model.predict_segmentation(inp=img_in)
    pred_resized = cv2.resize(pred, img_ref.shape[::-1], interpolation=cv2.INTER_NEAREST)
    ious = np.zeros(n_classes)
    for cl in range(n_classes):
      intersection = np.sum(( img_ref == cl )*( pred_resized == cl ))
      union = np.sum(np.maximum( ( img_ref == cl ) , ( pred_resized == cl ) ))
      ious[cl] = intersection/(union+ε)
    uiou = sum([ious[cl] for cl in range(n_classes)])/len(set(img_ref.flatten()))
    wiou = sum([ious[cl]*np.sum(img_ref == cl) for cl in range(n_classes)])/(img_ref.shape[0]*img_ref.shape[1])
    uious.append(uiou)
    wious.append(wiou)
    if nbr >= maxnbr:
      break
    # print("uiou = {:0.3f}  ;  wiou = {:0.3f}".format(uiou, wiou))
  avg_uiou = sum(uious)/len(uious)
  avg_wiou = sum(wious)/len(wious)
  muious.append(avg_uiou)
  mwious.append(avg_wiou)
  minuious.append(min(uious))
  minwious.append(min(wious))
  maxuious.append(max(uious))
  maxwious.append(max(wious))
  plt.plot(range(len(uious)), sorted(uious), label="{} uIoU".format(run['label']), color="GoldenRod", alpha=run['alpha'], linewidth=3)
  plt.plot(range(len(wious)), sorted(wious), label="{} iIoU".format(run['label']), color="MediumPurple", alpha=run['alpha'], linewidth=3)
  plt.axhline(y=avg_uiou, color='GoldenRod', alpha=run['alpha'], dashes=[1,1], linewidth=2)
  plt.axhline(y=avg_wiou, color='MediumPurple', alpha=run['alpha'], dashes=[1,1], linewidth=2)
plt.legend()
plt.xlabel("Image rank")
plt.show()
plt.plot(range(len(muious)), muious, label="uIoU", color="GoldenRod", linewidth=3, marker='o')
plt.plot(range(len(mwious)), mwious, label="iIoU", color="MediumPurple", linewidth=3, marker='o')
plt.fill_between(range(len(muious)), minuious, maxuious, color="GoldenRod", alpha=0.1)
plt.fill_between(range(len(mwious)), minwious, maxwious, color="MediumPurple", alpha=0.1)
plt.legend()
ticks = [run['label'] for run in runs]
plt.xticks(range(len(ticks)), ticks)
plt.show()

