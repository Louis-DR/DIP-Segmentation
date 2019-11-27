#%%
import skimage.io as io
import numpy as np
import png
import json

from tqdm.autonotebook import tqdm
import winsound
frequency = 1000  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second


#%%
# f = open('D:/annotations/panoptic_val2017.json')
f = open('E:/Cours/DIP/Coco/panoptic_annotations_trainval2017/annotations/panoptic_train2017.json')
data = json.load(f)
f.close()


# %%
cats = {}
for cat in data['categories'] :
    cats[cat['id']] = cat['name']


# %%
id_remap = {}
for id_old,id_new in zip(cats.keys(),range(1,len(cats.keys())+1)):
    id_remap[id_old] = id_new


# %%
anns = {}
for ann_data in tqdm(data['annotations']):
    anns[ann_data['image_id']] = ann_data['segments_info']


# %%
from os import listdir
filelist = listdir("F:/segmentation_train")
for img_data in tqdm(data['images']):
    if img_data['file_name'].replace('jpg','png') in filelist: continue  
    # pan_img = io.imread("D:/panoptic_val2017/{:012d}.png".format(img_data['id']))
    pan_img = io.imread("F:/panoptic_train2017/panoptic_train2017/{:012d}.png".format(img_data['id']))
    red, green, blue = pan_img[:,:,0], pan_img[:,:,1], pan_img[:,:,2]
    for ann in anns[img_data['id']]:
        ann_id = ann['id']
        rgb = [ann_id%256, int(ann_id/256)%256, int(ann_id/256**2)%256]
        mask = (red == rgb[0]) & (green == rgb[1]) & (blue == rgb[2])
        pix = id_remap[ann['category_id']]
        pan_img[:,:,:3][mask] = [pix, pix, pix]
    io.imsave('F:/segmentation_train/{:012d}.png'.format(img_data['id']),pan_img[:,:,0])    
    

winsound.Beep(frequency, duration)



# %%
