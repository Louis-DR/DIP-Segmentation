#%%
from pycocotools.coco import COCO
import numpy as np
import png

from tqdm import tqdm
import winsound
frequency = 1000  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second


#%%
dataDir='D:/'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)



#%%
cat_ids = coco.getCatIds()
coco_cats = coco.loadCats(cat_ids)
cats = ['nothing']
for coco_cat in coco_cats:
    cats.append(coco_cat['name'])

img_ids = coco.getImgIds()
img_ids.sort()
for img_id in tqdm(img_ids):
    img = coco.loadImgs(img_id)[0]
    anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)
    anns_img = np.zeros((img['height'],img['width']))
    for ann in anns:
        anns_img = np.maximum(anns_img,coco.annToMask(ann)*ann['category_id'])
    png.from_array(anns_img.astype(int).tolist(), 'L').save("annoted/{:012d}.png".format(img['id']))
winsound.Beep(frequency, duration)

