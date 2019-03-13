import pandas as pd
import gc
import os
from skimage.io import imread, imsave
from skimage.transform import resize
from additions import add_edges
from skimage.util import img_as_float32, img_as_uint
from hair_removal import hair_removal
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
metadata = pd.read_csv('metadata/HAM10000_metadata.csv')

imageid_path_dict = {x: f'HAM10000_images/{x}.jpg' for x in metadata.image_id}

del metadata
gc.collect()

ids = list(imageid_path_dict.keys())
present = os.listdir('HAM10000_small')
for idx, each in enumerate(tqdm(ids)):
    if not(f'{each}.jpg' in present):
        image = img_as_float32(imread(imageid_path_dict[each]))
        image = hair_removal(image)
        image = add_edges(image, 0.5, 0.5)
        image = resize(image, (64, 64), mode='reflect', anti_aliasing=True)
        imsave(f'HAM10000_small/{each}.jpg', img_as_uint(image))
        gc.collect()
