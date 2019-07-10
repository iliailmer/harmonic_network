import pandas as pd
import gc
import os
from skimage.io import imread, imsave
from skimage.transform import resize
from additions import add_edges, alpha_rooting_fourier
from skimage.util import img_as_float32, img_as_uint, img_as_ubyte
from hair_removal import hair_removal
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
metadata = pd.read_csv('ISIC2019/ISIC_2019_Training_GroundTruth.csv') #metadata/HAM10000_metadata.csv')

imageid_path_dict = {x: f'ISIC2019/ISIC_2019_Training_Input/{x}.jpg' for x in metadata.image}

del metadata
gc.collect()

ids = list(imageid_path_dict.keys())
present_224 = os.listdir('ISIC2019/ISIC_2019_Training_224/')
present_96 = os.listdir('ISIC2019/ISIC_2019_Training_96/')

for idx, each in enumerate(tqdm(ids)):
    if not(f'{each}.jpg' in present_224):
        image = img_as_float32(imread(imageid_path_dict[each]))
        image = hair_removal(image)
        image = 0.9*add_edges(image, 0.9, 0.1)
        image = alpha_rooting_fourier(image, alpha=0.98)
        image = resize(image, (224, 224), mode='reflect', anti_aliasing=True)
        imsave(f'ISIC2019/ISIC_2019_Training_224/{each}.jpg', img_as_ubyte(image))
        image = resize(image, (96, 96), mode='reflect', anti_aliasing=True)
        imsave(f'ISIC2019/ISIC_2019_Training_224/{each}.jpg', img_as_ubyte(image))
        gc.collect()
