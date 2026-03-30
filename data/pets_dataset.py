"""Dataset skeleton for Oxford-IIIT Pet.
"""

from torch.utils.data import Dataset
import os
from PIL import Image
from PIL import ImageEnhance
from PIL.ImageTransform import AffineTransform
from PIL.ImageOps import mirror
import numpy as np
import random

def Image_transform(img:Image):
    
    # flip with 0.5 prob -> Affine -> color changes

    eps = 0.02 ; translate = 2
    affine_coeffs = (
        1 + random.uniform(-eps, eps),  # a
        random.uniform(-eps, eps),      # b
        random.uniform(-translate, translate),  # c
        random.uniform(-eps, eps),      # d
        1 + random.uniform(-eps, eps),  # e
        random.uniform(-translate, translate)   # f
    )


    t = AffineTransform(affine_coeffs)
    
    e = img if random.random()>0.5 else mirror(img)
    e = t.transform(e.size,e)
    e = e.resize((224,224), Image.LANCZOS)

    e = ImageEnhance.Brightness(e).enhance(random.uniform(0.9, 1.1))
    e = ImageEnhance.Contrast(e).enhance(random.uniform(0.9, 1.1))
    e = ImageEnhance.Color(e).enhance(random.uniform(0.9, 1.1))

    return e


class OxfordIIITPetDataset_classify(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""
    
    def __init__(self, data: list=[], transform =None):
        
        self.data = []
        self.transform = transform
        self.one_hot = np.eye(37)

        for d in data:

            name,c_id,sp,brd = d.split(' ')
            class_id = int(c_id)-1
            img_path = os.path.join('images', name+'.jpg')
            
            self.data.append((img_path, class_id))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        img_path,class_id = self.data[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform!=None:
            img = self.transform(img)
        
        img = img.resize((224,224),resample=Image.LANCZOS)

        return np.array(img), self.one_hot[class_id]