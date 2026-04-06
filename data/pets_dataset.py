"""Dataset skeleton for Oxford-IIIT Pet.
"""

from torch.utils.data import Dataset
import os
import torch
from PIL import Image
from PIL import ImageEnhance
from PIL.ImageTransform import AffineTransform
from PIL.ImageOps import mirror
import numpy as np
import xml.etree.ElementTree as ET
import random

def Image_transform(img):
    eps = 0.1              # increase
    translate = 10         # increase

    affine_coeffs = (
        1 + random.uniform(-eps, eps),
        random.uniform(-eps, eps),
        random.uniform(-translate, translate),
        random.uniform(-eps, eps),
        1 + random.uniform(-eps, eps),
        random.uniform(-translate, translate)
    )

    t = AffineTransform(affine_coeffs)
    flip = random.random() > 0.5        

    e = img if flip else mirror(img)
    e = t.transform(e.size, e)
    e = ImageEnhance.Brightness(e).enhance(random.uniform(0.7, 1.3))
    e = ImageEnhance.Contrast(e).enhance(random.uniform(0.7, 1.3))
    e = ImageEnhance.Color(e).enhance(random.uniform(0.7, 1.3))

    return e, t, flip 

def preprocess_img(img):

    img = img.resize((224,224), Image.LANCZOS)
    img = np.array(img)/255.

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = (img - mean) / std
    img = np.transpose(img, (2,0,1))
    return img.astype(np.float32)


class OxfordIIITPetDataset_classify(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""
    
    def __init__(self, data: list=[], transform =None):
        
        self.data = []
        self.transform = transform

        for d in data:

            name,c_id,sp,brd = d.split(' ')
            class_id = int(c_id)-1
            img_path = os.path.join('data/images', name+'.jpg')
            
            self.data.append((img_path, class_id))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_id = self.data[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform != None:
            img,_,_ = self.transform(img)

        img = preprocess_img(img)
        
        return img, class_id
    


class OxfordIIITPetDataset_localize(Dataset):

    def __init__(self, data: list = [], transform=None):
        self.data = []
        self.transform = transform

        for d in data:
            name, c_id, sp, brd = d.split(' ')
            xml_path = os.path.join('data/annotations/xmls', name + '.xml')
            if not os.path.exists(xml_path):
                continue

            img_path = os.path.join('data/images', name + '.jpg')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            obj  = root.find('object')

            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text) 

            self.data.append((img_path, (xmin, ymin, xmax, ymax)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, (xmin, ymin, xmax, ymax) = self.data[idx]
        img = Image.open(img_path).convert('RGB')

        orig_w, orig_h = img.size     

        if self.transform is not None:
            img, t, flip = self.transform(img)

            if not flip:
                xmin, xmax = orig_w - xmax, orig_w - xmin 

            a, b, tx, c, d, ty = t.data
            M = np.array([
                [a, b, tx],
                [c, d, ty]
            ])
            corners = np.array([
                [xmin, ymin, 1],
                [xmax, ymin, 1],
                [xmin, ymax, 1],
                [xmax, ymax, 1],
            ]).T

            transformed = M @ corners
            xmin = transformed[0].min()
            ymin = transformed[1].min()
            xmax = transformed[0].max()
            ymax = transformed[1].max()
        
        sx = 224 / orig_w
        sy = 224 / orig_h

        xmin, ymin, xmax, ymax = (
            float(np.clip(xmin * sx, 0, 224)),
            float(np.clip(ymin * sy, 0, 224)),
            float(np.clip(xmax * sx, 0, 224)),
            float(np.clip(ymax * sy, 0, 224)),
        )

        xc = (xmin + xmax) / 2
        yc = (ymin + ymax) / 2
        w  =  xmax - xmin
        h  =  ymax - ymin

        bbox = np.array([xc, yc, w, h], dtype=np.float32)
        img = preprocess_img(img)

        return img.astype(np.float32), bbox
    



class OxfordIIITPetDataset_Segmentation(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""
    
    def __init__(self, data: list=[], transform =None):
        
        self.data = []
        self.transform = transform

        for d in data:

            name,c_id,sp,brd = d.split(' ')
            
            seg_path = os.path.join('data/annotations/trimaps',name+'.png') 
            img_path = os.path.join('data/images', name+'.jpg')
            if not os.path.exists(seg_path):
                continue
            
            self.data.append((img_path, seg_path))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, seg_path = self.data[idx]
        img   = Image.open(img_path).convert('RGB')
        trimap = Image.open(seg_path).convert('L')

        if self.transform:
            img, t, flip = Image_transform(img)

            trimap = trimap if flip else mirror(trimap)
            trimap = t.transform(trimap.size, trimap, resample=Image.NEAREST)
            trimap = trimap.resize((224, 224), Image.NEAREST)
        else:
            trimap = trimap.resize((224, 224), Image.NEAREST)

        img = preprocess_img(img)
        img_tensor = torch.tensor(img)

        trimap_tensor = torch.tensor(np.array(trimap), dtype=torch.long) 
        trimap_tensor = trimap_tensor - 1 
        trimap_tensor[trimap_tensor == -1] = 255 

        return img_tensor, trimap_tensor

        