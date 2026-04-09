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
import albumentations as A

def Image_transform_classify():
    return A.Compose([
        # Geometry
        A.RandomResizedCrop(
            (224,224),
            scale=(0.7, 1.0),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.05, 0.05),
            shear=(-5, 5),
            p=0.4         
        ),
        A.Rotate(limit=15, p=0.4),

        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.3,
            hue=0.05,        
            p=0.6
        ),    

        A.OneOf([
            A.GaussianBlur(blur_limit=3),
            A.MotionBlur(blur_limit=3),
        ], p=0.2),

        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(16, 32),
            hole_width_range=(16, 32),
            p=0.3
        )
    ],bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_visibility=0.3
        ))


def Image_transform_localize():
    return A.Compose([
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                shear=(-5, 5),
                p=1.0
            ),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),

            A.RandomSizedBBoxSafeCrop(224, 224, p=0.4),

            A.OneOf([
                A.RandomBrightnessContrast(0.08, 0.08),
                A.HueSaturationValue(5, 8, 5),
            ], p=0.4),

            A.OneOf([
                A.GaussianBlur(3),
                A.MotionBlur(3),
            ], p=0.2),

            A.GaussNoise(
                std_range=(0.02, 0.05),
                p=0.1
            ),

            A.CoarseDropout(
                num_holes_range=(1, 2),
                hole_height_range=(8, 16),
                hole_width_range=(8, 16),
                p=0.2
            ),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_visibility=0.3
        ))

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
        self.transform = transform() if transform != None else None

        for d in data:

            name,c_id,sp,brd = d.split(' ')
            class_id = int(c_id)-1
            img_path = os.path.join('data/images', name+'.jpg')
            
            self.data.append((img_path, class_id))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_id = self.data[idx]
        img = np.array(Image.open(img_path).convert('RGB'))

        if self.transform is not None:
            aug = self.transform(image=img, bboxes=[], labels=[])
            img = aug["image"]

        img = preprocess_img(Image.fromarray(img))
        
        return img, class_id
    


class OxfordIIITPetDataset_localize(Dataset):

    def __init__(self, data: list = [], transform=None):
        self.data = []
        self.transform = transform() if transform != None else None

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
        img = np.array(Image.open(img_path).convert('RGB'))

        if self.transform is not None:
            aug = self.transform(
                image=img,
                bboxes=[[xmin, ymin, xmax, ymax]],
                labels=[0]
            )
            img = aug["image"]
            xmin, ymin, xmax, ymax = aug["bboxes"][0]

        h, w = img.shape[:2]

        sx = 224 / w
        sy = 224 / h

        xmin *= sx
        xmax *= sx
        ymin *= sy
        ymax *= sy

        xmin = np.clip(xmin, 0, 224)
        xmax = np.clip(xmax, 0, 224)
        ymin = np.clip(ymin, 0, 224)
        ymax = np.clip(ymax, 0, 224)

        xc = (xmin + xmax) / 2
        yc = (ymin + ymax) / 2
        w  = xmax - xmin
        h  = ymax - ymin

        bbox = np.array([xc, yc, w, h], dtype=np.float32)

        img = preprocess_img(Image.fromarray(img))

        return img.astype(np.float32), bbox
    



class OxfordIIITPetDataset_Segmentation(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""
    
    def __init__(self, data: list=[], transform =None):
        
        self.data = []
        self.transform = transform() if transform != None else None

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
        img = np.array(Image.open(img_path).convert('RGB'))
        trimap = np.array(Image.open(seg_path).convert('L'))

        if self.transform:
            aug = self.transform(
                image=img,
                mask=trimap,
                bboxes=[],
                labels=[]
            )
            img = aug["image"]
            trimap = aug["mask"]

        trimap = Image.fromarray(trimap).resize((224, 224), Image.NEAREST)

        img = preprocess_img(Image.fromarray(img))
        img_tensor = torch.tensor(img)

        trimap_tensor = torch.tensor(np.array(trimap), dtype=torch.long) 
        trimap_tensor = trimap_tensor - 1 
        trimap_tensor[trimap_tensor == -1] = 255 

        return img_tensor, trimap_tensor


class OxfordIIITPetDataset_MultiTask(Dataset):
    """Oxford-IIIT Pet multi-task dataset: classification + localization + segmentation."""

    def __init__(self, data: list = [], transform=None):
        self.data = []
        self.transform = transform() if transform is not None else None

        for d in data:
            name, c_id, sp, brd = d.split(' ')
            class_id = int(c_id) - 1
            img_path = os.path.join('data/images', name + '.jpg')

   
            bbox = None
            xml_path = os.path.join('data/annotations/xmls', name + '.xml')
            if os.path.exists(xml_path):
                tree = ET.parse(xml_path)
                root = tree.getroot()
                obj = root.find('object')
                bbox = (
                    int(obj.find('bndbox/xmin').text),
                    int(obj.find('bndbox/ymin').text),
                    int(obj.find('bndbox/xmax').text),
                    int(obj.find('bndbox/ymax').text),
                )

  
            seg_path = os.path.join('data/annotations/trimaps', name + '.png')
            seg_path = seg_path if os.path.exists(seg_path) else None

            self.data.append((img_path, class_id, bbox, seg_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_id, bbox, seg_path = self.data[idx]

        img = np.array(Image.open(img_path).convert('RGB'))
        trimap = np.array(Image.open(seg_path).convert('L')) if seg_path else None
        xmin, ymin, xmax, ymax = bbox if bbox else (0, 0, 0, 0)

        if self.transform is not None:
            aug = self.transform(
                image=img,
                mask=trimap if trimap is not None else np.zeros(img.shape[:2], dtype=np.uint8),
                bboxes=[[xmin, ymin, xmax, ymax]] if bbox else [],
                labels=[0] if bbox else [],
            )
            img = aug["image"]
            trimap = aug["mask"] if seg_path else None
            if bbox and aug["bboxes"]:
                xmin, ymin, xmax, ymax = aug["bboxes"][0]

        bbox_tensor = torch.zeros(4, dtype=torch.float32)
        if bbox:
            h, w = img.shape[:2]
            sx, sy = 224 / w, 224 / h
            xmin = np.clip(xmin * sx, 0, 224)
            xmax = np.clip(xmax * sx, 0, 224)
            ymin = np.clip(ymin * sy, 0, 224)
            ymax = np.clip(ymax * sy, 0, 224)
            bbox_tensor = torch.tensor([
                (xmin + xmax) / 2,
                (ymin + ymax) / 2,
                xmax - xmin,
                ymax - ymin,
            ], dtype=torch.float32)

        if trimap is not None:
            trimap = Image.fromarray(trimap).resize((224, 224), Image.NEAREST)
            trimap_tensor = torch.tensor(np.array(trimap), dtype=torch.long) - 1
            trimap_tensor[trimap_tensor == -1] = 255
        else:
            trimap_tensor = torch.full((224, 224), 255, dtype=torch.long)  

    
        img_tensor = torch.tensor(preprocess_img(Image.fromarray(img)), dtype=torch.float32)

        return {
            "image":    img_tensor,       
            "class_id": class_id,    
            "bbox":     bbox_tensor,         
            "mask":     trimap_tensor,       
            "has_bbox": bbox is not None,     
            "has_mask": seg_path is not None, 
        }