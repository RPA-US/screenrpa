from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import torch
import json
import os
import pycocotools.mask as mask_utils
import cv2
from apps.featureextraction.SOM.segment_anything.utils.transforms import ResizeLongestSide


class Labelme2SAMDataset(Dataset):
    def __init__(self, root_path, labelme_path, path_to_save_masks, img_size, create=False):
        self.root_path = root_path
        # self.transform = transform
        self.img_size = img_size
        self.labelme_path = labelme_path
        self.mask_path = path_to_save_masks
        self.images = list(filter(lambda x:len(x.split('.'))>1, os.listdir(self.root_path)))
        # self.images = [dir for dir in os.listdir(self.root_path) if len(dir.split('.'))>0]
        # print(self.ima)
        self.masks = self.create_masks() if create else os.listdir(self.mask_path)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        image_name = self.images[idx]
        mask_name = self.masks[idx]
        image_path = os.path.join(self.root_path,image_name)
        mask_path = os.path.join(self.mask_path, mask_name)
        
        # image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
        image = cv2.imread(image_path)

        with open(mask_path,'r') as f:
            mask_json = json.load(f)
        
        bboxes=[]
        binary_masks=[]

        for ann in mask_json['annotations']:
            ann_copy = ann.copy()
            bboxes.append(ann_copy['bbox'])
            ann_copy['segmentation']['counts']= bytes(ann_copy['segmentation']['counts'])
            binary_masks.append(mask_utils.decode(ann_copy['segmentation']))

        bboxes = np.stack(bboxes,axis=0)
        binary_masks = np.stack(binary_masks, axis=0)
        
        # image = transforms.ToTensor()(image)
        # if self.transform:
        #     image = self.transform(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        transform = ResizeLongestSide(self.img_size)
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image)
        # transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        
        # input_image = sam_model.preprocess(transformed_image)
        original_image_size = image.shape[:2]
        # input_size = tuple(transformed_image.shape[-2:])

        return original_image_size,input_image_torch, torch.tensor(bboxes), torch.tensor(binary_masks)

    def create_masks(self):
        idx = 0
        for labelme, image_name in zip(os.listdir(self.labelme_path), self.images):
            json_path = os.path.join(self.labelme_path, labelme)
            img_path = os.path.join(self.root_path,image_name)
            mask_path = os.path.join(self.mask_path, labelme)
            img = Image.open(img_path)
            width, height = img.size 
            annotations = self.labelme_to_mask(json_path, height, width)

            image_dic = {}
            image_dic['image'] = {'image_id':idx, 'width':width, 'height':height, 'file_name':image_name}
            image_dic['annotations']=annotations

            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            with open(mask_path,'w') as f:
                json.dump(image_dic, f, indent=1)

            idx+=1        
        return os.listdir(self.mask_path)

    def labelme_to_mask(self, json_path, image_height, image_width):
        with open(json_path, 'r') as f:
            data = json.load(f)

        annotations=[]
        for idx,obj in enumerate(data['shapes']):
            mask = np.zeros((image_height, image_width), dtype=np.uint8)
            a,b = obj['points']
            x_min,y_min = int(np.floor(a[0])), int(np.floor(a[1]))
            x_max,y_max = int(np.ceil(b[0])), int(np.ceil(b[1]))
            mask[y_min:y_max, x_min:x_max] = 1
            fmask = np.asfortranarray(mask)
            rle = mask_utils.encode(fmask)
            # print(bytes(rle['counts'],'utf-8'))
            rle['counts']=list(rle['counts'])
            
            # print(rle)
            w = x_max-x_min
            h = y_max-y_min
            # polygon = np.array(obj['points'], dtype=np.int32)
            # polygon_mask = Image.new('L',(image_width, image_height),0)
            # ImageDraw.Draw(polygon_mask).polygon(polygon, outline=1, fill=1)
            # mask = np.maximum(mask, np.array(polygon))
            annotation={}
            annotation['id']=idx
            annotation['segmentation']=rle
            annotation['bbox']=[x_min,y_min,w,h]
            annotation['area']=w*h
            annotations.append(annotation)
        return annotations
    


