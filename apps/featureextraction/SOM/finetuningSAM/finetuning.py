from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import torch
import json
import os
import pycocotools.mask as mask_utils
from clasesFineTuning import Labelme2SAMDataset
import cv2
import time
import matplotlib.pyplot as plt
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import SamPredictor, sam_model_registry
import torch.nn.functional as F
from statistics import mean


#Variables Universales
CHECKPOINT_PATH='checkpoints/'
RESOURCES='../../../resources/'
sam_checkpoint = "sam_vit_l_0b3195.pth"
model_type = "vit_l"

# Loading the model
sam_model = sam_model_registry["vit_l"](checkpoint=CHECKPOINT_PATH+sam_checkpoint)
predictor = SamPredictor(sam_model)

transform = ResizeLongestSide(sam_model.image_encoder.img_size)


time0 = time.time()
data_v2 = Labelme2SAMDataset(root_path=RESOURCES+'data_v2/',labelme_path=RESOURCES+'data_v2/labelme/',path_to_save_masks=RESOURCES+'data_v2/masks/', img_size=sam_model.image_encoder.img_size,create=False)

time1 = time.time()
dataset = DataLoader(data_v2, batch_size=1, shuffle=True)

time2 = time.time()



# Fine-tuning the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
predictor.sam.to(device)
predictor.sam.train()

# Set up optimizer
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters())

# Set up loss function
loss_fn = torch.nn.MSELoss()
print('START EPOCH')
for epoch in range(2):
    epoch_losses = []
    for i,data in enumerate(dataset, 0):
        original_image_size,input_image_torch, box_torch, gt_binary_mask = data
        input_image_torch, gt_binary_mask = input_image_torch[0], gt_binary_mask[0]
        input_image_torch = input_image_torch.to(device)
        transformed_image = input_image_torch.permute(2,0,1).contiguous()
        transformed_image = transformed_image[None, :, :, :]


        box = transform.apply_boxes_torch(box_torch, original_image_size)

        # # box_torch = torch.as_tensor(box,dtype=torch.float, device=device)

        box_torch = box_torch[None, :]



        #
        # original_image = input_image[0]
        # example = original_image.permute(1, 2, 0)  # Rearrange dimensions for matplotlib

        # cv2.imshow('example',example)

        # input_image, box_torch, gt_binary_mask = input_image.to(device), box_torch.to(device), gt_binary_mask.to(device)

        # input_image = transform.apply_image(original_image)
        # input_image_torch = torch.as_tensor(original_image, device=device)
        # transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        # print(transformed_image.shape)
        input_image = sam_model.preprocess(transformed_image)
        # original_image_size = input_image_torch.shape[:2]
        input_size = tuple(transformed_image.shape[-2:])

        with torch.no_grad():
            image_embedding = sam_model.image_encoder(input_image)
            
            # prompt_bbox = box_torch
            # box_torch = transform.apply_boxes_torch(box_torch, original_image_size)
            # box_torch.to(device)
            # box_torch = box_torch[None,:]
        
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(points=None, boxes=box_torch, masks=None)
        
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        # Postprocessing
        upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)

        # Generate binary mask
        binary_mask = F.normalize(F.threshold(upscaled_masks, 0.0, 0)).to(device)

        # Calculate loss and backpropagate
        loss = loss_fn(binary_mask, gt_binary_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss)
    
    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')
# Save the fine-tuned model
torch.save(sam_model.state_dict(), 'fine_tuned_sam.pth')

time3 = time.time()
print(time3-time2)