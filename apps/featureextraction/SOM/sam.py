import os
from os.path import join as pjoin
import json
import time
import cv2
import torch
from . import ip_draw as draw
from art import tprint
from tqdm import tqdm
from .segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from .fastSAM import FastSAM
from .ip_draw import draw_bounding_box
from .UiComponent import UiComponent #QUIT

#AUXILIAR FUNCTIONS#######################
def nesting_compos(uicompos):
    '''
    params uicompos: list<Compo> 
    returns:
        --> compos_json: a json containing a list of Compos
        --> uicompos: Compo objects with 'contain' and 'category' updated
    '''
    n = len(uicompos)
    list_compo_dict=[]
    for i in range(n):
        compo_a = uicompos[i]
        for j in range(i+1,n):
            compo_b = uicompos[j]
            rel = compo_a.compo_relation(compo_b)
            if rel==-1:
                compo_b.contain.append(compo_a.id)
                compo_b.category='UI_Group'
            elif rel==1:
                compo_a.contain.append(compo_b.id)
                compo_a.category='UI_Group'
        list_compo_dict.append(compo_a.to_dict())
    compos_json = json.dumps(list_compo_dict,indent=1)
    return compos_json,uicompos

def resize_by_height(org, resize_height):
    w_h_ratio = org.shape[1] / org.shape[0]
    resize_w = resize_height * w_h_ratio
    re = cv2.resize(org, (int(resize_w), int(resize_height)))
    return re

def get_compos_mask_json(masks, image_shape):
    '''
    get components from masks and json with sam format
    returns:
    ---> arrays_dict: a dict containg list of each non-Json-serializable object of mask (segmentation, crop_box)
    ---> mask_json: a json format of the mask including the id of the Compo 
    ---> sorted_compos: a list of Compo elements sorted by bbox area 
    '''
    def bbox_area(bbox):
        return bbox[3]*bbox[2]
    sorted_compos=[]
    sorted_masks=list(sorted(masks, key=lambda mask: bbox_area(mask['bbox']), reverse=True))
    arrays_dict={
        'segmentation':[],
        'point_coords':[],
        'crop_box':[]
    }
    for i,mask in enumerate(sorted_masks): #TODO maybe here we can include a filter of areas
        #if bbox_area(mask)<uied_params['min-ele-area']: break
        compo=UiComponent(i,bbox_list=mask['bbox'],area=mask['area'], contain=[])
        compo.set_data(segmentation=mask['segmentation'],crop_box=mask['crop_box'], image_shape=image_shape)
        
        sorted_compos.append(compo)

        for n in ['segmentation','crop_box']:
            arrays_dict[n].append(mask[n])    
        
        mask['id']=compo.id
        mask.pop('segmentation')
        mask.pop('crop_box')

    
    mask_json = json.dumps(sorted_masks,indent=1)
    return arrays_dict,mask_json, sorted_compos 

def get_sam_gui_components_crops(param_img_root,image_names ,path_to_save_bordered_images,img_index,checkpoint_path,sam_type="sam",checkpoint='l'):
    '''
    Analyzes an image and extracts its UI components

    :param param_img_root: Path to the image
    :type param_img_root: str
    :param image_names: Names of the images in the path
    :type image_names: list
    :param path_to_save_bordered_images: Path to save the image along with the components detected
    :type path_to_save_bordered_images: str
    :param img_index: Index of the image we want to analyze in images_names
    :type img_index: int
    :param checkpoint: sam model checkpoint to use
    :type checkpoint: str in 'l','h','b'.
    :return: Crops and text inside components
    :rtype: Tuple
    '''

    time0=time.time()


    resize_height = 800
    input_img_path = pjoin(param_img_root, image_names[img_index])

    name = input_img_path.split('/')[-1][:-4] if '/' in input_img_path else input_img_path.split('\\')[-1][:-4]
    ip_root = pjoin(path_to_save_bordered_images, "ip")
    if not os.path.exists(ip_root):
        os.mkdir(ip_root)
    
    ##################################
    ##PREPROCESADO
    image = cv2.imread(input_img_path, cv2.COLOR_BGR2RGB)
    image = resize_by_height(image, resize_height)
    image_copy = image.copy()
    image_shape = image.shape

    time1=time.time()

    sam_type="fast-sam"
    checkpoint="x"
    match(sam_type):
        case "fast-sam":
            masks = get_fast_sam_masks(checkpoint, checkpoint_path, image_copy)
        case "sam":
            masks = get_sam_masks(sam_model_registry, checkpoint, checkpoint_path, image_copy)

    time2=time.time()
    '''
    masks contains the .generate return:
        - list of mask, each record is a dict(str, any) containg:
            - segmentation (dict(str, any) or np.ndarray):
                The mask. If output_mode='binary_mask', is an array of shape HW.
                Otherwise, is a dictionary containing the RLE
            - bbox (list(float)): The box around the mask, in XYWH format.
            - area (int): The area in pixels of the mask.
            - predicted_iou (float): The model's own prediction of the mask's
                    quality. This is filtered by the pred_iou_thresh parameter.
            - point_coords (list(list(float))): The point coordinates input
                    to the model to generate this mask.
            - stability_score (float): A measure of the mask's quality. This
                    is filtered on using the stability_score_thresh parameter.
            - crop_box (list(float)): The crop of the image used to generate
                    the mask, given in XYWH format.   
    '''
    arrays_dict,mask_json,uicompos = get_compos_mask_json(masks, image_shape)
    time3=time.time()

    # *** Step 4 ** nesting inspection: check if big compos have nesting element
    compos_json,uicompos= nesting_compos(uicompos)
    time4=time.time()
    # *** Step 5 *** save detection result
    draw_bounding_box(image_copy, uicompos, show=False, name='merged compo', 
                           write_path=pjoin(ip_root, name + '.jpg'), 
                           wait_key=0)
    
    time5=time.time()

    dict_times={
        'preprocess_image':time1-time0,
        'sam_extractor':time2-time1,
        'getting_compos_and_mask_json':time3-time2,
        'nesting_compos':time4-time3,
        'drawing':time5-time4
    }
    # ##########################
    # RESULTS
    # ##########################

    clips = [compo.compo_clipping(image_copy) for compo in uicompos]

    return clips, uicompos, mask_json, compos_json, arrays_dict, dict_times
  
def get_sam_masks(sam_model_registry, checkpoint, checkpoint_path, image_copy):

    ### SAM MODEL ####
    match checkpoint:
        case "h":
            sam_checkpoint = "sam_vit_h_4b8939.pth"
            model_type = "vit_h"
        case "b":
            sam_checkpoint = "sam_vit_b_01ec64.pth"
            model_type = "vit_b"
        case "l":
            sam_checkpoint = "sam_vit_l_0b3195.pth"
            model_type = "vit_l"
        case _:
            raise Exception("You select a type of sam's checkpoint that doesnt exists")

    # torch.cuda.set_per_process_memory_fraction(fraction=0.55, device=0)
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path+sam_checkpoint)
    # device = "cuda:0"
    # sam.to(device=device)

    ### GENERATE MASK ###
    # mask_generator = SamAutomaticMaskGenerator(sam)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        pred_iou_thresh=0.95,
    )
    masks = mask_generator.generate(image_copy)

    return masks

def get_fast_sam_masks(checkpoint, checkpoint_path, image_copy):

    ### FAST-SAM MODEL ###
    match checkpoint:
        case "s":
            sam_checkpoint = "FastSAM-s.pt"
        case "x":
            sam_checkpoint = "FastSAM-x.pt"
        case _:
            raise Exception("You select a type of fast sam's checkpoint that doesnt exists")

    fast_sam = FastSAM(checkpoint_path+sam_checkpoint)

    # device = "cuda:0"
    # fast_sam(device=device)

    ### GENERATE MASK ###

    masks = fast_sam.predict(source=image_copy)[0]

    # TODO: parse the masks object to be in line with the masks returned by sam

    return masks
