from django.test import TestCase
# from SOM import detection
import json
import cv2
import numpy as np
import time
import os 
import torch
from SOM.UiComponent import UiComponent
# import som_utils
from apps.featureextraction.SOM import utils
from SOM import ip_draw 
import random

IMG_FOLDER='C:/Users/pablo/OneDrive/Escritorio/new_rim/screenrpa/resources/sc_0_size25_Balanced/'
# IMG_FOLDER='/rim/resources/gaze/sc_0_size25_Balanced/'
# PATH_TO_SAVE='/rim/resources/output/'
PATH_TO_SAVE = IMG_FOLDER
COMPOS_JSON_PATH=PATH_TO_SAVE+'compos_json/'
MASK_JSON_PATH=PATH_TO_SAVE+'mask_json/'
TIME_JSON_PATH=PATH_TO_SAVE+'time_json/'
EXAMPLE_XPATH=PATH_TO_SAVE+'example_xpath/'
checkpoint='l'

compos_json_path = PATH_TO_SAVE+'compos_json.json'
mask_json_path = PATH_TO_SAVE+'mask_json.json'
img_name_base = '_img'
example ='10'+img_name_base
img_json = example+'_l.json'
img_example=IMG_FOLDER+example+'.png'
h_resize=800

###PRUEBAS
def resize_by_height(org, resize_height):
    w_h_ratio = org.shape[1] / org.shape[0]
    resize_w = resize_height * w_h_ratio
    re = cv2.resize(org, (int(resize_w), int(resize_height)))
    return re



img_example = cv2.imread(img_example)
img_example = resize_by_height(img_example,h_resize)

dic_sol={}
list_pt=[(822,214),(58,526),(949,109),(1179,148)]
colors=[(0,255,0),(255,0,0),(0,0,255),(255,255,0)]
store_components=[]
last_element_idx, last_compo_idx=0,0
for i,pt in enumerate(list_pt):
    time0=time.time()
    xpath_list, xpath, _, _ = utils.calculate_Xpath(COMPOS_JSON_PATH+img_json,pt)
    store_xpath, last_element_idx,last_compo_idx, new_components = utils.similar_uicomponent(store_components,xpath_list,'',last_element_idx, last_compo_idx)
    store_components.extend(new_components)
    dic_sol[str(i)]=store_xpath
    # color = (random.randrange(0,256),random.randrange(0,256),random.randrange(0,256))
    color = colors[i]
    img_example=ip_draw.draw_bounding_box(img_example, new_components,color=color, show=False, name='merged compo', 
                           write_path=EXAMPLE_XPATH+example+'_example.jpeg', 
                           wait_key=0)
    time1=time.time()
    print(time1-time0)

print(dic_sol)
print(store_components)


with open(COMPOS_JSON_PATH+img_json,'r') as f:
        components = json.load(f)
    
uicompos = list(map(
    lambda x:UiComponent(int(x['id']),area=int(x['area']),bbox_list=x['bbox']),
    components)) 

    

print('Xpath: {}\nLast_compo_idx: {}\nLast_ele_idx: {}'.format(xpath,last_compo_idx,last_element_idx))





# Create your tests here.
compos_list=[]
for compo in uicompos:
    bbox = compo.bbox
    box=[bbox.col_min, bbox.row_min, bbox.width, bbox.height]
    compo_dict={
        "id":compo.id,
        "bbox":box,
        "category":compo.category,
        "contain":compo.contain,
        "area":compo.area
    }
    compos_list.append(compo_dict)

compo_json = json.dumps(compos_list,indent=2)
print(compo_json)



grey = cv2.imread(IMG_FOLDER+img_example, cv2.COLOR_BGR2GRAY)
gauss = cv2.GaussianBlur(grey,(3,3),10)
canny = cv2.Canny(gauss,10,30)
reversed = cv2.bitwise_not(canny)
cv2.imwrite(PATH_TO_SAVE+'canny_reversed.png',reversed)

    
