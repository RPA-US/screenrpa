from django.test import TestCase
from SOM import detection
import json
import cv2
import numpy as np
import time

IMG_FOLDER='/rim/resources/gaze/gaze1/'
PATH_TO_SAVE='/rim/resources/output/'
COMPOS_JSON_PATH=PATH_TO_SAVE+'compos_json/'
MASK_JSON_PATH=PATH_TO_SAVE+'mask_json/'
TIME_JSON_PATH=PATH_TO_SAVE+'time_json/'
checkpoint='l'

# compos_json_path = PATH_TO_SAVE+'compos_json.json'
# mask_json_path = PATH_TO_SAVE+'mask_json.json'
img_name_base = 'screenshot00'
image_names=[]
for c in ['l']:
    checkpoint=c
    for i in range(10,12):
        image_names.append(img_name_base+str(i)+'.JPEG')

    for i,name in enumerate(image_names):
        name = name.split('.')[0]
        recortes_path=PATH_TO_SAVE+'compos_npy/'+name+'.npy'
        path_to_save_mask_npy=PATH_TO_SAVE+'mask_elements_npy/'+name
        clips, uicompos, mask_json, compos_json, arrays_dict,dict_times = detection.get_sam_gui_components_crops(param_img_root=IMG_FOLDER, path_to_save_bordered_images=PATH_TO_SAVE, image_names=image_names, img_index=i, checkpoint=checkpoint)

        time0 = time.time()
        with open(COMPOS_JSON_PATH+name+'_'+checkpoint+'.json','w') as f:
            f.write(compos_json)

        with open(MASK_JSON_PATH+name+'_'+checkpoint+'.json','w') as f:
            f.write(mask_json)

        compos_aux = np.array(clips)
        np.save(recortes_path,compos_aux)

        path=path_to_save_mask_npy
        for n in ['segmentation','crop_box']:
            path_element = path+'_'+n+'_'+checkpoint+'.npy'
            aux = np.array(arrays_dict[n])
            np.save(path_element,aux)
        time1=time.time()
        
        dict_times['saving_results']=time1-time0

        times_json = json.dumps(dict_times)
        with open(TIME_JSON_PATH+name+'_'+checkpoint+'.json','w') as f:
            f.write(times_json)

        print('There are {} components'.format(len(uicompos)))


# Create your tests here.
# compos_list=[]
# for compo in uicompos:
#     bbox = compo.bbox
#     box=[bbox.col_min, bbox.row_min, bbox.width, bbox.height]
#     compo_dict={
#         "id":compo.id,
#         "bbox":box,
#         "category":compo.category,
#         "contain":compo.contain,
#         "area":compo.area
#     }
#     compos_list.append(compo_dict)

# compo_json = json.dumps(compos_list,indent=2)
# print(compo_json)



# grey = cv2.imread(IMG_FOLDER+img_example, cv2.COLOR_BGR2GRAY)
# gauss = cv2.GaussianBlur(grey,(3,3),10)
# canny = cv2.Canny(gauss,10,30)
# reversed = cv2.bitwise_not(canny)
# cv2.imwrite(PATH_TO_SAVE+'canny_reversed.png',reversed)

    
