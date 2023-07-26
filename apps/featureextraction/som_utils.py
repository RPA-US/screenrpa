from SOM.UiComponent import UiComponent
import numpy as np
import json

def calculate_Xpath(path_json_components, pt,last_element_idx=0, last_group_idx=0):
    '''
    returns:
        xpath: string with the xpath explicit. Example (UI_Component_6/UI_Component_2/UI_Element_1)
        xpath_list: list of UiComponent from the xpath

    param: path_json_components: the path to the json where the components are saved.
    type: str
    param: pt: the point (x,y) to track the Xpath
    type: tuple(int, int)
    param: last_element_idx: recursive parameter to save the last UiElement index (is recursive in similar_uicompo function)
    type: int
    param: last_group_idx: recursive parameter to save the last UiGroup index
    type: int
    
    
    '''
    xpath=''
    xpath_list=[]
    with open(path_json_components,'r') as f:
        components = json.load(f)
    
    components_uicompo = list(map(
        lambda x:UiComponent(int(x['id']),area=int(x['area']),bbox_list=x['bbox'], contain=x['contain'], category=x['category']),
        components)) 
    
    copy_components = list.copy(components_uicompo)

    # First iteration: calculate the components where the point is inside of

    compo = min(list(filter(
        lambda x:x.contain_point(pt),
        copy_components)), 
        key=lambda x:int(x.bbox_area))
    
    id=compo.id
    if compo.category=='UI_Element':
        xpathid = f'UI_Element_{last_element_idx+1}'
        last_element_idx+=1
    else:
        xpathid = f'UI_Group_{last_group_idx+1}'
        last_group_idx+=1

    xpath=f'{xpathid}/{xpath}'
    xpath_list.append(compo)
    
    # Now all the iterative process is made using uicomponents not the point.

    copy_components = list(filter(lambda x:id in x.contain, copy_components))
    
    while copy_components:
        compo = min(copy_components, key=lambda x:int(x.bbox_area))
        id=compo.id
        if compo.category=='UI_Element':
            xpathid = 'UI_Element_{}'.format(last_element_idx+1)
            last_element_idx+=1
        else:
            xpathid = 'UI_Group_{}'.format(last_group_idx+1)
            last_group_idx+=1
        xpath='{}/'.format(xpathid)+xpath
        xpath_list.append(compo)
        copy_components = list(filter(lambda x:id in x.contain, copy_components))
    xpath_list.reverse()
    return xpath_list, xpath, last_group_idx, last_element_idx

def similar_uicomponent(components, xpath_list, store_xpath, last_element_idx=0, last_group_idx=0):
    '''
    This is a recursive function used to recalculate the xpath of a tracking point taking into account components of the screenshot,
    in order to use the components already used in another process or to use new ones.
    It can be used when we have several tracking points in the same screenshot,
    or when we have several screenshots with a lot of similarity where certain components are going to remain.

    returns:
        store_xpath: str: final xpath after recalculating from xpath_list
        last_element_idx: int: index of the last UiElement studied
        last_group_idx: int: index of the last UiGroup studied
        new_components: list(UiComponent): list of the new components obtained


    param: components: list of UiComponents
    type: list(UiComponent)
    param: xpath_list: list of components from the xpath
    type: list(UiComponent)
    param: store_xpath: in the first iteration is empty, in subsequent recursive calls it is the xpath modified with news idx
    type: str
    param: last_element_idx: recursive param to save the last UiElement index
    type: int
    param: last_group_idx: recursive param to save the last UiGroup index
    type: int
    '''
    store_components = list.copy(components)
    biggest_xpath_compo = xpath_list[0]
    similar_compos = list(filter(lambda x:x.bbox_distance(biggest_xpath_compo)>0.9, store_components))
    if not similar_compos:
        new_components=[]
        for compo in xpath_list:
            if compo.category=='UI_Element':
                xpathid = 'UI_Element_{}'.format(last_element_idx+1)
                last_element_idx+=1
                compo.id = last_element_idx
            else:
                xpathid = 'UI_Group_{}'.format(last_group_idx+1)
                last_group_idx+=1
                compo.id = last_group_idx
            store_xpath=store_xpath+'{}/'.format(xpathid)
            new_components.append(compo)
        return store_xpath, last_element_idx,last_group_idx, new_components
    
    else:
        similar_compo = min(similar_compos, key=lambda x:x.bbox_area)
        if similar_compo.category=='UI_Element':
                xpathid = 'UI_Element_{}'.format(similar_compo.id)
        else:
            xpathid = 'UI_Group_{}'.format(similar_compo.id)
        store_xpath=store_xpath+'{}/'.format(xpathid)

        store_components = list(filter(lambda x:x.id in similar_compo.contain, store_components))
        if len(xpath_list)==1:
            return store_xpath, last_element_idx,last_group_idx,[]
        
        new_xpath_list = xpath_list[1:]
        return similar_uicomponent(store_components,new_xpath_list,store_xpath,last_element_idx, last_group_idx)


#### Ejemplo de uso de similar_uicomponent
import cv2
from SOM.detection import resize_by_height
from SOM import ip_draw 

COMPOS_JSON_PATH=''
IMAGE_EXAMPLE_PATH = ''
EXAMPLE_XPATH = ''
IMAGE_JSON_PATH = ''
IMAGE_EXAMPLE_NAME = ''
h_resize=800

img_example = cv2.imread(IMAGE_EXAMPLE_PATH)
img_example = resize_by_height(img_example,h_resize)

dic_sol={}
list_pt=[(894,382),(960,293),(163,184),(195,353)]
colors=[(0,255,0),(255,0,0),(0,0,255),(255,255,0)]
store_components=[]
last_element_idx, last_compo_idx=0,0
for i,pt in enumerate(list_pt):
    xpath_list, xpath, _, _ = calculate_Xpath(COMPOS_JSON_PATH+IMAGE_JSON_PATH,pt)
    store_xpath, last_element_idx,last_compo_idx, new_components = similar_uicomponent(store_components,xpath_list,'',last_element_idx, last_compo_idx)
    store_components.extend(new_components)
    dic_sol[str(i)]=store_xpath

    color = colors[i]
    img_example=ip_draw.draw_bounding_box(img_example, new_components,color=color, show=False, name='merged compo', 
                           write_path=EXAMPLE_XPATH+IMAGE_EXAMPLE_NAME+'_example.jpeg', 
                           wait_key=0)