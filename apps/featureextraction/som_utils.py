from SOM.UiComponent import UiComponent
import numpy as np
import json

def calculate_Xpath(path_json_components, pt,last_element_idx=0, last_compo_idx=0):
    '''
    returns:
        xpath: string with the xpath explicit. Example (UI_Component_6/UI_Component_2/UI_Element_1)
        xpath_list: list of UiComponent from the xpath

    param: path_json_components: the path to the json where the components are saved.
    type: str
    param: pt: the point (x,y) to track the Xpath
    type: tuple(int, int)
    param: last_element_idx: recursive parameter to save the last element index (is recursive in similar_uicompo function)
    type: int
    param: last_compo_idx: recursive parameter to save the last component index
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
        xpathid = f'UI_Component_{last_compo_idx+1}'
        last_compo_idx+=1

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
            xpathid = 'UI_Component_{}'.format(last_compo_idx+1)
            last_compo_idx+=1
        xpath='{}/'.format(xpathid)+xpath
        xpath_list.append(compo)
        copy_components = list(filter(lambda x:id in x.contain, copy_components))
    xpath_list.reverse()
    return xpath_list, xpath, last_compo_idx, last_element_idx

def similar_uicomponent(components, xpath_list, store_xpath, last_element_idx=0, last_compo_idx=0):
    '''
    returns

    param: components: list of UiComponents
    type: list(UiComponent)
    param: xpath_list: list of components from the xpath
    type: list(UiComponent)
    param: store_xpath: in the first iteration is empty, in subsequent recursive calls it is the xpath modified with news idx
    type: str
    param: last_element_idx: recursive param to save the last element index
    type: int
    param: last_compo_idx: recursive param to save the last component index
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
                xpathid = 'UI_Component_{}'.format(last_compo_idx+1)
                last_compo_idx+=1
                compo.id = last_compo_idx
            store_xpath=store_xpath+'{}/'.format(xpathid)
            new_components.append(compo)
        return store_xpath, last_element_idx,last_compo_idx, new_components
    
    else:
        similar_compo = min(similar_compos, key=lambda x:x.bbox_area)
        if similar_compo.category=='UI_Element':
                xpathid = 'UI_Element_{}'.format(similar_compo.id)
        else:
            xpathid = 'UI_Component_{}'.format(similar_compo.id)
        store_xpath=store_xpath+'{}/'.format(xpathid)

        store_components = list(filter(lambda x:x.id in similar_compo.contain, store_components))
        if len(xpath_list)==1:
            return store_xpath, last_element_idx,last_compo_idx,[]
        
        new_xpath_list = xpath_list[1:]
        return similar_uicomponent(store_components,new_xpath_list,store_xpath,last_element_idx, last_compo_idx)


