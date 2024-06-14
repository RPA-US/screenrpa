from SOM.UiComponent import UiComponent
import numpy as np
import os
import json
import pandas as pd
import random
from SOM import ip_draw
from shapely.geometry import Polygon
import cv2

def save_screenshot_modified(path_to_original_screenshit,path_to_save_modified_screenshit, new_components,random_color=True, color=None,show=False, separator='/'):
    '''
    new_components are the same as the return of similar_uicomponent
    if random_color = False you should pass a color
    '''
    name = path_to_original_screenshit.split(separator)[-1]
    original_img = cv2.imread(path_to_original_screenshit)
    if random_color:
        color = (random.randrange(0,256),random.randrange(0,256),random.randrange(0,256))
    else:
        color = color
    _=ip_draw.draw_bounding_box(original_img, new_components,color=color, show=show, name='merged compo', 
                           write_path=path_to_save_modified_screenshit+name, 
                           wait_key=0)



def add_xpath_to_dataset(store_components,last_element_idx, last_group_idx,path_to_screenshit_json, pt):
    '''
    params: 
        @store_components: lista de las UiComponents almacenadas
        @last_element_idx: último índice del UiElement que ha sido almacenado
        @last_group_idx: úlitmo índiice del UiGroup que ha sido almacenado
        @path_to_screenshit_json: path hacia el json de la screenshit a estudiar
        @pt: puntos en coordenadas (x,y) de la pantalla

    returns:
        @store_components: lista de las UiComponents almacenadas
        @last_element_idx: último índice del UiElement que ha sido almacenado
        @last_group_idx: úlitmo índiice del UiGroup que ha sido almacenado
        @path_to_screenshit_json: path hacia el json de la screenshit a estudiar
        @resulting_xpath: la xpath resultante
    '''
    xpath_list, xpath, _, _ = calculate_Xpath(path_to_screenshit_json,pt)
    resulting_xpath, last_element_idx,last_group_idx, new_components = similar_uicomponent(store_components,xpath_list,'',last_element_idx, last_group_idx)
    store_components.extend(new_components)

    return store_components,last_element_idx, last_group_idx,resulting_xpath


def calculate_Xpath(path_json_components, pt,last_element_idx=0, last_group_idx=0):
    '''
    params:
        @path_json_components: path hacia las componentes json a estudiar
        @pt: punto de la pantalla en coordenadas (x,y)
        @last_element_idx: último índice del UiElement que ha sido almacenado
        @last_group_idx: úlitmo índiice del UiGroup que ha sido almacenado
    
    returns:
        @xpath_list: lista de las UiComponents que representa la xpath (es decir la xpath en forma de lista con las UiComponents)
        @xpath
        @last_element_idx: último índice del UiElement que ha sido almacenado
        @last_group_idx: úlitmo índiice del UiGroup que ha sido almacenado
    '''

    xpath=''
    xpath_list=[]
    with open(path_json_components,'r') as f:
        components = json.load(f)
    
    components_uicompo = list(map(
        lambda x:UiComponent(int(x['id']),area=int(x['area']),bbox_list=x['bbox'], contain=x['contain'], category=x['category']),
        components)) 
    copy_components = list.copy(components_uicompo)

    compo = min(list(filter(
        lambda x:x.contain_point(pt),
        copy_components)), 
        key=lambda x:int(x.bbox_area))
    
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

def similar_uicomponent(components, xpath_list, store_xpath, last_element_idx, last_group_idx, threshold=0.9):
    '''
    Es un algoritmo recursivo que va comparando las componentes de una xpath desde las más grandes a las pequeñas. 
    De forma que en el primer momento que una no coincida, todo el restante se considera como nuevas componentes y "se añaden al árbol".
    params:
        @components: componentes ya almacenadas en el sistema
        @xpath_list: lista de las UiComponents que representa la xpath (es decir la xpath en forma de lista con las UiComponents)
        @store_xpath: xpath que se va almacenando
        @last_element_idx: último índice del UiElement que ha sido almacenado
        @last_group_idx: úlitmo índiice del UiGroup que ha sido almacenado
        @threshold: umbral según se considera si dos bbox son iguales (se usa bbox_distance que se basa en el índice de Jaccard que mide el cociente entre el área
                                                                        de la intersección y el área de la unión)
    
    returns:
        caso base:
            - Si no hay más componentes similares:
                @store_xpath: xpath resultante
                @last_element_idx
                @last_group_idx
                @new_components: las componentes nuevas que no estaban anteriormente.
            - Si len(xpath_list)=1:
                @store_xpath: xpath resultante
                @last_element_idx
                @last_group_idx
                @new_components:[]
        si no:
            llamada recursiva
    '''
    
    store_components = list.copy(components)
    biggest_xpath_compo = xpath_list[0]
    similar_compos = list(filter(lambda x:x.bbox_distance(biggest_xpath_compo)>threshold, store_components))
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

#########################################
########## COMPONENT RETRIEVAL ##########
#########################################

def get_som_json_from_acts(prev_act, next_act, scenario_results_path, special_colnames) -> dict:
    log = pd.read_csv(os.path.join(scenario_results_path, "pd_log.csv")) 

    # Find two subsequent rows such that the fisrt 'Activity' is prev_act and the second 'Activity' is next_act
    prev_act_idx = log[log[special_colnames['Activity']]==prev_act].index
    next_act_idx = log[log[special_colnames['Activity']]==next_act].index

    if len(prev_act_idx)==0 or len(next_act_idx)==0:
        return None

    # Find two consequent numbers in prev_act_idx and next_act_idx
    img_name = None
    for idx in prev_act_idx:
        if idx+1 in next_act_idx and action==0:
            img_name = log.loc[idx, special_colnames['Screenshot']]
            break
        elif action>0:
            action-=1
    
    if not img_name:
        return None

    if not os.path.exists(os.path.join(scenario_results_path, "components_json")):
        raise Exception("No UI Elm. Det. Phase Conducted")

    return json.load(open(os.path.join(scenario_results_path, "components_json", img_name, ".json")))

def get_uicompo_from_id(prev_act, next_act, ui_compo_id, scenario_results_path, special_colnames, action=0) -> dict:
    """
    Recovers a component from the pd log based on the id of the component.
    Uses prevAct and nextAct to determine the imagen to analyze, which should correspond to the activity before the decision point.

    action determines which instance in order should be considered.

    params:
        @prev_act: previous activity
        @next_act: next activity
        @ui_compo_id: id of the component
        @scenario_results_path: path to the scenario results
        @action: instance of the component to recover
    returns:
        @uicompo: the component
    """
    som_json = get_som_json_from_acts(prev_act, next_act, scenario_results_path, special_colnames)

    uicompo_json = min(list(filter(
        lambda x: x["id"]==ui_compo_id and x["class"]!="Text",
        som_json["compos"]
    )), key=lambda x: Polygon(x["points"]))

    if not uicompo_json:
        return None

    return uicompo_json

def get_uicompo_from_centroid(prev_act, next_act, ui_compo_centroid, scenario_results_path, special_colnames, action=0) -> dict:
    """
    Recovers a component from the pd log based on the id of the component.
    Uses prevAct and nextAct to determine the imagen to analyze, which should correspond to the activity before the decision point.

    action determines which instance in order should be considered.

    params:
        @prev_act: previous activity
        @next_act: next activity
        @ui_compo_centroid: centroid of the component
        @scenario_results_path: path to the scenario results
        @action: instance of the component to recover
    returns:
        @uicompo: the component
    """
    som_json = get_som_json_from_acts(prev_act, next_act, scenario_results_path, special_colnames)

    uicompo_json = min(list(filter(
        lambda x: Polygon(x["points"]).contains(ui_compo_centroid) and x["class"]!="Text",
        som_json["compos"]
    )), key=lambda x: Polygon(x["points"]))

    if not uicompo_json:
        return None

    return uicompo_json