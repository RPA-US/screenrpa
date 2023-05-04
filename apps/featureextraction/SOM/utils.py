from SOM.UiComponent import UiComponent
import numpy as np
import json
import pandas as pd


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