import numpy as np

class UiComponent:
    def __init__(self,id,area,bbox_list,category='UI_Element', contain=[]):
        self.id=id
        # self.mask=segmentation
        self.area=area
        self.bbox=bbox_list #List in format XYWH
        self.bbox_area=bbox_list[2]*bbox_list[3]
        # self.crop_box=crop_box
        self.contain=contain
        self.category=category
        # self.image_shape=image_shape

    def set_data(self, segmentation, crop_box,image_shape):
        self.mask=segmentation
        self.crop_box=crop_box
        self.image_shape=image_shape


    def put_bbox(self):
        x,y,w,h = self.bbox
        return [x,y,x+w,y+h]
    
    def bbox_distance(self,compo2):
        '''
        IoU = intersection(bbox1, bbox2) / union(bbox1, bbox2)
        Entre 0 y 1
        '''
        bbox1 = self.bbox
        bbox2 = compo2.bbox

        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
        return iou

        
    def to_dict(self):
        return {
            'id':self.id,
            'image_shape':self.image_shape,
            'area':self.area,
            'bbox':self.bbox,
            'bbox_area':self.bbox_area,
            'category':self.category,
            'contain':self.contain
        }
    
    def compo_clipping(self, img, pad=0, show=False):
        column_min, row_min, column_max, row_max = self.put_bbox()
        column_min = max(column_min - pad, 0)
        column_max = min(column_max + pad, img.shape[1])
        row_min = max(row_min - pad, 0)
        row_max = min(row_max + pad, img.shape[0])
        clip = img[row_min:row_max, column_min:column_max]
        return clip
    
    def contain_point(self, pt):
        x,y = pt
        x0,y0,x1,y1 = self.put_bbox()
        if x0<=x and x<=x1 and y0<=x and y<=y1:
            return True
        else:
            return False
        
    # def contain_bbox(self,bbox_b,bias=(0,0)):
    #     col_min_a, row_min_a, w_a, h_a = self.bbox
    #     col_max_a = col_min_a + w_a
    #     row_max_a = row_min_a + h_a

    #     col_min_b, row_min_b, w_b, h_b = bbox_b
    #     col_max_b = col_min_b + w_b
    #     row_max_b = row_min_b + h_b

    #     bias_col, bias_row = bias
    #     # get the intersected area
    #     col_min_s = max(col_min_a - bias_col, col_min_b - bias_col)
    #     row_min_s = max(row_min_a - bias_row, row_min_b - bias_row)
    #     col_max_s = min(col_max_a + bias_col, col_max_b + bias_col)
    #     row_max_s = min(row_max_a + bias_row, row_max_b + bias_row)
    #     w = np.maximum(0, col_max_s - col_min_s)
    #     h = np.maximum(0, row_max_s - row_min_s)
    #     inter = w * h
    #     area_a = (col_max_a - col_min_a) * (row_max_a - row_min_a)
    #     area_b = (col_max_b - col_min_b) * (row_max_b - row_min_b)
    #     iou = inter / (area_a + area_b - inter)
    #     ioa = inter / self.bbox_area
    #     bbox_b_area = bbox_b[2]*bbox_b[3]
    #     iob = inter / bbox_b_area

    #     if iou == 0 and ioa == 0 and iob == 0:
    #         return False
    #     # contained by b
    #     if ioa >= 1:
    #         return False
    #     # contains b
    #     if iob >= 1:
    #         return True
    #     # not intersected with each other
    #     # intersected
    #     if iou >= 0.02 or iob > 0.2 or ioa > 0.2:
    #         return False
    #     # if iou == 0:
    #     return False

        
    def compo_relation(self,compo_b, bias=(0,0)):
        '''
        Calculate the relation between two rectangles by nms
       :return: -1 : a in b
         0  : a, b are not intersected
         1  : b in a
         2  : a, b are intersected
       '''
        bbox_b = compo_b

        col_min_a, row_min_a, w_a, h_a = self.bbox
        col_max_a = col_min_a + w_a
        row_max_a = row_min_a + h_a

        col_min_b, row_min_b, w_b, h_b = bbox_b.bbox
        col_max_b = col_min_b + w_b
        row_max_b = row_min_b + h_b

        bias_col, bias_row = bias
        # get the intersected area
        col_min_s = max(col_min_a - bias_col, col_min_b - bias_col)
        row_min_s = max(row_min_a - bias_row, row_min_b - bias_row)
        col_max_s = min(col_max_a + bias_col, col_max_b + bias_col)
        row_max_s = min(row_max_a + bias_row, row_max_b + bias_row)
        w = np.maximum(0, col_max_s - col_min_s)
        h = np.maximum(0, row_max_s - row_min_s)
        inter = w * h
        area_a = (col_max_a - col_min_a) * (row_max_a - row_min_a)
        area_b = (col_max_b - col_min_b) * (row_max_b - row_min_b)
        iou = inter / (area_a + area_b - inter)
        ioa = inter / self.bbox_area
        iob = inter / bbox_b.bbox_area

        if iou == 0 and ioa == 0 and iob == 0:
            return 0
        # contained by b
        if ioa >= 1:
            return -1
        # contains b
        if iob >= 1:
            return 1
        # not intersected with each other
        # intersected
        if iou >= 0.02 or iob > 0.2 or ioa > 0.2:
            return 2
        # if iou == 0:
        return 0