from audioop import reverse
from pickle import TRUE
import torch
import numpy as np
from tkinter import messagebox

from isegm.inference import clicker, utils
from isegm.inference.predictors import get_predictor
from isegm.utils.vis import draw_with_blend_and_clicks
import cv2
from sklearn.metrics import average_precision_score


class InteractiveController:
    def __init__(self, net, device, predictor_params, update_image_callback, prob_thresh=0.5, roi_size=10, manual_button_size=5):
        self.net = net
        self.prob_thresh = prob_thresh
        self.clicker = clicker.Clicker()
        self.states = []
        self.probs_history = []
        self.object_count = 0
        self._result_mask = None
        self._init_mask = None

        self.image = None
        self.predictor = None
        self.device = device
        self.update_image_callback = update_image_callback
        self.predictor_params = predictor_params
        self.reset_predictor()

        self.roi_size = roi_size
        self.manual_button_size = manual_button_size
        self.tempo = []

        # self.dice_list = []
        self.file_name = str()

    def set_image(self, image):
        self.image = image
        self._result_mask = np.zeros(image.shape[:2], dtype=np.uint16)
        self.object_count = 0
        self.reset_last_object(update_image=False)
        self.update_image_callback(reset_canvas=True)

    def set_mask(self, mask):
        if self.image.shape[:2] != mask.shape[:2]:
            messagebox.showwarning("Warning", "A segmentation mask must have the same sizes as the current image!")
            return

        if len(self.probs_history) > 0:
            self.reset_last_object()

        self._init_mask = mask.astype(np.float32)
        self.probs_history.append((np.zeros_like(self._init_mask), self._init_mask))
        # self.probs_history.append((self._init_mask, self._init_mask))
        self._init_mask = torch.tensor(self._init_mask, device=self.device).unsqueeze(0).unsqueeze(0)
        self.clicker.click_indx_offset = 1

    def add_click(self, x, y, is_positive):
        self.states.append({
            'clicker': self.clicker.get_state(),
            'predictor': self.predictor.get_states()
        })
        

        self.current_click_state = x, y, is_positive

        click = clicker.Click(is_positive=is_positive, coords=(y, x))
        self.clicker.add_click(click)

        pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)
        if self._init_mask is not None and len(self.clicker) == 1:
            pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)

        torch.cuda.empty_cache()

        if self.probs_history:
            self.probs_history.append((self.probs_history[-1][0], pred))
        else:
            self.probs_history.append((np.zeros_like(pred), pred))

        self.update_image_callback()

    def undo_click(self):
        if not self.states:
            return

        prev_state = self.states.pop()
        self.clicker.set_state(prev_state['clicker'])
        self.predictor.set_states(prev_state['predictor'])                          
        self.probs_history.pop()
        # self.dice_list.pop()
        if not self.probs_history:
            self.reset_init_mask()
        self.update_image_callback()

    def partially_finish_object(self):
        object_prob = self.current_object_prob
        if object_prob is None:
            return

        self.probs_history.append((object_prob, np.zeros_like(object_prob)))
        self.states.append(self.states[-1])

        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        self.update_image_callback()

    def finish_object(self):
        if self.current_object_prob is None:
            return

        self._result_mask = self.result_mask
        self.object_count += 1
        self.reset_last_object()

    def reset_last_object(self, update_image=True):
        self.states = []
        self.probs_history = []
        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        # self.dice_list = []
        if update_image:
            self.update_image_callback()

    def reset_predictor(self, predictor_params=None):
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.net, device=self.device,
                                       **self.predictor_params)
        if self.image is not None:
            self.predictor.set_input_image(self.image)

    def reset_init_mask(self):
        self._init_mask = None
        # self.dice_list = []
        self.clicker.click_indx_offset = 0

    @property
    def current_object_prob(self):
        if self.probs_history:
            current_prob_total, current_prob_additive = self.probs_history[-1]

            if len(self.probs_history) > 1:
                current_prob_total_previous, current_prob_additive_previous = self.probs_history[-2]

                current_prob_additive_roi = current_prob_additive_previous.copy()

                x, y, is_positive = self.current_click_state

                zero_mask = np.zeros_like(current_prob_additive)
                zero_mask_3_chanel = 255 * zero_mask[:, :, np.newaxis].repeat(3, axis=2).astype(np.uint8)
                cirle_image = cv2.circle(zero_mask_3_chanel, 
                                (x,y), 
                                self.roi_size, 
                                (255,255,255),
                                -1)
                cirle_mask = (cirle_image[:,:,0]/255).astype(np.uint8)
                revert_cirle_mask = (cirle_mask < 1).astype(np.uint8)

                if self.roi_size != 0:
                    if is_positive:
                        
                        outputted_prob_additive_roi = np.maximum(current_prob_additive_roi,current_prob_additive)
                        outputted_prob_additive_roi = outputted_prob_additive_roi * cirle_mask + current_prob_additive_roi * revert_cirle_mask
                        self.probs_history[-1] = current_prob_total, outputted_prob_additive_roi
                        return outputted_prob_additive_roi
                    else:
                        
                        outputted_prob_additive_roi = np.minimum(current_prob_additive_roi,current_prob_additive)
                        outputted_prob_additive_roi = outputted_prob_additive_roi * cirle_mask + current_prob_additive_roi * revert_cirle_mask
                        self.probs_history[-1] = current_prob_total, outputted_prob_additive_roi
                        return outputted_prob_additive_roi
                else:
                    return np.maximum(current_prob_total, current_prob_additive)

            else:
                return np.maximum(current_prob_total, current_prob_additive)
        else:
            return None

    @property
    def is_incomplete_mask(self):
        return len(self.probs_history) > 0

    @property
    def result_mask(self):
        result_mask = self._result_mask.copy()
        if self.probs_history:
            result_mask[self.current_object_prob > self.prob_thresh] = self.object_count + 1
        return result_mask


    def lesion_pixels_number(self):
        if len(self.probs_history) == 0:
            return 0
        result = self.result_mask
        return np.sum(result)
    
    def lesion_number(self):
        if len(self.probs_history) == 0:
            return 0,0,0
        new_mask = np.ones((1024,1024,3))
        mask = self.result_mask
        mask = mask * 255
        new_mask[:, :,0] = mask
        new_mask[:, :,1] = mask
        new_mask[:, :,2] = mask
        new_mask=new_mask.astype(np.uint8)

        gray = cv2.cvtColor(new_mask, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        invert = 255 - thresh
        output = cv2.connectedComponentsWithStats(invert, 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        list_size = []
        if len(stats)>=2:
            for i in range(0,len(stats)):
                list_size.append(stats[i,cv2.CC_STAT_AREA])
            list_size.sort()
            return numLabels-1,list_size[0], list_size[-2]
        else:
            return numLabels-1, 0, 0
    

    def get_visualization(self, alpha_blend, click_radius):
        if self.image is None:
            return None

        results_mask_for_vis = self.result_mask
        vis = draw_with_blend_and_clicks(self.image, mask=results_mask_for_vis, alpha=alpha_blend,
                                         clicks_list=self.clicker.clicks_list, radius=click_radius)

        return vis

    def update_roi_size(self, value):
        self.roi_size = value

    def update_manual_button_size(self, value):
        self.manual_button_size = value
    
    def update_filename(self, filename):
        self.file_name = filename
        # self.dice_list = []
        

