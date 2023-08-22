from time import time

import numpy as np
import torch
import copy
from isegm.inference import utils

from isegm.inference.clicker import Clicker
import cv2
try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm

device = torch.device('cuda:0')

def evaluate_dataset(dataset, predictor, **kwargs):
    all_ious = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)

        _, sample_ious, _ = evaluate_sample_dice(sample.image, sample.gt_mask, predictor,
                                            sample_id=index, **kwargs)
        all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time


def evaluate_sample(image, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []

    with torch.no_grad():
        predictor.set_input_image(image)

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr

            if callback is not None:
                callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

            iou = utils.get_iou(gt_mask, pred_mask)
            ious_list.append(iou)

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break

        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs


def evaluate_sample_dice(image, gt_mask, predictor, max_dice_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=30,
                    sample_id=None, callback=None):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    dice_list = []

    with torch.no_grad():
        predictor.set_input_image(image)

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr

            if callback is not None:
                callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

            dice = utils.get_dice(gt_mask, pred_mask)
            dice_list.append(dice)

            if dice >= max_dice_thr and click_indx + 1 >= min_clicks:
                break

        return clicker.clicks_list, np.array(dice_list, dtype=np.float32), pred_probs

def evaluate_dataset_with_auto(dataset, predictor, **kwargs):
    all_ious = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)

        for object_id in sample.objects_ids:
            _, sample_ious, _ = evaluate_sample_with_auto(sample.image, sample.gt_mask, sample.auto_masks, predictor,
                                                sample_id=index, **kwargs)
        all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time

def evaluate_sample_with_auto(image, gt_mask, auto_masks, predictor, max_dice_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []

    _init_mask = copy.deepcopy(auto_masks).astype(np.float32)
    _init_mask = torch.tensor(_init_mask, device=device).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        predictor.set_input_image(image)
        
        for click_indx in range(max_clicks):
            ## make the next interactive
            if click_indx == 0:
                clicker.make_next_click(auto_masks)
            else:
                clicker.make_next_click(pred_mask)
            pred_probs = predictor.get_prediction(clicker, prev_mask=_init_mask)

            _init_mask = torch.tensor(pred_probs, device=device).unsqueeze(0).unsqueeze(0)

            pred_mask = pred_probs > pred_thr

            if callback is not None:
                callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

            iou = utils.get_dice(gt_mask, pred_mask)
            ious_list.append(iou)

            if iou >= max_dice_thr and click_indx + 1 >= min_clicks:
                break

        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs


def evaluate_dataset_with_auto_limit_range(dataset, predictor, **kwargs):
    all_ious = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)

        for object_id in sample.objects_ids:
            _, sample_ious, _, final_prob = evaluate_sample_with_auto_limit_range(sample.image, sample.gt_mask, sample.auto_masks, predictor,
                                                sample_id=index, **kwargs)
        all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time

def evaluate_sample_with_auto_limit_range(image, gt_mask, auto_masks, predictor, max_dice_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20, ROI_size=10,
                    sample_id=None, callback=None):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []

    _init_mask = copy.deepcopy(auto_masks).astype(np.float32)
    _init_mask = torch.tensor(_init_mask, device=device).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        predictor.set_input_image(image)
        final_pred_probs = np.zeros_like(gt_mask)
        for click_indx in range(max_clicks):
            ## make the next interactive
            if click_indx == 0:
                next_click = clicker.make_next_click(auto_masks > pred_thr)
            else:
                next_click = clicker.make_next_click(pred_mask)
            pred_probs = predictor.get_prediction(clicker, prev_mask=_init_mask)

            # print("Max Min", np.max(pred_probs), np.min(pred_probs))
            
            
            ## Bounding box processing
            click_coor = next_click.coords
            is_positive = next_click.is_positive
            radius = ROI_size
            ## create the cirle from a point in image
            zero_mask = np.zeros_like(gt_mask)
            zero_mask_3_chanel = 255 * zero_mask[:, :, np.newaxis].repeat(3, axis=2).astype(np.uint8)
            cirle_image = cv2.circle(zero_mask_3_chanel, 
                                (click_coor[1],click_coor[0]), 
                                radius, 
                                (255,255,255),
                                -1)
            cirle_mask = (cirle_image[:,:,0]/255).astype(np.uint8)
            revert_cirle_mask = (cirle_mask < 1).astype(np.uint8)

            _init_mask_numpy = _init_mask.cpu().numpy()[0, 0]
            # combination_mask_final = np.add(pred_probs*weight_val, _init_mask_numpy*(1-weight_val))
            if is_positive:
                combination_mask_final = np.maximum(pred_probs,_init_mask_numpy)
            else:
                combination_mask_final = np.minimum(pred_probs,_init_mask_numpy)
            limit_range_mask_final = np.multiply(combination_mask_final, cirle_mask)
            fixed_range_mask_final = np.multiply(_init_mask_numpy, revert_cirle_mask)

            final_pred_probs = np.add(fixed_range_mask_final, limit_range_mask_final)

            _init_mask = torch.tensor(final_pred_probs, device=device).unsqueeze(0).unsqueeze(0)

            pred_mask = final_pred_probs > pred_thr

            
            

            if callback is not None:
                callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

            iou = utils.get_dice(gt_mask, pred_mask)
            ious_list.append(iou)

            if iou >= max_dice_thr and click_indx + 1 >= min_clicks:
                break
        

        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs, final_pred_probs