"""
SAM-notator is based on Segment-Anything, https://github.com/facebookresearch/segment-anything, which is licensed under the Apache License, Version 2.0.
See the LICENSE file for the full license text.
"""

from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import torch
import numpy as np

def sam(check_point, model_type):
    # for MacOS
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # for windows
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(device)

    sam = sam_model_registry[model_type](checkpoint=check_point)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
                                                model=sam,
                                                points_per_side=32,
                                                pred_iou_thresh=0.9,
                                                stability_score_thresh=0.96,
                                                crop_n_layers=1,
                                                crop_n_points_downscale_factor=2,
                                                min_mask_region_area=100, # Requires open-cv to run post-processing
                                                )

    predictor = SamPredictor(sam)

    return mask_generator, predictor

def segment_single_prompt(image, input_points, input_labels):
    """

    :param image:
    :param input_points:
    :param input_labels:
    :return:
    """
    global scores
    global logits

    predictor.set_image(image)

    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )

    return masks

def segment_mult_prompts(image, input_points, input_labels):
    """

    :param image:
    :param input_points:
    :param input_labels:
    :return:
    """
    global scores
    global logits

    predictor.set_image(image)
    for idx, point in enumerate(input_points):
        if idx == 0:
            _ = segment_single_prompt(image, np.array([point]), np.array([input_labels[idx]]))
        else:
            mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
            masks, _, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                mask_input=mask_input[None, :, :],
                multimask_output=False,
            )
            return masks


scores, logits = None, None
check_point = 'sam/sam_vit_h_4b8939.pth'
model_type = 'vit_h'
mask_generator, predictor = sam(check_point, model_type)