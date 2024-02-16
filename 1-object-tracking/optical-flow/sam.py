"""
SAM-notator is based on Segment-Anything, https://github.com/facebookresearch/segment-anything, which is licensed under the Apache License, Version 2.0.
See the LICENSE file for the full license text.
"""

from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import torch

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

