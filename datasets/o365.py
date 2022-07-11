if __name__=="__main__":
    # for debug only
    import os, sys
    sys.path.append(os.path.dirname(sys.path[0]))

from pathlib import Path
import random
import os

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

from .coco import CocoDetection, get_aux_target_hacks_list, make_coco_transforms

from .data_util import preparing_dataset
import datasets.transforms as T
from util.box_ops import box_cxcywh_to_xyxy, box_iou

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

__all__ = ['build_o365']

def build_o365_raw(image_set, args):
    root = Path(args.coco_path)
    # assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train", root / "slannos" / f'anno_preprocess_shilong_{image_set}_v2.json'),
        "val": (root / "val", root / "slannos" / f'anno_preprocess_shilong_{image_set}_v2.json'),
    }

    # add some hooks to datasets
    aux_target_hacks_list = get_aux_target_hacks_list(image_set, args)
    img_folder, ann_file = PATHS[image_set]

    # copy to local path
    if os.environ.get('DATA_COPY_SHILONG'):
        preparing_dataset(dict(img_folder=img_folder, ann_file=ann_file), image_set, args)

    dataset = CocoDetection(img_folder, ann_file, 
            transforms=make_coco_transforms(image_set, fix_size=args.fix_size), 
            return_masks=args.masks,
            aux_target_hacks=aux_target_hacks_list,
        )
    return dataset

def build_o365_combine(image_set, args):
    if image_set == 'train':
        train_ds = build_o365_raw('train', args)
        val_ds = build_o365_raw('val', args)
        val_ds.ids = val_ds.ids[5000:]
        return torch.utils.data.ConcatDataset([train_ds, val_ds])
    if image_set == 'val':
        val_ds = build_o365_raw('val', args)
        val_ds.ids = val_ds.ids[:5000]
        return val_ds
    raise ValueError('Unknown image_set: {}'.format(image_set))


def build_o365(image_set, args):
    root = Path(args.coco_path)
    # assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train", root / "slannos" / f'anno_preprocess_shilong_{image_set}_v2.json'),
        "val": (root / "val", root / "slannos" / f'anno_preprocess_shilong_{image_set}_v2.json'),
    }

    # add some hooks to datasets
    aux_target_hacks_list = get_aux_target_hacks_list(image_set, args)
    img_folder, ann_file = PATHS[image_set]

    # copy to local path
    if os.environ.get('DATA_COPY_SHILONG'):
        preparing_dataset(dict(img_folder=img_folder, ann_file=ann_file), image_set, args)

    dataset = CocoDetection(img_folder, ann_file, 
            transforms=make_coco_transforms(image_set, fix_size=args.fix_size), 
            return_masks=args.masks,
            aux_target_hacks=aux_target_hacks_list,
        )

    if image_set == 'val':
        # we use the first 5000 items for val only
        dataset.ids = dataset.ids[:5000]

    return dataset


if __name__ == "__main__":
    # Objects365 Val example
    dataset_o365 = CocoDetection(
            '/comp_robot/cv_public_dataset/Objects365/train/', 
            "/comp_robot/cv_public_dataset/Objects365/slannos/anno_preprocess_shilong_train_v2.json", 
            transforms=None, 
            return_masks=False,
        )
    print('len(dataset_o365):', len(dataset_o365))

    import ipdb; ipdb.set_trace()