from pathlib import Path

from .coco import CocoDetection
import datasets.transforms as T


def make_coco_transforms(image_set, fix_size=False):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        if fix_size:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize([(1333, 1333)]),
                # T.RandomResize([800], max_size=1333),
                normalize,
            ])
        else:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ),
                normalize,
            ])

    if image_set in ['val', 'eval_debug', 'train_reg', 'test']  :
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def make_vanke_transforms(image_set, fix_size=False):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    scales = [2 * _ for _ in scales]

    if image_set == 'train':
        if fix_size:
            return T.Compose([
                T.RandomResize([1600,], max_size=1600),
                normalize,
            ])
        else:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=2000),
                    T.Compose([
                        T.RandomResize([800, 1000, 1200]),
                        T.RandomSizeCrop(768, 1200),
                        T.RandomResize(scales, max_size=2000),
                    ])),
                normalize,
            ])

    if image_set in ['val', 'eval_debug', 'train_reg', 'test']:
        return T.Compose([
            T.RandomResize([1600], max_size=1600),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_vanke(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided vanke path {root} does not exist'
    PATHS = {
        "train": (root / "val" / "images", root / "val.cocojson"),
        "val": (root / "val" / "images", root / "val.cocojson"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder,
                            ann_file,
                            transforms=make_coco_transforms(
                                image_set, fix_size=args.fix_size),
                            return_masks=args.masks)
    return dataset