import cv2
import numpy as np

from util.utils import renorm
from util.misc import color_sys

_color_getter = color_sys(100)

# plot known and unknown box
def add_box_to_img(img, boxes, colorlist, brands=None):
    """[summary]

    Args:
        img ([type]): np.array, H,W,3
        boxes ([type]): list of list(4)
        colorlist: list of colors.
        brands: text.

    Return:
        img: np.array. H,W,3.
    """
    H, W = img.shape[:2]
    for _i, (box, color) in enumerate(zip(boxes, colorlist)):
        x, y, w, h = box[0] * W, box[1] * H, box[2] * W, box[3] * H
        img = cv2.rectangle(img.copy(), (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), color, 2)
        if brands is not None:
            brand = brands[_i]
            org = (int(x-w/2), int(y+h/2))
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            thickness = 1
            img = cv2.putText(img.copy(), str(brand), org, font, 
                fontScale, color, thickness, cv2.LINE_AA)
    return img

def plot_dual_img(img, boxes, labels, idxs, probs=None):
    """[summary]

    Args:
        img ([type]): 3,H,W. tensor.
        boxes (): tensor(Kx4) or list of tensor(1x4).
        labels ([type]): list of ints.
        idxs ([type]): list of ints.
        probs (optional): listof floats.

    Returns:
        img_classcolor: np.array. H,W,3. img with class-wise label.
        img_seqcolor: np.array. H,W,3. img with seq-wise label.
    """
    # import ipdb; ipdb.set_trace()
    boxes = [i.cpu().tolist() for i in boxes]
    img = (renorm(img.cpu()).permute(1,2,0).numpy() * 255).astype(np.uint8)
    # plot with class
    class_colors = [_color_getter(i) for i in labels]
    if probs is not None:
        brands = ["{},{:.2f}".format(j,k) for j,k in zip(labels, probs)]
    else:
        brands = labels
    img_classcolor = add_box_to_img(img, boxes, class_colors, brands=brands)
    # plot with seq
    seq_colors = [_color_getter((i * 11) % 100) for i in idxs]
    img_seqcolor = add_box_to_img(img, boxes, seq_colors, brands=idxs)
    return img_classcolor, img_seqcolor


def plot_raw_img(img, boxes, labels):
    """[summary]

    Args:
        img ([type]): 3,H,W. tensor. 
        boxes ([type]): Kx4. tensor
        labels ([type]): K. tensor.

    return:
        img: np.array. H,W,3. img with bbox annos.
    
    """
    img = (renorm(img.cpu()).permute(1,2,0).numpy() * 255).astype(np.uint8)
    H, W = img.shape[:2]
    for box, label in zip(boxes.tolist(), labels.tolist()):
        x, y, w, h = box[0] * W, box[1] * H, box[2] * W, box[3] * H
        # import ipdb; ipdb.set_trace()
        img = cv2.rectangle(img.copy(), (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), _color_getter(label), 2)
        # add text
        org = (int(x-w/2), int(y+h/2))
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 1
        img = cv2.putText(img.copy(), str(label), org, font, 
            fontScale, _color_getter(label), thickness, cv2.LINE_AA)

    return img