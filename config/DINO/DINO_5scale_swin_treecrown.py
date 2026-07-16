# SelvaBox tree-crown DINO: Swin-L 384, 5-scale, single class.
#
# Architecture config for the CanopyRS/detrex `dino-swin-l-384-multi-NQOS` checkpoint after
# conversion to standalone IDEA-Research/DINO. These overrides are NOT cosmetic -- every one is
# required for bit-exact parity with the original detrex model (see selvabox/loader.py and
# SELVABOX.md). Loading the checkpoint under stock DINO defaults silently produces wrong outputs.
_base_ = ['DINO_5scale.py']

# --- single class (tree crown) ---
num_classes = 1
dn_labelbook_size = 1

# --- Swin-L 384 backbone, all 4 stages feed the 5-scale neck ---
backbone = 'swin_L_384_22k'
return_interm_indices = [0, 1, 2, 3]
num_feature_levels = 5

# --- position embedding: detrex uses temperature 10000 + offset -0.5 (stock DINO: 20 / none) ---
pe_temperatureH = 10000
pe_temperatureW = 10000
pe_offset = -0.5

# --- detrex builds INDEPENDENT per-decoder-layer heads (copy.deepcopy), not shared ---
dec_pred_bbox_embed_share = False
dec_pred_class_embed_share = False

# no grad checkpointing (parity), inference uses top-300 like detrex select_box_nums_for_evaluation
use_checkpoint = False
num_select = 300
