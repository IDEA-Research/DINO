# SelvaBox tree-crown DINO — fork notes

This is a fork of [IDEA-Research/DINO](https://github.com/IDEA-Research/DINO) (base commit
`d84a491`) that can load the **SelvaBox / CanopyRS `dino-swin-l-384-multi-NQOS`** tree-crown
detector — a Swin-L 384, 5-scale, single-class DINO trained with **detrex** — into this
standalone DINO, with **bit-exact forward parity** to the original.

```python
import sys; sys.path.insert(0, "<path to this repo>")
from selvabox.loader import load_selvabox_treecrown

# first call downloads/reads the ORIGINAL detrex checkpoint, converts it, and caches the result;
# later calls load straight from the cache (no download, no omegaconf).
model, criterion, postprocessors = load_selvabox_treecrown(
    cache_dir="~/.cache/selvabox",
    ckpt_source="/path/to/dino-swin-l-384-multi-NQOS.pth",  # or a URL; or set DEFAULT_CKPT_URL
    device="cuda",
)
```

`load_selvabox_treecrown` returns the raw `(model, criterion, postprocessors)`. In treebench,
wrap them with `DINODetector.from_components(...)` (see `treebench/detection/dino.py`) for the
torchvision-style detector — that wrapper owns the inference contract (normalize/pad/postprocess).

## Why a fork? (the changes vs upstream DINO)

### 1. Code change — `models/dino/position_encoding.py` (the parity-critical one)
detrex's `PositionEmbeddingSine` normalizes with an **`offset = -0.5`**; stock DINO's
`PositionEmbeddingSineHW` has no offset. Without it, `enc_layer0` already diverges ~0.14 and it
cascades. We add an `offset` argument (default **0.0**, so every other DINO config is unchanged)
and wire `args.pe_offset` through `build_position_encoding`:

```diff
-def __init__(self, num_pos_feats=64, temperatureH=10000, temperatureW=10000, normalize=False, scale=None):
+def __init__(self, num_pos_feats=64, temperatureH=10000, temperatureW=10000, normalize=False, scale=None, offset=0.0):
     ...
+    self.offset = offset
     ...
-    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
-    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
+    y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + eps) * self.scale
+    x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + eps) * self.scale
     ...
 position_embedding = PositionEmbeddingSineHW(
-    N_steps, temperatureH=args.pe_temperatureH, temperatureW=args.pe_temperatureW, normalize=True)
+    N_steps, temperatureH=args.pe_temperatureH, temperatureW=args.pe_temperatureW,
+    normalize=True, offset=getattr(args, "pe_offset", 0.0))
```

### 2. Build fixes — `models/dino/ops/src/{ms_deform_attn.h, cuda/ms_deform_attn_cuda.cu}`
Modern-PyTorch API updates so the deformable-attention CUDA op compiles: `value.type().is_cuda()`
→ `value.is_cuda()`, and `value.type()` → `value.scalar_type()` in `AT_DISPATCH_FLOATING_TYPES`.
No numerical change.

### 3. Added files (this fork)
- `config/DINO/DINO_5scale_swin_treecrown.py` — architecture config for the checkpoint.
- `selvabox/loader.py` — `remap_state_dict` + `load_selvabox_treecrown`.

## The config knobs the checkpoint requires (all in the tree config)
Loading the checkpoint under **stock DINO defaults silently gives wrong outputs** — none of these
are cosmetic:

| setting | stock DINO | required | why |
|---|---|---|---|
| `pe_temperatureH/W` | 20 | **10000** | detrex swin config uses 10000 |
| `pe_offset` | (none) | **-0.5** | detrex `PositionEmbeddingSine` offset |
| `dec_pred_bbox_embed_share` | True | **False** | detrex uses distinct per-layer heads (`copy.deepcopy`) |
| `dec_pred_class_embed_share` | True | **False** | same |
| `num_feature_levels` | 4/5 | **5** | 5-scale |
| `return_interm_indices` | — | **[0,1,2,3]** | all 4 Swin stages |
| `backbone` | resnet50 | **swin_L_384_22k** | |
| `num_classes` / `dn_labelbook_size` | 91 | **1** | single class (tree crown) |
| `num_select` | 300 | **300** | matches detrex `select_box_nums_for_evaluation` |

## Inference contract (for exact CanopyRS/detrex behaviour)
- **RGB** input, normalize with ImageNet `mean=[123.675,116.28,103.53]`,
  `std=[58.395,57.12,57.375]`. The standalone model does **not** self-normalize.
- Batch by padding to a multiple of **32** with a mask (detectron2/detrex behaviour); the mask
  flows into deformable-attention and the position embedding.
- Post-process: top-`300` queries → `xyxy` absolute (rescaled by original image size).
- Resize/tiling is the caller's job — given identical input pixels the output is bit-identical.

## Conversion details
- `remap_state_dict` is a pure key-rename + shape-match (no detrex import). The one subtle part:
  the standalone decoder layer orders norms as `norm1=post-cross-attn, norm2=post-self-attn,
  norm3=post-ffn`, the **opposite** of detrex's `norms.0/1`, so `norms.0→norm2, norms.1→norm1`.
- `label_enc` is intentionally left reinitialised (DN-only, different shape); it retrains during
  fine-tuning and does not affect detection queries.
- Reading the **raw** detrex checkpoint needs `pip install omegaconf` (the checkpoint was pickled
  with it — a light, pure-python config lib, not detrex). Only needed for the one-time conversion;
  the cached converted file loads with just torch.
</content>
