"""
selvabox/loader.py -- load the SelvaBox tree-crown DINO into standalone IDEA-Research/DINO.

The published SelvaBox / CanopyRS detector `dino-swin-l-384-multi-NQOS` is trained with detrex.
This module downloads THAT original checkpoint and converts its state_dict onto this (standalone)
DINO in one call, with no detrex dependency -- the conversion is pure key-rename + shape-match
(`remap_state_dict`, verified bit-exact against the detrex forward pass).

    from selvabox.loader import load_selvabox_treecrown
    model, criterion, postprocessors = load_selvabox_treecrown(cache_dir="~/.cache/selvabox")

The architecture MUST be built from config/DINO/DINO_5scale_swin_treecrown.py -- stock DINO
defaults (pe_temperature=20, no pos-embed offset, shared per-layer heads) silently produce wrong
outputs for this checkpoint. See SELVABOX.md for the full rationale and the upstream diff.

Reading the *raw* detrex checkpoint needs `omegaconf` (a light, pure-python config lib the
checkpoint was pickled with -- NOT detrex). It is only needed for the one-time conversion; once a
converted file is cached in `cache_dir`, subsequent loads need nothing but torch + this repo.
"""

from __future__ import annotations

import os
import re
import sys

import torch

# --- make sure this DINO repo root is importable (config fromfile + build_dino) ----------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_TREECROWN_CFG = os.path.join(_REPO_ROOT, "config", "DINO", "DINO_5scale_swin_treecrown.py")
_CACHE_FILENAME = "selvabox_treecrown_converted.pth"

# Default source = the original detrex/CanopyRS checkpoint on the HuggingFace Hub.
# (The same repo also carries detector.yaml / tilerizer.yaml / aggregator.yaml -- the CanopyRS
# tiling + inference config, the source of truth for input resize/tile size.)
DEFAULT_HF_REPO = "CanopyRS/dino-swin-l-384-multi-NQOS"
DEFAULT_HF_FILENAME = "model_best.pth"


# ==============================================================================================
# remap: detrex/CanopyRS DINO Swin-L 5-scale (single class) -> standalone DINO.
# Copied verbatim from the verified conversion (bit-exact forward parity). Do not edit casually --
# the decoder norm reordering below (norms.0->norm2, norms.1->norm1) is the subtle, tested part.
# ==============================================================================================
RENAME_RULES = [
    # level embed
    (r"^transformer\.level_embeds$", "transformer.level_embed"),
    # neck -> input_proj
    (r"^neck\.convs\.(\d+)\.conv\.(weight|bias)$", r"input_proj.\1.0.\2"),
    (r"^neck\.convs\.(\d+)\.norm\.(weight|bias)$", r"input_proj.\1.1.\2"),
    (r"^neck\.extra_convs\.0\.conv\.(weight|bias)$", r"input_proj.4.0.\1"),
    (r"^neck\.extra_convs\.0\.norm\.(weight|bias)$", r"input_proj.4.1.\1"),
    # shared interm/enc head: detrex index 6 -> standalone enc_out_*
    (r"^class_embed\.6\.(weight|bias)$", r"transformer.enc_out_class_embed.\1"),
    (
        r"^bbox_embed\.6\.layers\.(\d+)\.(weight|bias)$",
        r"transformer.enc_out_bbox_embed.layers.\1.\2",
    ),
    # encoder layers
    (
        r"^transformer\.encoder\.layers\.(\d+)\.attentions\.0\.(.*)$",
        r"transformer.encoder.layers.\1.self_attn.\2",
    ),
    (
        r"^transformer\.encoder\.layers\.(\d+)\.ffns\.0\.layers\.0\.0\.(weight|bias)$",
        r"transformer.encoder.layers.\1.linear1.\2",
    ),
    (
        r"^transformer\.encoder\.layers\.(\d+)\.ffns\.0\.layers\.1\.(weight|bias)$",
        r"transformer.encoder.layers.\1.linear2.\2",
    ),
    (
        r"^transformer\.encoder\.layers\.(\d+)\.norms\.0\.(weight|bias)$",
        r"transformer.encoder.layers.\1.norm1.\2",
    ),
    (
        r"^transformer\.encoder\.layers\.(\d+)\.norms\.1\.(weight|bias)$",
        r"transformer.encoder.layers.\1.norm2.\2",
    ),
    # decoder layers
    (
        r"^transformer\.decoder\.layers\.(\d+)\.attentions\.0\.attn\.(.*)$",
        r"transformer.decoder.layers.\1.self_attn.\2",
    ),
    (
        r"^transformer\.decoder\.layers\.(\d+)\.attentions\.1\.(.*)$",
        r"transformer.decoder.layers.\1.cross_attn.\2",
    ),
    (
        r"^transformer\.decoder\.layers\.(\d+)\.ffns\.0\.layers\.0\.0\.(weight|bias)$",
        r"transformer.decoder.layers.\1.linear1.\2",
    ),
    (
        r"^transformer\.decoder\.layers\.(\d+)\.ffns\.0\.layers\.1\.(weight|bias)$",
        r"transformer.decoder.layers.\1.linear2.\2",
    ),
    # detrex decoder operation_order = (self_attn, norm, cross_attn, norm, ffn, norm)
    #   -> norms.0=post-self_attn, norms.1=post-cross_attn, norms.2=post-ffn
    # standalone decoder layer: norm1=post-cross_attn, norm2=post-self_attn, norm3=post-ffn
    #   -> so norms.0->norm2, norms.1->norm1 (NOT the naive 0->1, 1->2)
    (
        r"^transformer\.decoder\.layers\.(\d+)\.norms\.0\.(weight|bias)$",
        r"transformer.decoder.layers.\1.norm2.\2",
    ),
    (
        r"^transformer\.decoder\.layers\.(\d+)\.norms\.1\.(weight|bias)$",
        r"transformer.decoder.layers.\1.norm1.\2",
    ),
    (
        r"^transformer\.decoder\.layers\.(\d+)\.norms\.2\.(weight|bias)$",
        r"transformer.decoder.layers.\1.norm3.\2",
    ),
    # Swin backbone: detrex 'backbone.*' -> standalone Joiner-wrapped 'backbone.0.*'
    (r"^backbone\.(.*)$", r"backbone.0.\1"),
]

# Intentionally left behind (no destination in the standalone model):
#   label_enc              -> DN-only, different shape, reinitialised
#   decoder.{class,bbox}_embed.6 -> duplicate of {class,bbox}_embed.6 (mapped to enc_out_*)
EXEMPT = (
    "label_enc",
    "transformer.decoder.class_embed.6",
    "transformer.decoder.bbox_embed.6",
)


def _apply_rules(k):
    for pat, repl in RENAME_RULES:
        new, n = re.subn(pat, repl, k)
        if n:
            return new, pat
    return None, None


def remap_state_dict(ckpt_sd, model_b_sd):
    """detrex state_dict -> standalone DINO state_dict. Returns (mapped_sd, report)."""
    mapped, report = (
        {},
        {"unmapped_ckpt_keys": [], "shape_mismatches": [], "rules_hit": {}},
    )
    b_shapes = {k: tuple(v.shape) for k, v in model_b_sd.items()}
    used_b, consumed_a = set(), set()

    def place(bk, ak, v):
        mapped[bk] = v
        used_b.add(bk)
        consumed_a.add(ak)

    # pass 1: rule renames FIRST (rules take priority over coincidental exact names)
    for k, v in ckpt_sd.items():
        if any(k.startswith(e) for e in EXEMPT):
            consumed_a.add(k)
            continue
        bk, rule = _apply_rules(k)
        if bk is None:
            continue
        if bk in b_shapes and tuple(v.shape) == b_shapes[bk]:
            place(bk, k, v)
            report["rules_hit"][rule] = report["rules_hit"].get(rule, 0) + 1
        elif bk in b_shapes:
            report["shape_mismatches"].append((bk, b_shapes[bk], tuple(v.shape)))

    # pass 2: exact name + shape for everything not yet consumed
    for k, v in ckpt_sd.items():
        if k in consumed_a or any(k.startswith(e) for e in EXEMPT):
            continue
        if k in b_shapes and tuple(v.shape) == b_shapes[k] and k not in used_b:
            place(k, k, v)

    # pass 3: unambiguous shape-unique auto-match, BACKBONE ONLY
    rem_b = {k: s for k, s in b_shapes.items() if k not in used_b and k.startswith("backbone")}
    rem_a = {
        k: v
        for k, v in ckpt_sd.items()
        if k not in consumed_a and not any(k.startswith(e) for e in EXEMPT)
    }
    b_by_shape, a_by_shape = {}, {}
    for k, s in rem_b.items():
        b_by_shape.setdefault(s, []).append(k)
    for k, v in rem_a.items():
        a_by_shape.setdefault(tuple(v.shape), []).append(k)
    for shape, aks in a_by_shape.items():
        bks = b_by_shape.get(shape, [])
        if len(aks) == 1 and len(bks) == 1:
            place(bks[0], aks[0], rem_a[aks[0]])
            report["rules_hit"]["auto_shape_backbone"] = (
                report["rules_hit"].get("auto_shape_backbone", 0) + 1
            )

    for k in ckpt_sd:
        if k not in consumed_a and not any(k.startswith(e) for e in EXEMPT):
            report["unmapped_ckpt_keys"].append(k)
    report["unmapped_ckpt_keys"].sort()
    report["n_mapped"] = len(mapped)
    return mapped, report


# ==============================================================================================
# checkpoint io
# ==============================================================================================
def _unwrap_state_dict(ckpt):
    """Return the model-only state_dict from a raw detrex/detectron2 checkpoint (drop EMA/wrappers)."""
    obj = ckpt
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "models"):
            if key in ckpt and isinstance(ckpt[key], dict):
                obj = ckpt[key]
                break
    if isinstance(obj, dict) and obj and all(k.startswith("module.") for k in obj):
        obj = {k[len("module.") :]: v for k, v in obj.items()}  # strip DDP prefix
    return obj


def _load_raw_checkpoint(ckpt_source, cache_dir):
    """Resolve ckpt_source (None -> HF hub default, or a URL, or a local path) to a raw checkpoint."""
    if ckpt_source is None:
        # default: pull the original checkpoint from the HuggingFace Hub
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RuntimeError(
                "Downloading the default checkpoint needs `pip install huggingface_hub`, or pass "
                "ckpt_source=<local .pth path or URL>."
            ) from e
        path = hf_hub_download(
            repo_id=DEFAULT_HF_REPO, filename=DEFAULT_HF_FILENAME, cache_dir=cache_dir
        )
    elif isinstance(ckpt_source, str) and ckpt_source.startswith(("http://", "https://")):
        dst_dir = cache_dir or os.path.join(torch.hub.get_dir(), "selvabox")
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(ckpt_source.split("?")[0]))
        if not os.path.exists(dst):
            torch.hub.download_url_to_file(ckpt_source, dst)
        path = dst
    else:
        path = os.path.expanduser(ckpt_source)
        if not os.path.exists(path):
            raise FileNotFoundError(f"ckpt_source not found: {path!r}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as e:
        if "omegaconf" in str(e):
            raise RuntimeError(
                "Reading the raw detrex checkpoint needs `pip install omegaconf` (one-time, for "
                "conversion only -- not needed once a converted file is cached in cache_dir)."
            ) from e
        raise


def _build_treecrown_model(device):
    """Build the standalone DINO tree-crown model (+criterion, postprocessors) from the config."""
    from util.slconfig import SLConfig
    from models.dino.dino import build_dino

    args = SLConfig.fromfile(_TREECROWN_CFG)
    args.device = device
    model, criterion, postprocessors = build_dino(args)
    return model, criterion, postprocessors


def load_selvabox_treecrown(cache_dir=None, ckpt_source=None, device="cpu"):
    """Load the SelvaBox tree-crown DINO, converting the original detrex checkpoint on first use.

    Args:
        cache_dir: if set, the converted standalone state_dict is written here as
            'selvabox_treecrown_converted.pth' and reused on subsequent calls (skips download +
            conversion, and avoids needing omegaconf).
        ckpt_source: URL or local path to the ORIGINAL detrex checkpoint
            (`model_best.pth`). Default (None) pulls it from HF `CanopyRS/dino-swin-l-384-multi-NQOS`. Ignored if a
            converted cache already exists.
        device: "cpu" or "cuda".

    Returns:
        (model, criterion, postprocessors) -- raw standalone-DINO components. Wrap with the
        treebench SDK (DINODetector.from_components) for the torchvision-style detector.
    """
    if cache_dir:
        cache_dir = os.path.expanduser(cache_dir)
    model, criterion, postprocessors = _build_treecrown_model(device)

    cache_path = os.path.join(cache_dir, _CACHE_FILENAME) if cache_dir else None
    if cache_path and os.path.exists(cache_path):
        sd = torch.load(cache_path, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        missing, unexpected = model.load_state_dict(sd, strict=False)
        _check_load(missing, unexpected)
        return model.to(device), criterion.to(device) if hasattr(criterion, "to") else criterion, postprocessors

    # first use: download (or read) + unwrap + remap + load + cache
    raw = _load_raw_checkpoint(ckpt_source, cache_dir)
    ckpt_sd = _unwrap_state_dict(raw)
    mapped, report = remap_state_dict(ckpt_sd, model.state_dict())
    if report["unmapped_ckpt_keys"] or report["shape_mismatches"]:
        raise RuntimeError(
            "Remap incomplete -- checkpoint/architecture mismatch. "
            f"unmapped={report['unmapped_ckpt_keys'][:5]} "
            f"shape_mismatches={report['shape_mismatches'][:5]}"
        )
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    _check_load(missing, unexpected)

    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save({"model": model.state_dict()}, cache_path)

    return model.to(device), criterion.to(device) if hasattr(criterion, "to") else criterion, postprocessors


def _check_load(missing, unexpected):
    # label_enc is intentionally reinitialised (DN-only); everything else must be filled.
    missing = [k for k in missing if "label_enc" not in k]
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
        )


__all__ = ["load_selvabox_treecrown", "remap_state_dict", "DEFAULT_HF_REPO", "DEFAULT_HF_FILENAME"]
