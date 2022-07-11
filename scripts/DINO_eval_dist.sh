coco_path=$1
checkpoint=$2
python -m torch.distributed.launch --nproc_per_node=8 main.py \
  --output_dir logs/DINO/R50-MS4-%j \
	-c config/DINO/DINO_4scale.py --coco_path $coco_path  \
	--eval --resume $checkpoint \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
