coco_path=$1
backbone_dir=$2
python run_with_submitit.py --timeout 3000 --job_name DINO \
	--job_dir logs/DINO/R50-MS4-%j --ngpus 8 --nodes 1 \
	-c config/DINO/DINO_4scale_convnext.py --coco_path $coco_path \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 backbone_dir=$backbone_dir
