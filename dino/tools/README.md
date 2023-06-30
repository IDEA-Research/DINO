We provide a scirpt to calculate model size, GFLOPS, and FPS.

An example to use it:
```bash
python tools/benchmark.py \
    --output_dir logs/test_flops \
    -c config/DINO/DINO_4scale.py \
    --options batch_size=1 \
    --coco_path /path/to/your/coco/dir
```
