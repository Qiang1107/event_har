# har_rgbe

## Training Script

`train.py` supports selecting different backbones and loading a pretrained
model before training. Use the `--model_type` flag to choose between `cnn`,
`vit` and `pointnet2`. PointNet2 uses the configuration block `pointnet2_model`
in the yaml file. The `--pretrained` argument can be used for ViT models:


```bash
python train.py --config path/to/config.yaml \
                --model path/to/save_best.pth \
                --log path/to/log.txt \
                --model_type pointnet2 \
                --pretrained path/to/pretrained_weights.pth
```
