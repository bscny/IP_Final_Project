# U-Net Pretrain Run Script

Run from the `project` directory:

```bash
bash unet_pretrain/run.sh
```

By default, `run.sh` enables Weights & Biases with:

```text
project: ip-final-project
run name: unet-pretrain-002
```

## Background Training

Equivalent to `nohup ... > output.log 2>&1 &`:

```bash
bash unet_pretrain/run.sh --background
```

Use a custom log file:

```bash
bash unet_pretrain/run.sh --background --log-file output-002.log
```

Check logs:

```bash
tail -f output.log
```

## Multi-GPU Training

Use 4 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash unet_pretrain/run.sh --gpus 4 --background
```

`--batch-size` is per GPU. For example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash unet_pretrain/run.sh --gpus 4 --batch-size 2 --background
```

This gives a global batch size of `4 * 2 = 8`.

## Common Options

```bash
bash unet_pretrain/run.sh \
  --epochs 200 \
  --batch-size 8 \
  --wandb-run-name unet-pretrain-003
```

Disable wandb:

```bash
bash unet_pretrain/run.sh --no-wandb
```
