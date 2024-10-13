#!/bin/zsh

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export MODEL_NAME="~/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9"
export DATA_DIR="./cat/train"
export OUTPUT_DIR="./textual_inversion_cat"/v5

accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" \
  --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --repeats=1 \
  --flip_p=0.0 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=1000 \
  --learning_rate=5e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --num_vectors=6