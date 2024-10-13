#!/bin/zsh

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./cat/train"
export EMBEDDING_PATH="./textual_inversion_cat/v4/learned_embeds.safetensors"
export NOISE_PATH="./textual_inversion_cat/v4/noise.pt"

python textual_inversion_inference.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --prompt="<cat-toy>" \
  --embeddings_path=$EMBEDDING_PATH \
  --noise_path=$NOISE_PATH \
  --resolution=512 \
  --train_batch_size=1 \
  --num_inference_steps=100 \
  --flip_p=0.0