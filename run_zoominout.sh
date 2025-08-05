#!/bin/bash

video_names=(city_monet_1 garden_monet real_campus_1 real_campus_2 ukiyo-e venice zelda)
angles=(20 40 60 -20 -40 -60)
gen_sky=False
gen_layer=False

for video in "${video_names[@]}"; do
  for angle in "${angles[@]}"; do
    if (( angle < 0 )); then
      angle_suffix="out"
    else
      angle_suffix="in"
    fi

    output_filename="Ours_{$video}_zoom_${angle_suffix}_degree_{${angle}}_genSky_${gen_sky}_genLayer_${gen_layer}.mp4"

    echo "Processing video: $video with angle: $angle -> $output_filename"
    CUDA_VISIBLE_DEVICES=2 python test_video_zoominout.py \
      --video_name "$video" \
      --target_angle "$angle" \
      --gen_sky "$gen_sky" \
      --gen_layer "$gen_layer" \
      --output_filename "$output_filename"
  done
done

echo "END!!!"
