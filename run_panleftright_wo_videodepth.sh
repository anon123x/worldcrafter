#!/bin/bash

video_names=(city_monet_1 garden_monet real_campus_1 real_campus_2 ukiyo-e venice zelda)
angles=(20 40 60 -20 -40 -60)
gen_sky=False
gen_layer=False

for video in "${video_names[@]}"; do
  for angle in "${angles[@]}"; do
    if (( angle < 0 )); then
      angle_suffix="right"
    else
      angle_suffix="left"
    fi

    output_filename="Ours_ablation_wo_videodepth_{$video}_pan_${angle_suffix}_degree_{${angle}}_genSky_${gen_sky}_genLayer_${gen_layer}.mp4"

    echo "Processing video: $video with angle: $angle -> $output_filename"
    CUDA_VISIBLE_DEVICES=1 python test_video_panleftright_wo_videodepth.py \
      --video_name "$video" \
      --target_angle "$angle" \
      --gen_sky "$gen_sky" \
      --gen_layer "$gen_layer" \
      --output_filename "$output_filename"
  done
done

echo "END!!!"
