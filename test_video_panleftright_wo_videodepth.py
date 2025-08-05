
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import gc
import random
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from datetime import datetime
import threading
from torchvision.transforms import ToTensor

from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm
from diffusers import AutoencoderKL, DDIMScheduler, EulerDiscreteScheduler
from util.stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from marigold_lcm.marigold_pipeline import MarigoldPipeline, MarigoldPipelineNormal, MarigoldNormalsPipeline

from models.models import KeyframeGen, save_point_cloud_as_ply
from util.gs_utils import save_pc_as_3dgs, convert_pc_to_splat
from util.chatGPT4 import TextpromptGen
from util.general_utils import apply_depth_colormap, save_video, read_video_frames, save_video_2
from util.utils import save_depth_map, prepare_scheduler, soft_stitching
from util.utils import load_example_yaml, convert_pt3d_cam_to_3dgs_cam
from util.segment_utils import create_mask_generator_repvit
from util.free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
 
from arguments import GSParams, CameraParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.loss import l1_loss, ssim
from scene.cameras import Camera
from random import randint
import time
import cv2
from syncdiffusion.syncdiffusion_model import SyncDiffusion
from kornia.morphology import dilation
import warnings
import os
import copy
from models.infer import DepthCrafterDemo

from models.autoencoder_magvit import AutoencoderKLCogVideoX
from models.crosstransformer3d import CrossTransformer3DModel
from models.pipeline_trajectorycrafter import TrajCrafter_Pipeline
from transformers import T5EncoderModel
from diffusers import (
    AutoencoderKL,
    CogVideoXDDIMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
)

import shutil

warnings.filterwarnings("ignore")

xyz_scale = 1000
client_id = None
scene_name = None
view_matrix = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
view_matrix_wonder = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
view_matrix_delete = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

view_matrix_fixed = np.array([
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [0, 0.2, 0.5, 1]
])
theta = np.radians(-3)
rotation_matrix_x = np.array([
    [1, 0, 0, 0],
    [0, np.cos(theta), -np.sin(theta), 0],
    [0, np.sin(theta), np.cos(theta), 0],
    [0, 0, 0, 1]
])
view_matrix_fixed = np.dot(view_matrix_fixed, rotation_matrix_x)
view_matrix_fixed = view_matrix_fixed.flatten().tolist()

background = torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32, device='cuda')
latest_frame = None
latest_viz = None
iter_number = None
kf_gen = None
gaussians = None
opt = None
scene_dict = None
style_prompt = None
pt_gen = None
change_scene_name_by_user = False
undo = False
save = False
delete = False
exclude_sky = False

# TRAJ_ROOT = "/data5/xxx/TrajectoryCrafter"
TRAJ_ROOT = "/hdd/xxx/TrajectoryCrafter"

def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()


def seeding(seed):
    if seed == -1:
        seed = np.random.randint(2 ** 32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(f"running with seed: {seed}.")


def run(config, input_path=None, output_filename=None, target_angle=None):
    global client_id, view_matrix, scene_name, latest_frame, kf_gen, latest_viz, gaussians, opt, background, scene_dict, style_prompt, pt_gen, change_scene_name_by_user, undo, save, delete, exclude_sky, view_matrix_delete, view_matrix_wonder
    ###### ------------------ Load modules ------------------ ######
    seeding(config["seed"])
    example = config['example_name']

    segment_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    segment_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large").to('cuda')
    
    mask_generator = create_mask_generator_repvit()
    
    rotation_path = config['rotation_path'][:config['num_scenes']]
    assert len(rotation_path) == config['num_scenes']

    depth_model = MarigoldPipeline.from_pretrained("prs-eth/marigold-v1-0", torch_dtype=torch.bfloat16).to(config["device"])
    depth_model.scheduler = EulerDiscreteScheduler.from_config(depth_model.scheduler.config)
    depth_model.scheduler = prepare_scheduler(depth_model.scheduler)

    normal_estimator = MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-v0-1", torch_dtype=torch.bfloat16).to(config["device"])
    
    yaml_data = load_example_yaml(config["example_name"], 'examples/examples.yaml')
    content_prompt, style_prompt, adaptive_negative_prompt, background_prompt, control_text, outdoor = yaml_data['content_prompt'], yaml_data['style_prompt'], yaml_data['negative_prompt'], yaml_data.get('background', None), yaml_data.get('control_text', None), yaml_data.get('outdoor', False)
    if adaptive_negative_prompt != "":
        adaptive_negative_prompt += ", "

    generator = torch.Generator(device=config["device"]).manual_seed(config["seed"])
    model_name=f'/{TRAJ_ROOT}/checkpoints/CogVideoX-Fun-V1.1-5b-InP'
    transformer_path=f'/{TRAJ_ROOT}/checkpoints/TrajectoryCrafter'
    sampler_name='DDIM_Origin'
    weight_dtype = torch.bfloat16
    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_name, subfolder="vae"
    ).to(weight_dtype)
    text_encoder = T5EncoderModel.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=weight_dtype
    )
    # Get Scheduler
    Choosen_Scheduler = {
        "Euler": EulerDiscreteScheduler,
        "Euler A": EulerAncestralDiscreteScheduler,
        "DPM++": DPMSolverMultistepScheduler,
        "PNDM": PNDMScheduler,
        "DDIM_Cog": CogVideoXDDIMScheduler,
        "DDIM_Origin": DDIMScheduler,
    }[sampler_name]
    scheduler = Choosen_Scheduler.from_pretrained(
        model_name, subfolder="scheduler"
    )
    transformer = CrossTransformer3DModel.from_pretrained(transformer_path).to(
        weight_dtype
    )
    inpainter_video = TrajCrafter_Pipeline.from_pretrained(
        model_name,
        vae=vae,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=weight_dtype,
    )
    
    low_gpu_memory_mode = False
    if low_gpu_memory_mode:
        inpainter_video.enable_sequential_cpu_offload()
    else:
        inpainter_video.enable_model_cpu_offload()

    ## 1. load frames
    frames = read_video_frames(video_path=input_path, process_length=49, stride=1)

    frames_orig = copy.deepcopy(frames)
    frame_len = len(frames)
    print("Loaded len(frames), frames.shape: ", frame_len, frames.shape)
    print("-" * 50)
    
    # ## 2. predict depths from video
    # depth_estimater = DepthCrafterDemo(
    #         unet_path=f'{TRAJ_ROOT}/checkpoints/DepthCrafter',
    #         pre_train_path=f'{TRAJ_ROOT}/checkpoints/stable-video-diffusion-img2vid',
    #         cpu_offload='model',
    #         device=config["device"],
    #     )
    # depths = depth_estimater.infer(
    #     frames,
    #     near=0.0001,
    #     far=10000.0,
    #     num_denoising_steps=5,
    #     guidance_scale=1.0,
    #     window_size=110,
    #     overlap=25,
    # ).to(config["device"])
    # print("Estimated len(depths): ", len(depths))
    # print("-" * 50)
    
    pan_scale = 0.3
    target_angle = np.radians(target_angle) * pan_scale
    angle_list = np.linspace(0, target_angle, frame_len)
    view_matrix_list = []
    for current_angle in angle_list:
        view_matrix = np.array([
            [-np.cos(current_angle), 0, np.sin(current_angle), 0],
            [0, -1, 0, 0],
            [np.sin(current_angle), 0, np.cos(current_angle), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        view_matrix_list.append(view_matrix)

    save_data_list = []
    dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
    output_fold = f"ablation_wo_videodepth_-{dt_string}"

    outpaint_condition_image_list = []
    outpaint_mask_list = []
    frames_ref_list = []
    for k in range(len(frames)):
        view_matrix = view_matrix_list[k]
        curr_frame = frames[k]
        # curr_depth = depths[k]

        print("k: ", k)
        print("view_matrix: \n", view_matrix)
        
        print('###### ------------------ Keyframe (the major part of point clouds) generation ------------------ ######') 
        print(">>> output_fold: ", output_fold)
        kf_gen = KeyframeGen(config=config, inpainter_pipeline=None, mask_generator=mask_generator, depth_model=depth_model,
            segment_model=segment_model, segment_processor=segment_processor, normal_estimator=normal_estimator,
            rotation_path=rotation_path, inpainting_resolution=config['inpainting_resolution_gen'], output_fold=output_fold)
        kf_gen = kf_gen.to(config["device"])
        kf_gen.kf_idx  = k
        print("!!! output root kf_gen.run_dir: ", kf_gen.run_dir)
        
        ## update image_latest should replaced w curr_latest image start from 2nd frame
        # curr_frame.shape, type(curr_frame), curr_frame.dtype, np.max(curr_frame), np.min(curr_frame):  (512, 512, 3) <class 'numpy.ndarray'> float32 1.0 0.0
        print("curr_frame.shape, type(curr_frame), curr_frame.dtype, np.max(curr_frame), np.min(curr_frame): ", curr_frame.shape, type(curr_frame), curr_frame.dtype, np.max(curr_frame), np.min(curr_frame))
        kf_gen.image_latest = ToTensor()(curr_frame).unsqueeze(0).to(config['device'])
        print("kf_gen.image_latest.shape, type(kf_gen.image_latest): ", kf_gen.image_latest.shape, type(kf_gen.image_latest), torch.max(kf_gen.image_latest), torch.min(kf_gen.image_latest))
        # kf_gen.image_latest.shape, type(kf_gen.image_latest):  torch.Size([1, 3, 512, 512]) <class 'torch.Tensor'> tensor(1., device='cuda:0') tensor(0., device='cuda:0')
        print("+" * 50)

        if config['gen_sky_image'] or (not os.path.exists(f'examples/sky_images/{example}/sky_0.png') and not os.path.exists(f'examples/sky_images/{example}/sky_1.png')):
            syncdiffusion_model = SyncDiffusion(config['device'], sd_version='2.0-inpaint')
        else:
            syncdiffusion_model = None
        sky_mask = kf_gen.generate_sky_mask().float()
        
        ## start generate the 0th frame
        kf_gen.generate_sky_pointcloud(syncdiffusion_model, image=kf_gen.image_latest, mask=sky_mask, gen_sky=config['gen_sky_image'], style=style_prompt)
        # kf_gen.recompose_image_latest_and_set_current_pc(scene_name=scene_name) 
        
        print(f"!!! current frame: {k}")
        print(f"!!! current kf_gen.kf_idx is: {kf_gen.kf_idx}")
        ## for wo_videodepth ablation study
        # kf_gen.recompose_image_latest_and_set_current_pc_v(scene_name=scene_name, depth=curr_depth, save_curr=False) # this function will call get_depth, inpaint
        # kf_gen.recompose_image_latest_and_set_current_pc(scene_name=scene_name, save_curr=False)
        kf_gen.recompose_image_latest_and_set_current_pc(scene_name=scene_name)

        # kf_gen.increment_kf_idx() # previous save_curr is False
        pt_gen = TextpromptGen(kf_gen.run_dir, isinstance(control_text, list))
        
        content_list = content_prompt.split(',')
        scene_name = content_list[0]
        entities = content_list[1:]
        scene_dict = {'scene_name': scene_name, 'entities': entities, 'style': style_prompt, 'background': background_prompt}
        inpainting_prompt = content_prompt
        print("1 +++ inpainting_prompt: ", inpainting_prompt)
        
        ###### ------------------ Main loop ------------------ ######
        if config['gen_sky'] or not os.path.exists(f'examples/sky_images/{example}/finished_3dgs_sky_tanh.ply'):
            traindatas = kf_gen.convert_to_3dgs_traindata(xyz_scale=xyz_scale, remove_threshold=None, use_no_loss_mask=False)
            if config['gen_layer']:
                traindata, traindata_sky, traindata_layer = traindatas
            else:
                traindata, traindata_sky = traindatas
            gaussians = GaussianModel(sh_degree=0, floater_dist2_threshold=9e9)
            opt = GSParams()
            opt.max_screen_size = 100  # Sky is supposed to be big; set a high max screen size
            opt.scene_extent = 1.5  # Sky is supposed to be big; set a high scene extent
            opt.densify_from_iter = 200  # Need to do some densify
            opt.prune_from_iter = 200  # Don't prune for sky because sky 3DGS are supposed to be big; prevent it by setting a high prune iter
            opt.densify_grad_threshold = 1.0  # Do not need to densify; Set a high threshold to prevent densifying
            opt.iterations = 399  # More iterations than 100 needed for sky
            scene = Scene(traindata_sky, gaussians, opt, is_sky=True)
            # dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
            # save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene_sky"
            train_gaussian(gaussians, scene, opt, save_dir=None, initialize_scaling=False)
            gaussians.save_ply_with_filter(f'examples/sky_images/{example}/finished_3dgs_sky_tanh.ply')
        else:
            gaussians = GaussianModel(sh_degree=0)
            gaussians.load_ply_with_filter(f'examples/sky_images/{example}/finished_3dgs_sky_tanh.ply')  # pure sky

        gaussians.visibility_filter_all = torch.zeros(gaussians.get_xyz_all.shape[0], dtype=torch.bool, device='cuda')
        gaussians.delete_mask_all = torch.zeros(gaussians.get_xyz_all.shape[0], dtype=torch.bool, device='cuda')
        gaussians.is_sky_filter = torch.ones(gaussians.get_xyz_all.shape[0], dtype=torch.bool, device='cuda')
        
        if config['load_gen'] and os.path.exists(f'examples/sky_images/{example}/finished_3dgs.ply') and os.path.exists(f'examples/sky_images/{example}/visibility_filter_all.pth') and os.path.exists(f'examples/sky_images/{example}/is_sky_filter.pth') and os.path.exists(f'examples/sky_images/{example}/delete_mask_all.pth'):
            print("Loading existing 3DGS...")
            gaussians = GaussianModel(sh_degree=0)
            gaussians.load_ply_with_filter(f'examples/sky_images/{example}/finished_3dgs.ply')
            gaussians.visibility_filter_all = torch.load(f'examples/sky_images/{example}/visibility_filter_all.pth').to('cuda')
            gaussians.is_sky_filter = torch.load(f'examples/sky_images/{example}/is_sky_filter.pth').to('cuda')
            gaussians.delete_mask_all = torch.load(f'examples/sky_images/{example}/delete_mask_all.pth').to('cuda')
        opt = GSParams()

        ### First scene 3DGS
        if config['gen_layer']:
            ## the 0-th frame 
            traindata, traindata_layer = kf_gen.convert_to_3dgs_traindata_latest_layer(xyz_scale=xyz_scale)
            gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
            scene = Scene(traindata_layer, gaussians, opt) ## process traindata_layer
            # dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
            # save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene_layer{0:02d}"
            train_gaussian(gaussians, scene, opt, save_dir=None)  # Base layer training, render layer obj
        else:
            traindata = kf_gen.convert_to_3dgs_traindata_latest(xyz_scale=xyz_scale, use_no_loss_mask=False)

        gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
        scene = Scene(traindata, gaussians, opt) ## process traindata
        # dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        # save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene{i:02d}"
        train_gaussian(gaussians, scene, opt, save_dir=None)  ## render bg train

        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_at_origin(), xyz_scale=xyz_scale)
        gaussians.set_inscreen_points_to_visible(tdgs_cam)
    
        inpainting_prompt = pt_gen.generate_prompt(style=style_prompt, entities=scene_dict['entities'], background=scene_dict['background'], scene_name=scene_dict['scene_name'])
        scene_name = scene_dict['scene_name'] if isinstance(scene_dict['scene_name'], str) else scene_dict['scene_name'][0]
        print("2 +++ inpainting_prompt: ", inpainting_prompt)
        
        ###### ------------------ Keyframe (the major part of point clouds) generation ------------------ ######        
        kf_gen.set_kf_param(inpainting_resolution=config['inpainting_resolution_gen'], inpainting_prompt=inpainting_prompt, adaptive_negative_prompt=adaptive_negative_prompt)
        current_pt3d_cam = kf_gen.get_camera_by_js_view_matrix(view_matrix, xyz_scale=xyz_scale)
        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(current_pt3d_cam, xyz_scale=xyz_scale)
        kf_gen.set_current_camera(current_pt3d_cam, archive_camera=True)

        with torch.no_grad():
            # render_pkg = render(tdgs_cam, gaussians, opt, background)
            render_pkg_nosky = render(tdgs_cam, gaussians, opt, background, exclude_sky=True)

        side_sky_height = 128
        sky_cond_width = 40
        # inpaint_mask_0p5_nosky = (render_pkg_nosky["final_opacity"]<0.6)
        inpaint_mask_0p5_nosky = (render_pkg_nosky["final_opacity"]<0.1)
        # inpaint_mask_0p0_nosky = (render_pkg_nosky["final_opacity"]<0.01)  # Should not have holes in existing regions
        # inpaint_mask_0p5 = (render_pkg["final_opacity"]<0.6)
        # inpaint_mask_0p0 = (render_pkg["final_opacity"]<0.01)  # Should not have holes in existing regions
        # fg_mask_0p5_nosky = ~inpaint_mask_0p5_nosky.clone()
        # foreground_cols = torch.sum(fg_mask_0p5_nosky == 1, dim=1)>150  # [1, 512]
        # foreground_cols_idx = torch.nonzero(foreground_cols, as_tuple=True)[1]

        # mask_using_full_render = torch.zeros(1, 1, 512, 512).to(config['device'])
        # if foreground_cols_idx.numel() > 0:
        #     min_index = foreground_cols_idx.min().item()
        #     max_index = foreground_cols_idx.max().item()
        #     mask_using_full_render[:, :, :, min_index:max_index+1] = 1
        # mask_using_full_render[:, :, :sky_cond_width, :] = 1
        # mask_using_full_render[:, :, :side_sky_height, :sky_cond_width] = 1
        # mask_using_full_render[:, :, :side_sky_height, -sky_cond_width:] = 1
        
        # mask_using_nosky_render = 1 - mask_using_full_render
        # outpaint_condition_image = render_pkg_nosky["render"] * mask_using_nosky_render + render_pkg["render"] * mask_using_full_render
        # fill_mask = inpaint_mask_0p5_nosky * mask_using_nosky_render + inpaint_mask_0p5 * mask_using_full_render
        # outpaint_mask = inpaint_mask_0p0_nosky * mask_using_nosky_render + inpaint_mask_0p0 * mask_using_full_render
        # outpaint_mask = dilation(outpaint_mask, kernel=torch.ones(7, 7).cuda())


        outpaint_mask = inpaint_mask_0p5_nosky.unsqueeze(0)
        outpaint_condition_image = render_pkg_nosky["render"].unsqueeze(0)
        
        outpaint_condition_image_list.append(outpaint_condition_image)
        outpaint_mask_list.append(outpaint_mask)
        frames_ref_list.append(kf_gen.image_latest)
        
        # # save rendered_image for visualization and debug
        # # print(render_pkg["render"].shape, render_pkg_nosky["render"].shape)
        
        # # viz_r = render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
        # # viz_r = (viz_r * 255).astype(np.uint8)
        # # viz_r = viz_r[..., ::-1]
        
        # viz_r_nosky = render_pkg_nosky["render"].permute(1, 2, 0).detach().cpu().numpy()
        # viz_r_nosky = (viz_r_nosky * 255).astype(np.uint8)
        # viz_r_nosky = viz_r_nosky[..., ::-1]
        
        # viz = outpaint_condition_image[0].permute(1, 2, 0).detach().cpu().numpy()
        # viz = (viz * 255).astype(np.uint8)
        # viz = viz[..., ::-1]
        # outpaint_mask_viz = outpaint_mask[0].permute(1, 2, 0).detach().cpu().numpy()
        # outpaint_mask_viz = (outpaint_mask_viz * 255).astype(np.uint8)
        # outpaint_mask_viz = outpaint_mask_viz[..., ::-1]

        # fill_mask_nosky_viz = inpaint_mask_0p5_nosky.permute(1, 2, 0).detach().cpu().numpy()
        # fill_mask_nosky_viz = (fill_mask_nosky_viz * 255).astype(np.uint8)
        # fill_mask_nosky_viz = fill_mask_nosky_viz[..., ::-1]
        
        # # fill_mask_full_viz = inpaint_mask_0p5.permute(1, 2, 0).detach().cpu().numpy()
        # # fill_mask_full_viz = (fill_mask_full_viz * 255).astype(np.uint8)
        # # fill_mask_full_viz = fill_mask_full_viz[..., ::-1]
                    
        # # fill_mask_viz = fill_mask[0].permute(1, 2, 0).detach().cpu().numpy()
        # # fill_mask_viz = (fill_mask_viz * 255).astype(np.uint8)
        # # fill_mask_viz = fill_mask_viz[..., ::-1]
        
        # # ToPILImage()(viz_r[..., ::-1]).save(kf_gen.run_dir / 'images' / "debug" / f'viz_r_{kf_gen.kf_idx}.png')
        # ToPILImage()(viz_r_nosky[..., ::-1]).save(kf_gen.run_dir / 'images' / "debug" / f'viz_r_nosky_{kf_gen.kf_idx}.png')
        # ToPILImage()(viz[..., ::-1]).save(kf_gen.run_dir / 'images' / "debug" / f'rendered_viz_{kf_gen.kf_idx}.png')
        # ToPILImage()(fill_mask_nosky_viz[..., ::-1]).save(kf_gen.run_dir / 'images' / "debug" / f'fill_mask_nosky_viz_{kf_gen.kf_idx}.png')
        # # ToPILImage()(fill_mask_full_viz[..., ::-1]).save(kf_gen.run_dir / 'images' / "debug" / f'fill_mask_full_viz_{kf_gen.kf_idx}.png')
        # ToPILImage()(outpaint_mask_viz[..., ::-1]).save(kf_gen.run_dir / 'images' / "debug" / f'outpaint_mask_viz_{kf_gen.kf_idx}.png')
        # # ToPILImage()(fill_mask_viz[..., ::-1]).save(kf_gen.run_dir / 'images' / "debug" / f'fill_mask_viz_{kf_gen.kf_idx}.png')
        # # print(kf_gen.run_dir / 'images' / "debug" / f'viz_r_{kf_gen.kf_idx}.png')
        # print(kf_gen.run_dir / 'images' / "debug" / f'viz_r_nosky_{kf_gen.kf_idx}.png')
        # print(kf_gen.run_dir / 'images' / "debug" / f'rendered_viz_{kf_gen.kf_idx}.png')
        # print(kf_gen.run_dir / 'images' / "debug" / f'fill_mask_nosky_viz_{kf_gen.kf_idx}.png')
        # # print(kf_gen.run_dir / 'images' / "debug" / f'fill_mask_full_viz_{kf_gen.kf_idx}.png')
        # print(kf_gen.run_dir / 'images' / "debug" / f'outpaint_mask_viz_{kf_gen.kf_idx}.png')
        # # print(kf_gen.run_dir / 'images' / "debug" / f'fill_mask_viz_{kf_gen.kf_idx}.png')
        # print("-" * 50)
    
    cond_video = torch.cat(outpaint_condition_image_list).permute(1, 0, 2, 3).unsqueeze(0)
    cond_masks = torch.cat(outpaint_mask_list).permute(1, 0, 2, 3).unsqueeze(0) * 255.0
    frames_ref = torch.cat(frames_ref_list[:10]).permute(1, 0, 2, 3).unsqueeze(0)
    print("cond_video.shape: ", cond_video.shape, torch.max(cond_video), torch.min(cond_video), type(cond_video), cond_video.dtype)
    print("cond_masks.shape: ", cond_masks.shape, torch.max(cond_masks), torch.min(cond_masks), type(cond_masks), cond_masks.dtype)
    print("frames_ref.shape: ", frames_ref.shape, torch.max(frames_ref), torch.min(frames_ref), type(frames_ref), frames_ref.dtype)
    
    '''
    cond_video.shape:  torch.Size([2, 3, 512, 512]) tensor(0.9953, device='cuda:0') tensor(0.0004, device='cuda:0')
    cond_masks.shape:  torch.Size([2, 1, 512, 512]) tensor(255., device='cuda:0') tensor(0., device='cuda:0')
    frames_ref.shape:  torch.Size([2, 3, 512, 512]) tensor(1., device='cuda:0') tensor(0., device='cuda:0')
    --->
    !!!! torch.max(cond_video):  tensor(1., dtype=torch.float64) tensor(0., dtype=torch.float64) torch.Size([1, 3, 49, 512, 512])
    !!!! torch.max(cond_masks):  tensor(255., dtype=torch.float64) tensor(0., dtype=torch.float64) torch.Size([1, 1, 49, 512, 512])
    !!!! torch.max(frames_ref):  tensor(1., device='cuda:0') tensor(0., device='cuda:0') torch.Size([1, 3, 10, 512, 512])
    outpaint_condition_image.shape:  torch.Size([1, 3, 512, 512]) tensor(0.9962, device='cuda:0') tensor(7.0178e-05, device='cuda:0')
    outpaint_mask.shape:  torch.Size([1, 1, 512, 512]) tensor(1., device='cuda:0') tensor(0., device='cuda:0')
    cond_video.shape:  torch.Size([1, 3, 49, 512, 512]) tensor(1.0000, device='cuda:0') tensor(7.0023e-05, device='cuda:0') <class 'torch.Tensor'> torch.float32
    cond_masks.shape:  torch.Size([1, 1, 49, 512, 512]) tensor(255., device='cuda:0') tensor(0., device='cuda:0') <class 'torch.Tensor'> torch.float32
    frames_ref.shape:  torch.Size([1, 3, 10, 512, 512]) tensor(1., device='cuda:0') tensor(0., device='cuda:0') <class 'torch.Tensor'> torch.float32
    '''
    print("+" * 50)
    ## debug hard code
    # inpainting_prompt_fixed = "Animate this zelda scene into a video with a fixed camera. A brave adventurer in green tunic and cape stands on a hillside overlooking a fantasy kingdom. Start with a gentle breeze rustling the colorful wildflowers in the foreground as butterflies flutter among them. Pan slowly across the lush valley where farmers tend fields, animals graze peacefully, and travelers move along winding paths. Birds soar gracefully above the valley toward a majestic castle perched on a distant mountain. The castle's flag waves gently in the wind as sunlight breaks through fluffy clouds, casting moving shadows across the green landscape. The adventurer's cape sways slightly as they take a step forward, hand resting on their sword, ready to begin their journey toward the castle."
    # prompt = "a boat is traveling down the canal in venice. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic."
    
    # default inpainting_prompt:  Style: animation. Entities:  majestic mountains,  river,  boy.
    inpainting_prompt = inpainting_prompt + "The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic."
    # inpainting_prompt = inpainting_prompt_fixed + "The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic."
    
    # inpaint_prompt_gpt = "Expand the scene into a vivid, looping countryside landscape. Inpaint the grey regions with natural terrain that matches the visible image: lush meadows, rolling green hills, colorful wildflowers, and forested mountains. Add grazing animals like deer, sheep, and cows to blend with the existing ones. Preserve the sunny, warm lighting and the painterly, slightly stylized look. Animate subtle motion — swaying grass and trees, walking animals, and light shifts to mimic time progression."
    # inpainting_prompt = inpaint_prompt_gpt + inpainting_prompt

    # inpaint_prompt_gpt2 = "Fill in the zero-value (black or transparent) regions of the frame with a natural continuation of the visible countryside. Extend the rolling meadows, distant forested hills, wildflowers, and grazing animals across the entire canvas. Keep the painterly, colorful, and sunlit atmosphere. Animate soft movements like rustling trees, walking animals, and shifting sunlight to create a gentle, seamless inpainting video loop."
    # inpainting_prompt = inpaint_prompt_gpt2 + inpainting_prompt

    print("!!!inpainting_prompt: ", inpainting_prompt)
    
    negative_prompt = "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion."       
    with torch.no_grad():
        sample = inpainter_video(
            inpainting_prompt,
            num_frames=49,
            negative_prompt=negative_prompt,
            height=512,
            width=512,
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=50,
            video=cond_video,
            mask_video=cond_masks,
            reference=frames_ref,
        ).videos
    # print(">" * 50)
    # print("sample.shape, type(sample), sample.dtype: ", sample.shape, type(sample), sample.dtype) # sample.shape, type(sample), sample.dtype:  torch.Size([1, 3, 49, 512, 512]) <class 'torch.Tensor'> torch.float32
    # print("sample[0].permute(1, 2, 3, 0).shape: ", sample[0].permute(1, 2, 3, 0).shape)  # sample[0].permute(1, 2, 3, 0).shape:  torch.Size([49, 512, 512, 3])
    # print("torch.max(sample), torch.min(sample): ", torch.max(sample), torch.min(sample)) # torch.max(sample), torch.min(sample):  tensor(1.) tensor(0.)
    save_data = sample[0].permute(1, 2, 3, 0)
    save_path = os.path.join(kf_gen.run_dir, output_filename)
    print("save_data.shape: ", save_data.shape)
    print("save_path: ", save_path)
    save_video_2(
        save_data, 
        save_path,
        # fps=10,
        fps=30,
    )
    empty_cache()
    print("saved: ", save_path)

    comm_output_fold = "output/ours_ablations_wo_videodepth_panleftright_all_results_fps30"
    os.makedirs(comm_output_fold, exist_ok=True)
    cp_path = os.path.join(comm_output_fold, output_filename)
    shutil.copy(save_path, cp_path)

    print(f"cp to: {cp_path}")
    print("=" * 50)
    print("END!!")


def train_gaussian(gaussians: GaussianModel, scene: Scene, opt: GSParams, save_dir: Path, initialize_scaling=True):
    global latest_frame, iter_number, view_matrix, latest_viz
    iterable_gauss = range(1, opt.iterations + 1)
    trainCameras = scene.getTrainCameras().copy()
    gaussians.compute_3D_filter(cameras=trainCameras, initialize_scaling=initialize_scaling)

    for iteration in iterable_gauss:
        # Pick a random Camera
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # import pdb; pdb.set_trace()
        # Render
        render_pkg = render(viewpoint_cam, gaussians, opt, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if iteration == opt.iterations:
        # if iteration % 5 == 0 or iteration == 1:
            time.sleep(0.1)
            print(f'Iteration {iteration}, Loss: {loss.item()}')
            with torch.no_grad():
                tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_by_js_view_matrix(view_matrix, xyz_scale=xyz_scale), xyz_scale=xyz_scale)
                render_pkg = render(tdgs_cam, gaussians, opt, background)
                image = render_pkg['render']
                # rendered_normal = render_pkg['render_normal']
                # rendered_normal_map = rendered_normal/2-0.5
            rendered_image = image.permute(1, 2, 0).detach().cpu().numpy()
            rendered_image = (rendered_image * 255).astype(np.uint8)
            rendered_image = rendered_image[..., ::-1]
            latest_frame = rendered_image
        loss.backward()
        if iteration == opt.iterations:
            print(f'Final loss: {loss.item()}')

        # Use variables that related to the trainable GS
        n_trainable = gaussians.get_xyz.shape[0]
        viewspace_point_tensor_grad, visibility_filter, radii = viewspace_point_tensor.grad[:n_trainable], visibility_filter[:n_trainable], radii[:n_trainable]

        with torch.no_grad():
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if iteration >= opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    max_screen_size = opt.max_screen_size if iteration >= opt.prune_from_iter else None
                    camera_height = 0.0003 * xyz_scale
                    scene_extent = camera_height * 2 if opt.scene_extent is None else opt.scene_extent
                    opacity_lowest = 0.05
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, opacity_lowest, scene_extent, max_screen_size)
                    gaussians.compute_3D_filter(cameras=trainCameras)
                
                # if (iteration % opt.opacity_reset_interval == 0 
                #     or (opt.white_background and iteration == opt.densify_from_iter)
                # ):
                #     gaussians.reset_opacity()

            # if iteration % 100 == 0 and iteration > opt.densify_until_iter:
            #     if iteration < opt.iterations - 100:
            #         # don't update in the end of training
            #         gaussians.compute_3D_filter(cameras=trainCameras)
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

    
def render_current_scene(curr_fid):
    global latest_frame, client_id, iter_number, latest_viz, kf_gen, gaussians, opt, background, view_matrix_wonder, save
    # while True:
    #     time.sleep(0.05)
    if True:
        try:
            with torch.no_grad():
                tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_by_js_view_matrix(view_matrix_wonder, xyz_scale=xyz_scale), xyz_scale=xyz_scale)
                render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True)
            rendered_img = render_pkg['render']
            rendered_image = rendered_img.permute(1, 2, 0).detach().cpu().numpy()
            rendered_image = (rendered_image * 255).astype(np.uint8)
            rendered_image = rendered_image[..., ::-1]
            latest_frame = rendered_image

            with torch.no_grad():
                tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_by_js_view_matrix(view_matrix_fixed, xyz_scale=xyz_scale, big_view=True), xyz_scale=xyz_scale)
                tdgs_cam.image_width = 1536
                # tdgs_cam.image_height = 1024
                render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True)
            rendered_img = render_pkg['render']
            rendered_image = rendered_img.permute(1, 2, 0).detach().cpu().numpy()
            rendered_image = (rendered_image * 255).astype(np.uint8)
            rendered_image = rendered_image[..., ::-1]
            latest_viz = rendered_image
            if save:
                ToPILImage()(rendered_img).save(kf_gen.run_dir / 'rendered_img.png')
        except Exception as e:
            pass
        
        if latest_frame is not None and latest_viz is not None:
            ToPILImage()(latest_frame[..., ::-1]).save(kf_gen.run_dir / 'images' / 'latest_frame' / f'latest_frame_{curr_fid}.png')
            ToPILImage()(latest_viz[..., ::-1]).save(kf_gen.run_dir / 'images'  / 'latest_viz' / f'latest_viz_{curr_fid}.png')
            print(kf_gen.run_dir / 'images' / 'latest_frame' / f'latest_frame_{curr_fid}.png')
            print(kf_gen.run_dir / 'images' / 'latest_viz' / f'latest_viz_{curr_fid}.png')
            print("-" * 50)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    elif v.lower() in ("false", "0", "no"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--video_name",
        default="venice",
        help="Config path",
    )
    parser.add_argument(
        "--output_filename",
        default="debug",
        help="Config path",
    )
    parser.add_argument(
        "--target_angle",
        default=20,
        type=int,
        help="Config path",
    )
    parser.add_argument(
        "--gen_layer",
        default=False,
        type=str2bool,
        help="Config path",
    )
    parser.add_argument(
        "--gen_sky",
        default=False,
        type=str2bool,
        help="Config path",
    )
    parser.add_argument(
        "--base-config",
        default="./config/base-config.yaml",
        help="Config path",
    )
    parser.add_argument(
        "--example_config",
        default=f"config/more_examples/venice.yaml",
    )
    parser.add_argument(
        "--input_path",
        default=f"/hdd/xxx/world_crafter/data/wonderworld_videos/venice.mp4",
    )

    args = parser.parse_args()
    video_name = args.video_name
    args.example_config = f"config/more_examples/{video_name}.yaml"
    args.input_path = f"/hdd/xxx/world_crafter/data/wonderworld_videos/{video_name}.mp4"

    base_config = OmegaConf.load(args.base_config)
    example_config = OmegaConf.load(args.example_config)
    config = OmegaConf.merge(base_config, example_config)

    config.gen_sky = args.gen_sky
    config.gen_layer = args.gen_layer
    print("-" * 50)
    print("args.video_name: ", args.video_name)
    print("args.example_config: ", args.example_config)
    print("args.input_path: ", args.input_path)
    print("config.gen_sky: ", config.gen_sky )
    print("config.gen_layer: ", config.gen_layer)
    print("-" * 50)

    POSTMORTEM = config['debug']
    if POSTMORTEM:
        try:
            run(config, input_path=args.input_path, output_filename=args.output_filename, target_angle=args.target_angle)
        except Exception as e:
            print(e)
            import ipdb
            ipdb.post_mortem()
    else:
        run(config, input_path=args.input_path, output_filename=args.output_filename, target_angle=args.target_angle)

