import kornia
import numpy as np
import torch
from matplotlib import cm
from torchvision.io import write_video
import imageio
from decord import VideoReader, cpu
import torchvision
from tqdm import tqdm
import imageio.v3 as iio

import os
import cv2
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from utils.colmap import get_colmap_camera_params
import torch.nn.functional as F
from PIL import Image



def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(0.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


class LatentStorer:
    def __init__(self):
        self.latent = None

    def __call__(self, i, t, latent):
        self.latent = latent


def sobel_filter(disp, mode="sobel", beta=10.0):
    sobel_grad = kornia.filters.spatial_gradient(disp, mode=mode, normalized=False)
    sobel_mag = torch.sqrt(sobel_grad[:, :, 0, Ellipsis] ** 2 + sobel_grad[:, :, 1, Ellipsis] ** 2)
    alpha = torch.exp(-1.0 * beta * sobel_mag).detach()

    return alpha


def apply_colormap(image, cmap="viridis"):
    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return colormap[image_long[..., 0]]


def apply_depth_colormap(
    depth,
    near_plane=None,
    far_plane=None,
    cmap="viridis",
):
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)

    colored_image = apply_colormap(depth, cmap=cmap)

    return colored_image


def save_video(video, path, fps=10, save_gif=True):
    video = video.permute(0, 2, 3, 1)
    video_codec = "libx264"
    video_options = {
        "crf": "30",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
        "preset": "slow",
    }
    write_video(str(path), video, fps=fps, video_codec=video_codec, options=video_options)
    if not save_gif:
        return
    video_np = video.cpu().numpy()
    gif_path = str(path).replace('.mp4', '.gif')
    imageio.mimsave(gif_path, video_np, duration=1000/fps, loop=0)
    

def read_video_frames(video_path, process_length, stride, max_res=[512, 512], dataset="open"):
    # if dataset == "open":
    #     print("==> processing video: ", video_path)
    #     vid = VideoReader(video_path, ctx=cpu(0))
    #     print("==> original video shape: ", (len(vid), *vid.get_batch([0]).shape[1:]))
    #     # original_height, original_width = vid.get_batch([0]).shape[1:3]
    #     # height = round(original_height / 64) * 64
    #     # width = round(original_width / 64) * 64
    #     # if max(height, width) > max_res:
    #     #     scale = max_res / max(original_height, original_width)
    #     #     height = round(original_height * scale / 64) * 64
    #     #     width = round(original_width * scale / 64) * 64

    #     # FIXME: hard coded
    #     width = 1024
    #     height = 576
    
    width = 512
    height = 512

    vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

    frames_idx = list(range(0, len(vid), stride))
    print(
        f"==> downsampled shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}, with stride: {stride}"
    )
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
    print(
        f"==> final processing shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}"
    )
    frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0

    return frames


def load_frames(img_paths):
    frames = []
    for img_path in tqdm(img_paths, desc="Loading frames"):
        img = iio.imread(img_path).astype(np.float32) / 255.0
        frames.append(img[..., :3])  
    frames = np.stack(frames)
    print("type(frames), dtype:", type(frames), frames.dtype, np.max(frames), np.min(frames), frames.shape)

    return frames

def save_video_2(data, images_path, folder=None, fps=8):
    if isinstance(data, np.ndarray):
        tensor_data = (torch.from_numpy(data) * 255).to(torch.uint8)
    elif isinstance(data, torch.Tensor):
        tensor_data = (data.detach().cpu() * 255).to(torch.uint8)
    elif isinstance(data, list):
        folder = [folder] * len(data)
        images = [
            np.array(Image.open(os.path.join(folder_name, path)))
            for folder_name, path in zip(folder, data)
        ]
        stacked_images = np.stack(images, axis=0)
        tensor_data = torch.from_numpy(stacked_images).to(torch.uint8)
    torchvision.io.write_video(
        images_path, tensor_data, fps=fps, video_codec='h264', options={'crf': '5'}
    )


def save_frames(data, frame_dir, name_list=None):
    """
    Save a sequence of image frames from a numpy array or torch tensor.

    Args:
        data (np.ndarray or torch.Tensor): Image sequence of shape (T, H, W, C) in [0, 1] range.
        frame_dir (str): Directory to save individual frames.
        name_list (list of str): List of filenames (without extension) for each frame.
    """
    if isinstance(data, np.ndarray):
        tensor_data = (torch.from_numpy(data) * 255).to(torch.uint8)
    elif isinstance(data, torch.Tensor):
        tensor_data = (data.detach().cpu() * 255).to(torch.uint8)
    else:
        raise ValueError("Unsupported data type. Expected np.ndarray or torch.Tensor.")

    assert name_list is not None, "name_list must be provided when saving frames."
    assert len(name_list) == tensor_data.shape[0], "Length of name_list must match number of frames."
    assert frame_dir is not None, "frame_dir must be specified."

    os.makedirs(frame_dir, exist_ok=True)

    for i, name in enumerate(name_list):
        img = Image.fromarray(tensor_data[i].numpy())
        img.save(os.path.join(frame_dir, f"{name}.png"))


def resize(img, size, mode='bilinear') -> torch.Tensor:
    """
    Center crop and resize a 5D video tensor [B, C, T, H, W] to the target size.
    
    Args:
        img (torch.Tensor): Input tensor of shape [B, C, T, H, W].
        target_h (int): Target height after resizing.
        target_w (int): Target width after resizing.

    Returns:
        torch.Tensor: Output tensor of shape [B, C, T, target_h, target_w].
    """
    B, C, T, H, W = img.shape

    # Flatten temporal dimension: [B*T, C, H, W]
    img = img.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)

    if mode == "nearest":
        img = F.interpolate(img, size=size, mode=mode)
    else:
        img = F.interpolate(img, size=size, mode=mode, align_corners=False)

    img = img.reshape(B, T, C, size[0], size[1]).permute(0, 2, 1, 3, 4)

    return img



def pad_batch_to_multiple_numpy(frames: np.ndarray, multiple=32, mode='reflect'):
    """
    Pads a batch of frames (N, H, W, C) to the nearest multiple of `multiple`.

    Args:
        frames: NumPy array of shape (N, H, W, C)
        multiple: multiple to pad to (e.g., 32)
        mode: 'reflect' or 'constant'

    Returns:
        padded_frames: (N, H_pad, W_pad, C)
        pad_info: (top, bottom, left, right)
    """
    n, h, w, c = frames.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded = np.pad(
        frames,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode=mode
    )

    return padded, [pad_top, pad_bottom, pad_left, pad_right]


def crop_batch_to_original_numpy(padded_frames: np.ndarray, pad_info):
    """
    Crops a batch of padded frames (N, H_pad, W_pad, C) back to original size.
    """
    pad_top, pad_bottom, pad_left, pad_right = pad_info
    return padded_frames[:, :, pad_top:-pad_bottom or None, pad_left:-pad_right or None]

def resize_numpy_min_side_batch(frames: np.ndarray, target_min_side: int = 512) -> list:
    resized = []
    for frame in frames:
        h, w = frame.shape[:2]
        scale = target_min_side / min(h, w)
        new_h, new_w = round(h * scale), round(w * scale)
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        resized.append(resized_frame.astype(np.float32))  
    return np.stack(resized) 


def resize_tensor_min_side_batch(frames: torch.Tensor, target_min_side: int = 512) -> list:
    resized = []
    for frame in frames:  # shape: [1, H, W]
        _, h, w = frame.shape
        scale = target_min_side / min(h, w)
        new_h, new_w = round(h * scale), round(w * scale)
        resized_frame = TF.resize(frame, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
        resized.append(resized_frame.to(torch.float32))  # 保持 float32
    return torch.stack(resized)


def resize_numpy_batch(frames: np.ndarray, w: int, h: int) -> np.ndarray:
    resized = [cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
               for frame in frames]
    return np.stack(resized)
    

def resize_tensor_batch(frames: torch.Tensor, w: int, h: int) -> torch.Tensor:
    resized = [TF.resize(frame, [h, w], interpolation=InterpolationMode.BILINEAR)
               for frame in frames]
    return torch.stack(resized).to(torch.float32)


def get_K_c2w_w2c(data_dir, frame_names, device):
    Ks, w2cs = get_colmap_camera_params(
        os.path.join(data_dir, "flow3d_preprocessed/colmap/sparse/"),
        [name + ".png" for name in frame_names],
    )

    Ks = Ks[:, :3, :3]
    scale = np.load(os.path.join(data_dir, "flow3d_preprocessed/colmap/scale.npy")).item()
    c2ws = np.linalg.inv(w2cs)
    c2ws[:, :3, -1] *= scale
    w2cs = np.linalg.inv(c2ws)
    
    Ks = torch.from_numpy(Ks).float().to(device)
    c2ws = torch.from_numpy(c2ws).float().to(device)
    w2cs = torch.from_numpy(w2cs).float().to(device)

    return Ks, c2ws, w2cs

def load_frames_by_names(data_dir, frame_names):
    # frames = np.array([
    #     iio.imread(os.path.join(data_dir, "rgb/1x", f"{name}.png"))
    #     for name in tqdm(frame_names, desc="Loading frames")
    # ])
    # print("before type(frames), frames.dtype, np.max(frames), np.min(frames), frames.shape: ", type(frames), frames.dtype, np.max(frames), np.min(frames), frames.shape)
    # frames = frames[..., :3] / 255.0
    # frames = frames.astype(np.float32)
    # print("after type(frames), frames.dtype, np.max(frames), np.min(frames), frames.shape: ", type(frames), frames.dtype, np.max(frames), np.min(frames), frames.shape)

    frames = []
    for name in tqdm(frame_names, desc="Loading frames"):
        img = iio.imread(os.path.join(data_dir, "rgb/1x", f"{name}.png")).astype(np.float32) / 255.0
        frames.append(img[..., :3])  
    frames = np.stack(frames)
    print("type(frames), dtype:", type(frames), frames.dtype, np.max(frames), np.min(frames), frames.shape)

    return frames