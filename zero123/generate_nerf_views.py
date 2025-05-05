import torch
import numpy as np
import math
import argparse
import os
import json
from PIL import Image
from omegaconf import OmegaConf
from einops import rearrange
from torchvision import transforms
from contextlib import nullcontext
from torch import autocast
from datetime import datetime

# Assuming these modules are accessible
from ldm.util import instantiate_from_config, load_and_preprocess, create_carvekit_interface
from ldm.models.diffusion.ddim import DDIMSampler

# --- Camera Pose Utilities ---

# Returns camera-to-world matrix
def pose_spherical(azimuth_deg, polar_deg, radius):
    """
    Calculate 4x4 camera-to-world matrix from spherical coordinates.
    Assumes camera always looks at the origin.
    Args:
        azimuth_deg (float): Azimuth angle (rotation around Y-axis) in degrees.
        polar_deg (float): Polar angle (elevation from XZ-plane) in degrees.
        radius (float): Distance from the origin.
    Returns:
        np.ndarray: 4x4 camera-to-world matrix.

    Coordinate System:
        +X right, +Y up, +Z towards viewer (OpenGL convention).
        Azimuth=0, Polar=90 -> Camera at (0, 0, R), looking towards -Z.
        Azimuth=90, Polar=90 -> Camera at (R, 0, 0), looking towards -X.
        Polar=0 -> Camera at (0, R, 0), looking towards -Y (from top).
    """
    phi = np.radians(azimuth_deg) # Azimuth
    theta = np.radians(90 - polar_deg) # Convert polar angle to inclination

    # Camera position
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.cos(theta)
    z = radius * np.sin(theta) * np.sin(phi)
    camera_pos = np.array([x, y, z])

    # Camera orientation (look at origin)
    forward = -camera_pos / np.linalg.norm(camera_pos) # Normalized vector from camera to origin

    # Calculate right vector (assuming +Y is up)
    world_up = np.array([0, 1, 0])
    # Ensure world_up is not parallel to 'forward'
    if np.abs(np.dot(forward, world_up)) > 0.999:
        world_up = np.array([0, 0, 1]) # Use +Z as up if looking straight down/up
        if np.abs(np.dot(forward, world_up)) > 0.999: # If also parallel to Z
             world_up = np.array([1, 0, 0]) # Use +X

    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)

    # Calculate camera up vector
    camera_up = np.cross(right, forward) # forward and right are already normalized and orthogonal

    # Construct rotation matrix (camera-to-world basis vectors)
    rotation_matrix = np.stack((right, camera_up, -forward), axis=1) # View direction is -Z in camera space

    # Construct transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = camera_pos

    return transform_matrix


# --- Model Loading (same as generate_views.py) ---
def load_model_from_config(config, ckpt, device, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=device)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model

# --- Preprocessing (same as generate_views.py) ---
def preprocess_image(model_dict, input_im_path, preprocess_bg_removal):
    '''
    :param input_im_path (str): Path to the input image.
    :return input_im (H, W, 3) numpy array in [0, 1].
    '''
    input_im = Image.open(input_im_path)
    print('Input image size:', input_im.size)

    if preprocess_bg_removal:
        if 'carvekit' not in model_dict:
             raise ValueError("Background removal requested but carvekit model not loaded. Use --no_bg_removal or ensure carvekit is available.")
        print("Performing background removal...")
        input_im = load_and_preprocess(model_dict['carvekit'], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
    else:
        print("Skipping background removal...")
        # Ensure 3 channels if no BG removal
        if input_im.mode == 'RGBA':
             # Simple white background composite
            input_im = input_im.convert("RGBA")
            input_im_arr = np.asarray(input_im, dtype=np.float32) / 255.0
            alpha = input_im_arr[:, :, 3:4]
            white_im = np.ones_like(input_im_arr)
            composited = alpha * input_im_arr + (1.0 - alpha) * white_im
            input_im = Image.fromarray((composited[:, :, :3] * 255).astype(np.uint8))
        else:
             input_im = input_im.convert("RGB")

        # Resize after potential conversion
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0

    print('Preprocessed image shape:', input_im.shape)
    return input_im

# --- Sampling (same as generate_views.py, radius handled outside) ---
@torch.no_grad()
def sample_model(input_im_tensor, model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, polar_deg, azimuth_deg, radius_m):
    '''
    :param input_im_tensor: (1, 3, H, W) tensor in [-1, 1].
    '''
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im_tensor).tile(n_samples, 1, 1)

            # Camera pose calculation
            T = torch.tensor([math.radians(polar_deg),
                              math.sin(math.radians(azimuth_deg)),
                              math.cos(math.radians(azimuth_deg)),
                              radius_m]) # Pass radius directly
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)

            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage(input_im_tensor).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]

            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()

# --- Main Script ---
def main():
    parser = argparse.ArgumentParser(description='Generate multi-view images and poses suitable for NeRF training.')
    parser.add_argument('--config', type=str, default='configs/sd-objaverse-finetune-c_concat-256.yaml', help='Path to the model config file.')
    parser.add_argument('--ckpt', type=str, default='105000.ckpt', help='Path to the model checkpoint file.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--output_dir', type=str, default='nerf_dataset_output', help='Base directory to save the generated images and poses.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., cuda:0, cpu).')
    parser.add_argument('--no_bg_removal', action='store_true', help='Disable background removal preprocessing (recommended for NeRF).')

    # NeRF-specific generation parameters
    parser.add_argument('--num_views', type=int, default=100, help='Number of views to generate.')
    parser.add_argument('--radius', type=float, default=0.5, help='Camera distance from the origin.') # Zero123 uses relative radius; NeRF needs absolute. Adjust as needed.
    parser.add_argument('--upper_views_only', action='store_true', help='Generate views only from the upper hemisphere (polar > 0).')
    parser.add_argument('--focal_length', type=float, default=None, help='Focal length in pixels. If None, estimated from image width.')

    # Standard generation parameters
    parser.add_argument('--scale', type=float, default=3.0, help='Classifier-free guidance scale.')
    # parser.add_argument('--n_samples', type=int, default=1, help='Number of samples per view (set to 1 for NeRF).') # Force n_samples=1
    parser.add_argument('--ddim_steps', type=int, default=50, help='Number of DDIM sampling steps.')
    parser.add_argument('--ddim_eta', type=float, default=1.0, help='DDIM eta parameter.')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'autocast'], help='Sampling precision.')
    parser.add_argument('--H', type=int, default=256, help='Image height.')
    parser.add_argument('--W', type=int, default=256, help='Image width.')

    args = parser.parse_args()
    n_samples = 1 # NeRF typically uses one sample per pose

    # --- Create Unique Output Subdirectory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.output_dir, timestamp)
    images_output_dir = os.path.join(run_output_dir, "images")
    os.makedirs(images_output_dir, exist_ok=True)
    print(f"Saving outputs to: {run_output_dir}")

    # --- Sanity Checks ---
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")
    if args.num_views < 1:
        raise ValueError("--num_views must be at least 1.")

    # --- Load Model ---
    device = torch.device(args.device)
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, device=device)

    model_dict = {'turncam': model}
    if not args.no_bg_removal:
        print("Loading Carvekit for background removal...")
        model_dict['carvekit'] = create_carvekit_interface()

    # --- Preprocess Input Image ---
    print("Preprocessing input image...")
    input_im_np = preprocess_image(model_dict, args.input, not args.no_bg_removal) # (H, W, 3) in [0, 1]

    # Convert to tensor and normalize
    input_im_tensor = transforms.ToTensor()(input_im_np).unsqueeze(0).to(device)
    input_im_tensor = input_im_tensor * 2 - 1 # Normalize to [-1, 1]
    # Resize the input image conditioning to the target generation size
    input_im_tensor = transforms.functional.resize(input_im_tensor, [args.H, args.W])


    # --- Generate Views and Poses ---
    sampler = DDIMSampler(model)
    all_poses = []
    image_paths = []

    # Generate viewpoints
    # Simple spherical distribution for now
    polar_angles_deg = []
    azimuth_angles_deg = []

    if args.upper_views_only:
        # Sample elevations from just above horizon to top
        polar_angles = np.linspace(np.pi / 18, np.pi / 2, 4, endpoint=True) # e.g., 10, 36, 63, 90 degrees
    else:
        # Sample elevations from bottom to top
        polar_angles = np.linspace(-np.pi / 2, np.pi / 2, 7, endpoint=True) # e.g., -90, -60, -30, 0, 30, 60, 90

    views_per_elevation = args.num_views // len(polar_angles) + 1

    for polar in polar_angles:
        polar_deg = np.degrees(polar)
        # Vary azimuth count based on elevation to avoid crowding at poles
        num_azi = int(np.ceil(views_per_elevation * np.sin(polar + np.pi/2))) # More samples near equator
        num_azi = max(1, num_azi) # Ensure at least one sample
        azimuths_rad = np.linspace(0, 2 * np.pi, num_azi, endpoint=False)
        for azimuth in azimuths_rad:
            azimuth_deg = np.degrees(azimuth)
            polar_angles_deg.append(polar_deg)
            azimuth_angles_deg.append(azimuth_deg)

    # Ensure we have roughly the desired number of views
    # view_indices = np.random.choice(len(polar_angles_deg), args.num_views, replace=False)
    # Check if enough viewpoints were generated
    num_generated = len(polar_angles_deg)
    if num_generated < args.num_views:
        print(f"Warning: Generated {num_generated} unique viewpoints, which is less than the requested {args.num_views}. Using all generated viewpoints.")
        view_indices = np.arange(num_generated)
    else:
        # Sample the exact number requested without replacement
        view_indices = np.random.choice(num_generated, args.num_views, replace=False)

    polar_angles_deg = np.array(polar_angles_deg)[view_indices]
    azimuth_angles_deg = np.array(azimuth_angles_deg)[view_indices]


    print(f"Generating {len(view_indices)} views...")
    for i, (polar, azimuth) in enumerate(zip(polar_angles_deg, azimuth_angles_deg)):
        view_idx = i # Use sequential index for filename
        print(f"Generating view {view_idx+1}/{len(view_indices)}: polar={polar:.1f}, azimuth={azimuth:.1f}")

        # Use the fixed radius provided
        current_radius = args.radius

        output_tensor = sample_model(input_im_tensor, model, sampler, args.precision, args.H, args.W,
                                    args.ddim_steps, n_samples, args.scale, args.ddim_eta,
                                    polar, azimuth, current_radius) # Pass absolute radius

        # --- Save Output Image ---
        # NeRF expects only one sample per pose
        sample_tensor = output_tensor[0] # Get the first (only) sample
        sample_np = 255.0 * rearrange(sample_tensor.cpu().numpy(), 'c h w -> h w c')
        output_im = Image.fromarray(sample_np.astype(np.uint8))

        filename = f"{view_idx:04d}.png"
        output_image_path = os.path.join(images_output_dir, filename)
        output_im.save(output_image_path)

        # --- Calculate and Store Pose ---
        transform_matrix = pose_spherical(azimuth, polar, current_radius)
        rel_image_path = os.path.join("images", filename)
        frame_data = {
            "file_path": rel_image_path,
            "transform_matrix": transform_matrix.tolist() # Convert numpy array to list for JSON
        }
        all_poses.append(frame_data)
        image_paths.append(rel_image_path) # Keep track if needed


    # --- Write transforms.json ---
    if args.focal_length is None:
        # Estimate focal length: Assume fov = 60 deg -> f = W / (2 * tan(fov/2))
        fov_rad = np.radians(60)
        focal_pixels = 0.5 * args.W / np.tan(0.5 * fov_rad)
        print(f"Estimating focal length: {focal_pixels:.2f} pixels (for W={args.W})")
    else:
        focal_pixels = args.focal_length

    # Assume principal point is image center
    cx = args.W / 2.0
    cy = args.H / 2.0

    # Calculate camera_angle_x from focal length
    camera_angle_x = 2 * np.arctan(args.W / (2 * focal_pixels))

    transforms_data = {
        "camera_angle_x": camera_angle_x,
        "fl_x": focal_pixels,
        "fl_y": focal_pixels, # Assume square pixels
        "cx": cx,
        "cy": cy,
        "w": args.W,
        "h": args.H,
        "frames": all_poses
    }

    transforms_path = os.path.join(run_output_dir, "transforms.json")
    with open(transforms_path, 'w') as f:
        json.dump(transforms_data, f, indent=4)

    print(f"Saved {len(all_poses)} images to {images_output_dir}")
    print(f"Saved poses and intrinsics to {transforms_path}")
    print("Done.")


if __name__ == "__main__":
    main() 