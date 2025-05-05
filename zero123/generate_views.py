import torch
import numpy as np
import math
import argparse
import os
from PIL import Image
from omegaconf import OmegaConf
from einops import rearrange
from torchvision import transforms
from contextlib import nullcontext
from torch import autocast
from datetime import datetime

# Assuming these modules are accessible from the stable-diffusion directory
from ldm.util import instantiate_from_config, load_and_preprocess, create_carvekit_interface
from ldm.models.diffusion.ddim import DDIMSampler

# --- Model Loading ---
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

# --- Preprocessing ---
def preprocess_image(model_dict, input_im_path, preprocess_bg_removal):
    '''
    :param input_im_path (str): Path to the input image.
    :return input_im (H, W, 3) numpy array in [0, 1].
    '''
    input_im = Image.open(input_im_path)
    print('Input image size:', input_im.size)
    
    if preprocess_bg_removal:
        # Assumes carvekit interface is part of model_dict if bg removal is needed
        if 'carvekit' not in model_dict:
             raise ValueError("Background removal requested but carvekit model not loaded.")
        input_im = load_and_preprocess(model_dict['carvekit'], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        # (H, W, 3) array in [0, 1].
    else:
        input_im = input_im.convert("RGBA")
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].

        # Apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im
        input_im = input_im[:, :, 0:3] # Keep only RGB
        # (H, W, 3) array in [0, 1].

    print('Output image shape:', input_im.shape)
    return input_im

# --- Sampling ---
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
                              radius_m]) # Calc distance later
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
    parser = argparse.ArgumentParser(description='Generate multi-view images from a single input image.')
    parser.add_argument('--config', type=str, default='configs/sd-objaverse-finetune-c_concat-256.yaml', help='Path to the model config file.')
    parser.add_argument('--ckpt', type=str, default='105000.ckpt', help='Path to the model checkpoint file.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save the generated images.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., cuda:0, cpu).')
    parser.add_argument('--no_bg_removal', action='store_true', help='Disable background removal preprocessing.')
    
    # Generation parameters
    parser.add_argument('--polar', type=float, nargs='+', default=[0.], help='List of polar angles (degrees).')
    parser.add_argument('--azimuth', type=float, nargs='+', default=[0.], help='List of azimuth angles (degrees).')
    parser.add_argument('--radius', type=float, nargs='+', default=[0.], help='List of radius values (relative distance). Zero123 uses fixed radius so this might not have effect.') # TODO: Check if radius is used
    
    parser.add_argument('--scale', type=float, default=3.0, help='Classifier-free guidance scale.')
    parser.add_argument('--n_samples', type=int, default=1, help='Number of samples to generate per view.')
    parser.add_argument('--ddim_steps', type=int, default=50, help='Number of DDIM sampling steps.')
    parser.add_argument('--ddim_eta', type=float, default=1.0, help='DDIM eta parameter.')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'autocast'], help='Sampling precision.')
    parser.add_argument('--H', type=int, default=256, help='Image height.')
    parser.add_argument('--W', type=int, default=256, help='Image width.')
    
    args = parser.parse_args()

    # --- Get Timestamp for unique filenames ---
    # --- Create Unique Output Subdirectory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Saving outputs to: {run_output_dir}")

    # --- Sanity Checks ---
    if len(args.polar) != len(args.azimuth) or len(args.polar) != len(args.radius):
        raise ValueError("Lists for --polar, --azimuth, and --radius must have the same length.")
    if args.n_samples < 1:
        raise ValueError("--n_samples must be at least 1.")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Model ---
    device = torch.device(args.device)
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, device=device)
    
    # TODO: Add optional loading of carvekit if needed
    model_dict = {'turncam': model}
    if not args.no_bg_removal:
        print("Loading Carvekit...")
        # Need to import create_carvekit_interface if used
        # from ldm.util import create_carvekit_interface 
        model_dict['carvekit'] = create_carvekit_interface()

    # --- Preprocess Input Image ---
    print("Preprocessing image...")
    input_im_np = preprocess_image(model_dict, args.input, not args.no_bg_removal) # (H, W, 3) in [0, 1]
    
    # Convert to tensor and normalize
    input_im_tensor = transforms.ToTensor()(input_im_np).unsqueeze(0).to(device)
    input_im_tensor = input_im_tensor * 2 - 1 # Normalize to [-1, 1]
    input_im_tensor = transforms.functional.resize(input_im_tensor, [args.H, args.W])

    # --- Generate Views ---
    sampler = DDIMSampler(model)
    
    for i, (polar, azimuth, radius) in enumerate(zip(args.polar, args.azimuth, args.radius)):
        print(f"Generating view {i+1}/{len(args.polar)}: polar={polar}, azimuth={azimuth}, radius={radius}")
        
        output_tensor = sample_model(input_im_tensor, model, sampler, args.precision, args.H, args.W,
                                    args.ddim_steps, args.n_samples, args.scale, args.ddim_eta,
                                    polar, azimuth, radius) # Output is (n_samples, 3, H, W) in [0, 1]

        # --- Save Output ---
        for j, sample_tensor in enumerate(output_tensor):
            sample_np = 255.0 * rearrange(sample_tensor.cpu().numpy(), 'c h w -> h w c')
            output_im = Image.fromarray(sample_np.astype(np.uint8))
            
            # Construct filename with timestamp
            base_filename = f"view_{i+1:03d}_p{polar:+04.0f}_a{azimuth:+04.0f}_r{radius:.2f}"
            if args.n_samples > 1:
                filename = f"{base_filename}_sample{j+1:02d}.png"
            else:
                filename = f"{base_filename}.png"
            
            output_path = os.path.join(run_output_dir, filename)
            output_im.save(output_path)
            print(f"Saved: {output_path}")

    print("Done.")


if __name__ == "__main__":
    main() 