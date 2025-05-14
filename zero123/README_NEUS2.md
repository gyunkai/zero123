# NeuS2 Dataset Generator

This script generates multi-view images and a `transforms.json` file from a single input image using the Zero123 model. The output follows the NeuS2 data convention for novel view synthesis and 3D reconstruction.

## Requirements

The script depends on the Zero123 model and its dependencies. Make sure you have installed all the required packages from the Zero123 repository.

## Usage

```bash
python generate_neus2_dataset.py --input path/to/input_image.jpg --output_dir neus2_dataset
```

### Common Options

- `--input`: Path to the input image (required)
- `--output_dir`: Directory to save the dataset (default: "neus2_dataset")
- `--device`: Device to use (default: "cuda:0")
- `--no_bg_removal`: Disable background removal preprocessing (flag)

### Dataset Generation Parameters

- `--num_views`: Number of views to generate (default: 50)
- `--width`: Output image width (default: 512)
- `--height`: Output image height (default: 512)
- `--radius`: Camera distance from origin in meters (default: 4.0)
- `--focal_length`: Focal length in pixels (default: 800.0)
- `--upper_views_only`: Generate views only from upper hemisphere (flag)
- `--mask_brightness`: Optional brightness threshold for creating masks (default: None)

### Zero123 Parameters

- `--config`: Path to model config file (default: "configs/sd-objaverse-finetune-c_concat-256.yaml")
- `--ckpt`: Path to model checkpoint file (default: "105000.ckpt")
- `--scale`: Classifier-free guidance scale (default: 3.0)
- `--ddim_steps`: Number of DDIM sampling steps (default: 50)
- `--ddim_eta`: DDIM eta parameter (default: 1.0)
- `--precision`: Sampling precision (default: "fp32", choices: ["fp32", "autocast"])

## Output Format

The script creates a timestamped folder in the specified output directory containing:

1. An `images` folder with the generated view images
2. A `transforms.json` file following the NeuS2 convention:

```json
{
    "from_na": true,
    "w": 512,
    "h": 512,
    "aabb_scale": 1.0,
    "scale": 0.5,
    "offset": [0.5, 0.5, 0.5],
    "frames": [
        {
            "file_path": "images/000000.png",
            "transform_matrix": [...],
            "intrinsic_matrix": [...]
        },
        ...
    ]
}
```

## Examples

Generate 100 views with a radius of 5 meters and masks:

```bash
python generate_neus2_dataset.py --input object.png --num_views 100 --radius 5.0 --mask_brightness 200
```

Generate only upper hemisphere views:

```bash
python generate_neus2_dataset.py --input object.png --upper_views_only
```

Generate higher resolution images:

```bash
python generate_neus2_dataset.py --input object.png --width 1024 --height 1024 --focal_length 1600
``` 