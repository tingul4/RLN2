import argparse
import cv2
import glob
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import zipfile
from datetime import datetime
from basicsr.models import create_model
from basicsr.utils.options import parse
from basicsr.utils import img2tensor, tensor2img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opt",
        type=str,
        default="Options/RLN2-Lf.yml",
        help="Path to option YAML file.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="checkpoints/RLN2-Lf/best_psnr_20.51_5600.pth",
        help="Path to weights.",
    )
    parser.add_argument(
        "--input_dir", type=str, default="datasets/val", help="Input directory."
    )
    parser.add_argument(
        "--output_root", type=str, default="RLN2_results", help="Output root directory."
    )
    args = parser.parse_args()

    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_dir = os.path.join(args.output_root, f"submission_{timestamp}")
    os.makedirs(submission_dir, exist_ok=True)

    # Load options
    opt = parse(args.opt, is_train=False)
    opt["dist"] = False
    opt["path"]["pretrain_network_g"] = args.weights
    opt["path"]["strict_load_g"] = True

    # Initialize model
    print(f"Loading model: {opt['network_g']['type']}")
    model = create_model(opt)
    net_g = model.net_g
    net_g.eval()
    net_g.cuda()

    # Get input files
    img_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
    print(f"Found {len(img_paths)} images in {args.input_dir}")

    total_time = 0
    img_count = 0

    with torch.no_grad():
        for img_path in img_paths:
            img_name = os.path.basename(img_path)

            # Load and preprocess
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = img.astype(np.float32) / 255.0
            img_t = img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(0).cuda()

            # Pad if needed (factor 32 for CC36)
            h, w = img_t.shape[2:]
            factor = 32
            pad_h = (factor - h % factor) % factor
            pad_w = (factor - w % factor) % factor
            img_t = F.pad(img_t, (0, pad_w, 0, pad_h), mode="reflect")

            # Inference
            torch.cuda.synchronize()
            start_time = time.time()

            output_t = net_g(img_t)

            torch.cuda.synchronize()
            elapsed = time.time() - start_time

            if img_count > 0:  # Skip first one for warmup warming
                total_time += elapsed

            # Postprocess
            output_t = output_t[:, :, :h, :w]
            output = tensor2img(
                output_t, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)
            )

            # Save
            save_path = os.path.join(submission_dir, img_name)
            cv2.imwrite(save_path, output)

            img_count += 1
            if img_count % 10 == 0:
                print(f"Processed {img_count}/{len(img_paths)}...")

    # Calculate average time
    avg_runtime = total_time / (img_count - 1) if img_count > 1 else total_time
    print(f"Average runtime per image: {avg_runtime:.4f}s")

    # Generate readme.txt
    readme_path = os.path.join(submission_dir, "readme.txt")
    with open(readme_path, "w") as f:
        f.write(f"runtime per image [s] : {avg_runtime:.4f}\n")
        f.write(f"CPU[1] / GPU[0] : 0\n")  # We used GPU
        f.write(f"Extra Data [1] / No Extra Data [0] : 0\n")  # Adjust if needed
        f.write(
            f"Other description : RLN2 model inference code. Output generated on {timestamp}.\n"
        )

    # Zip the contents
    zip_path = os.path.join(args.output_root, f"submission_{timestamp}.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for root, _, files in os.walk(submission_dir):
            for file in files:
                zipf.write(os.path.join(root, file), file)

    print(f"Submission results saved to: {submission_dir}")
    print(f"Submission zip created: {zip_path}")


if __name__ == "__main__":
    main()
