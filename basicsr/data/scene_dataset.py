import os
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import paired_random_crop, random_augmentation
from basicsr.utils import img2tensor, padding
import numpy as np
import cv2
import random


class Dataset_PairedScene(data.Dataset):
    """Paired image dataset for image restoration.
    Adapted for CL3AN dataset structure:
    - GT images in 'gt' folder: {SceneID}_GT.png
    - Input images in 'input' folder: {SceneID}_{Variation}_IN.png
    """

    def __init__(self, opt):
        super(Dataset_PairedScene, self).__init__()
        self.opt = opt
        self.root = opt["baseroot"]
        self.phase = opt["phase"]
        self.lphase = opt["lphase"]
        self.max_samples = opt.get("max_samples", None)

        self.mean = opt["mean"] if "mean" in opt else None
        self.std = opt["std"] if "std" in opt else None

        self.max_size = opt.get("maxsize", None)
        self.square = opt.get("square", False)

        gt_dir = os.path.join(self.root, self.lphase, "gt")
        input_dir = os.path.join(self.root, self.lphase, "input")

        # Fallback for case sensitivity if lowercase not found
        if not os.path.exists(gt_dir) and os.path.exists(
            os.path.join(self.root, self.lphase, "GT")
        ):
            gt_dir = os.path.join(self.root, self.lphase, "GT")
        if not os.path.exists(input_dir) and os.path.exists(
            os.path.join(self.root, self.lphase, "IN_SH")
        ):
            # If strictly following Retinexformer structure with IN_SH/IN_CR, this script might need more adaptation.
            # But here we target CL3AN structure.
            pass

        if not os.path.exists(gt_dir):
            raise FileNotFoundError(f"GT directory not found: {gt_dir}")
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        self.img_index = {}

        # Load GTs
        gts = sorted([f for f in os.listdir(gt_dir) if f.endswith("_GT.png")])

        if self.max_samples:
            gts = gts[: self.max_samples]

        for f in gts:
            # Extract scene ID. naming convention: XX_GT.png -> XX
            scene_id = f.split("_")[0]
            self.img_index[scene_id] = {"gt": os.path.join(gt_dir, f), "lq": []}

        # Load Inputs
        # Naming convention: XX_YY_IN.png -> corresponds to XX_GT.png
        # We scan all files in input_dir
        all_inputs = sorted([f for f in os.listdir(input_dir) if f.endswith("_IN.png")])

        for f in all_inputs:
            scene_id = f.split("_")[0]
            if scene_id in self.img_index:
                self.img_index[scene_id]["lq"].append(os.path.join(input_dir, f))

        # Remove GTs with no inputs
        to_remove = []
        for k, v in self.img_index.items():
            if not v["lq"]:
                print(f"Warning: No inputs found for GT {k}")
                to_remove.append(k)
        for k in to_remove:
            del self.img_index[k]

        self.keys = sorted(list(self.img_index.keys()))
        print(f"Loaded {len(self.keys)} paired scenes for phase {self.phase}")

        # val inputs
        self.val_inputs = []
        if self.opt["phase"] != "train":
            for k in self.keys:
                gt_path = self.img_index[k]["gt"]
                for lq_path in self.img_index[k]["lq"]:
                    self.val_inputs.append((gt_path, lq_path))
            # Limit val samples if max_samples is set (applied to pairs)
            # Note: max_samples in __init__ for val usually means max scenes or max pairs?
            # Existing code used max_samples to slice GTs. Here we already sliced GTs.
            # So val_inputs are already limited by scenes.
            pass

        self.geometric_augs = opt.get("geometric_augs", False)

    def imread_norm(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found or empty: {img_path}")

        if self.max_size is not None:
            th = (self.max_size // 32) * 32
            if not self.square:
                tw = (int(th * 4 / 3) // 32) * 32
            else:
                tw = th

            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LANCZOS4)

        img = img.astype(np.float32) / 255.0
        return img

    def __len__(self):
        if self.opt["phase"] == "train":
            return len(self.keys)
        else:
            return len(self.val_inputs)

    def __getitem__(self, index):
        scale = self.opt["scale"]

        if self.opt["phase"] == "train":
            index = index % len(self.keys)
            key = self.keys[index]
            gt_path = self.img_index[key]["gt"]

            # Randomly select one LQ image for this scene
            if len(self.img_index[key]["lq"]) > 0:
                lq_path = random.choice(self.img_index[key]["lq"])
            else:
                raise FileNotFoundError(f"No LQ images for {key}")
        else:
            gt_path, lq_path = self.val_inputs[index]

        try:
            img_gt = self.imread_norm(gt_path)
            img_lq = self.imread_norm(lq_path)
        except Exception as e:
            print(f"Error loading {lq_path} or {gt_path}: {e}")
            raise e

        # augmentation for training
        if self.opt["phase"] == "train":
            h, w, _ = img_gt.shape

            # padding
            img_gt, img_lq = padding(img_gt, img_lq, int(0.8 * h))

            # random crop
            img_gt, img_lq = paired_random_crop(
                img_gt, img_lq, int(0.8 * h), scale, gt_path
            )

            gt_size = self.opt["gt_size"]
            ch, cw, _ = img_gt.shape

            if gt_size < ch:
                img_gt = cv2.resize(
                    img_gt, (gt_size, gt_size), interpolation=cv2.INTER_LANCZOS4
                )
                img_lq = cv2.resize(
                    img_lq, (gt_size, gt_size), interpolation=cv2.INTER_LANCZOS4
                )

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {"lq": img_lq, "gt": img_gt, "lq_path": lq_path, "gt_path": gt_path}


if __name__ == "__main__":
    # Simple test
    pass
