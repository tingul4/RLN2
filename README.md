# RLN2

To download the pretrained checkpoints and other project resources use the following [Google Drive link](https://drive.google.com/file/d/16aJAlqH490wwccT1ZzRkayOut3JjUZcF/view?usp=drive_link). 

Unzip the resources and move the folder to the project root dir.

# Quick Start (Environment Setup)

We recommend using [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management.

### 1. Install uv (If not already installed)
```bash
curl -LsSf https://astral-sh.uv.run/install.sh | sh
```

### 2. Create and Activate Virtual Environment
```bash
uv venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
uv pip install -r requirements.txt
```

### 4. Install Local Package (Editable Mode)
This step is **required** to ensure the `basicsr` module can be imported correctly from the project root.
```bash
uv pip install -e .
```

# Inference & Submission
To generate a Codabench-compliant submission:
```bash
python basicsr/inference.py --opt Options/RLN2-Lf.yml --weights checkpoints/RLN2-Lf/best_psnr_20.51_5600.pth
```
The results will be saved in `RLN2_results/submission_YYYYMMDD_HHMMSS.zip`.


# Updates
So far we have published a minimal set of resources for our project. 

Follow this repository for future updates. 


# Checkpoints
To download public weights for pretrained models use the following [Google Drive link](https://drive.google.com/file/d/15FXyfQiedxmvBssuDQaC62s_nKrjF3X8/view?usp=drive_link). 

Unzip the archive and move the folder inside to the project root dir. 

# AMBIENT6K Resources

Refer to the [IFBLEND](https://github.com/fvasluianu97/IFBlend) repository to trian and test RLN2 on AMBIENT6K. 

# CL3AN

The data is provided as 24 MP images, the default resolution of the camera.

The data is available through Google Drive, with a size of approx. 150 GB. 
* [Training data](https://drive.google.com/drive/folders/1QCV2Cfc1XpXw8XOoQR533OV1y1R0a7fz?usp=sharing)
* [Testing data](https://drive.google.com/drive/folders/1CKVX9z09lD4W5jxlEqWVe4eRz3-qCxhF?usp=drive_link)

# Issues
In case of questions or other requests, feel free to post it in the Issues section. 

Alternatively, drop us an [email](mailto:florin-alexandru.vasluianu@uni-wuerzburg.de).

# Acknowledgements
This repository borrows resources from [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer). We thank the authors for their efforts.

# License
Copyright (c) 2025 Computer Vision Lab, University of Wurzburg

Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

The code is released for academic research use only. For commercial use, please contact Computer Vision Lab, University of Wurzburg.
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.

