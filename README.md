# dinov23.cpp.py  
A Python ggml implementation of DINOv2 and DINOv3, heavily inspired by [dinov2.cpp](https://github.com/lavaman131/dinov2.cpp).  


## Table of Contents  
- [Dependencies](#dependencies)  
- [Setup](#setup)  
- [CLI Usage](#cli-usage)  
- [Program Usage](#program-usage)  
- [Model Weights](#model-weights)  
- [Notes](#notes)  


## Dependencies  
- My ggml-python [fork](https://github.com/ZisIsNotZis/ggml-python) (original is outdated)
- Python 3.12+ (for newer typing syntax)  
- Pillow
- scikit-learn  


## Setup  
```bash
pip install dinov23-cpp-py git+https://github.com/ZisIsNotZis/ggml-python.git
```  


## CLI Usage  

### DINOv2  
Run feature extraction on images with:  
```bash
dino2.py <img>...
```  

**Output Files**:  
- `<img>dino2<C>f`: Class vector (global feature, `C` = number of channels, e.g., 384 for small models).  
- `<img>dino2.<H>,<W>,<C>f`: Feature map (local features, `H`/`W` = height/width of the map, smaller than input).  
- `<img>dino2.vis`: PCA(3) RGB visualization (essentially JPEG) if the `VIS` environment variable is non-empty (e.g., `VIS=1`).  

**Numpy Loading Tip**:  
Files like `xxx.aaaf` (class vector) or `xxx.aaa,bbb,cccf` (feature map) can be loaded directly with:  
```python
import numpy as np
# Load class vector (remove leading singleton dimension with [0])
class_vec = np.fromfile("xxx.aaaf", dtype="aaaf")[0]
# Load feature map (preserves H,W,C shape)
feat_map = np.fromfile("xxx.aaa,bbb,cccf", dtype="aaa,bbb,cccf")[0]
```  


### DINOv3  
Currently only supports ViT models. Run with:  
```bash
dino3.py <img>...
```  

**Output Files**:  
- `<img>dino3<C>f`: Class vector.  
- `<img>dino3.<H>,<W>,<C>f`: Feature map.  
- `<img>dino3.vis`: PCA(3) RGB visualization (if `VIS` is set).  


## Program Usage  

### DINOv2  
```python
from dinov23_cpp_py.dinov2 import run
import numpy as np

# Dummy input: shape = (batch_size, height, width, channels)
# Requirements: 
# - height/width < 504  
# - height/width must be divisible by patch size (14 for DINOv2)
img = np.random.randint(0, 256, (8, 504, 504, 3), dtype="B")

# Extract features (class vector + feature map)
clsvec, featmap = run(img, "../dinov2-with-registers-small-imagenet1k-1-layer-f16.gguf")
```  


### DINOv3  
Currently only supports ViT models.  
```python
from dinov23_cpp_py.dinov3 import run
import numpy as np

# Dummy input: shape = (batch_size, height, width, channels)
# Requirements: 
# - height/width < 224  
# - height/width must be divisible by patch size (16 for DINOv3 ViT)
img = np.random.randint(0, 256, (8, 224, 224, 3), dtype="B")

# Extract features (class vector + feature map)
clsvec, featmap = run(img, "../dinov3-vit7b16-pretrain-lvd1689m-f16.gguf")
```  


## Model Weights  
Pre-converted GGUF weights are available (conversion script included in the repo). Quantized models are on work. If https://huggingface.co is hard to access, try mirrors at https://hf-mirror.com/zisisnotzis/dinov3-gguf or https://www.modelscope.cn/models/ziszis/dinov3-gguf.

### DINOv2  
| Name | Size | Quantization |  
|------|------|--------------|  
| [dinov2-base-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-base-f16.gguf) | 176 MB | f16 |  
| [dinov2-base-imagenet1k-1-layer-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-base-imagenet1k-1-layer-f16.gguf) | 179 MB | f16 |  
| [dinov2-giant-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-giant-f16.gguf) | 2.28 GB | f16 |  
| [dinov2-giant-imagenet1k-1-layer-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-giant-imagenet1k-1-layer-f16.gguf) | 2.29 GB | f16 |  
| [dinov2-large-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-large-f16.gguf) | 612 MB | f16 |  
| [dinov2-large-imagenet1k-1-layer-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-large-imagenet1k-1-layer-f16.gguf) | 616 MB | f16 |  
| [dinov2-small-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-small-f16.gguf) | 45.3 MB | f16 |  
| [dinov2-small-imagenet1k-1-layer-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-small-imagenet1k-1-layer-f16.gguf) | 46.9 MB | f16 |  
| [dinov2-with-registers-base-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-with-registers-base-f16.gguf) | 176 MB | f16 |  
| [dinov2-with-registers-base-imagenet1k-1-layer-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-with-registers-base-imagenet1k-1-layer-f16.gguf) | 179 MB | f16 |  
| [dinov2-with-registers-giant-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-with-registers-giant-f16.gguf) | 2.28 GB | f16 |  
| [dinov2-with-registers-giant-imagenet1k-1-layer-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-with-registers-giant-imagenet1k-1-layer-f16.gguf) | 2.29 GB | f16 |  
| [dinov2-with-registers-large-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-with-registers-large-f16.gguf) | 612 MB | f16 |  
| [dinov2-with-registers-large-imagenet1k-1-layer-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-with-registers-large-imagenet1k-1-layer-f16.gguf) | 616 MB | f16 |  
| [dinov2-with-registers-small-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-with-registers-small-f16.gguf) | 45.3 MB | f16 |  
| [dinov2-with-registers-small-imagenet1k-1-layer-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-with-registers-small-imagenet1k-1-layer-f16.gguf) | 46.9 MB | f16 |  


### DINOv3  
| Name | Size | Quantization |  
|------|------|--------------|  
| [dinov3-convnext-base-pretrain-lvd1689m-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov3-convnext-base-pretrain-lvd1689m-f16.gguf) | 175 MB | f16 |  
| [dinov3-convnext-large-pretrain-lvd1689m-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov3-convnext-large-pretrain-lvd1689m-f16.gguf) | 393 MB | f16 |  
| [dinov3-convnext-small-pretrain-lvd1689m-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov3-convnext-small-pretrain-lvd1689m-f16.gguf) | 99.2 MB | f16 |  
| [dinov3-convnext-tiny-pretrain-lvd1689m-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov3-convnext-tiny-pretrain-lvd1689m-f16.gguf) | 55.8 MB | f16 |  
| [dinov3-vit7b16-pretrain-lvd1689m-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov3-vit7b16-pretrain-lvd1689m-f16.gguf) | 13.4 GB | f16 |  
| [dinov3-vitb16-pretrain-lvd1689m-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov3-vitb16-pretrain-lvd1689m-f16.gguf) | 172 MB | f16 |  
| [dinov3-vith16plus-pretrain-lvd1689m-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov3-vith16plus-pretrain-lvd1689m-f16.gguf) | 1.68 GB | f16 |  
| [dinov3-vitl16-pretrain-lvd1689m-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov3-vitl16-pretrain-lvd1689m-f16.gguf) | 607 MB | f16 |  
| [dinov3-vitl16-pretrain-sat493m-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov3-vitl16-pretrain-sat493m-f16.gguf) | 607 MB | f16 |  
| [dinov3-vits16-pretrain-lvd1689m-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov3-vits16-pretrain-lvd1689m-f16.gguf) | 43.3 MB | f16 |  
| [dinov3-vits16plus-pretrain-lvd1689m-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov3-vits16plus-pretrain-lvd1689m-f16.gguf) | 57.6 MB | f16 |  


## Notes  
- DINOv3 feature behavior may be unexpected (potential bug).  
- DINOv3 currently only supports ViT models.  
- GPU mode is untested.  
