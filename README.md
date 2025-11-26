# dinov23.cpp.py
A python ggml implementation of dinov2 and dinov3, heavily inspired by [dinov2.cpp](https://github.com/lavaman131/dinov2.cpp)

## Dependency
* python 3.12+ (for newer typing syntax)
* my ggml-python [fork](https://github.com/ZisIsNotZis/ggml-python) (since the original one is really outdated)
* pillow
* scikit-learn

## Setup
* ~`pip install .`~(doesn't work for git+https) `pip install -r requirements.txt`

## CLI Usage

### dinov2
* `dino2.py <img>...`: Will generate `<img>dino2<C>f` for class vector (C is channel, e.g. 384 for lower dinov2 model) `<img>dino2.<H>,<W>,<C>f` for feature map (H and W is height and width for feature map, which is definitely smaller than the original image, and usually can be interpolated to original size), and `<img>dino2.vis` for PCA(3) RGB visulization (if env var `VIS` non empty, the file is essentially a JPEG, and vis is just a fancy extension to differentiate from original images)
* A file named `xxx.aaaf` `xxx.aaa,bbb,cccf` can be easily loaded by numpy using `np.fromfile('<filename>','<extension>')[0]` (0 to remove the leading one element dimension), e.g. `np.fromfile('xxx.aaa,bbb,cccf','aaa,bbb,cccf')[0]`, which is why the file extension looks like this. This is really handy.

### dinov3
Currently only supports vit model
* `dino3.py <img>...`: Will generate `<img>dino3<C>f` for class vector, `<img>dino3.<H>,<W>,<C>f` for feature map, and `<img>dino3.vis` for PCA(3) RGB visulization (if env var `VIS` non empty)

## Program Usage

### dinov2
```py
from dinov23_cpp_py.dinov2 import run
import numpy as np
img = np.random.randint(0, 256, (8,504,504,3), 'B') #n,h,w,c where h,w<504 and h,w%patch_size=0
# patch_size is usually 14 for dinov2
clsvec, featmap = run(img, '../dinov2-with-registers-small-imagenet1k-1-layer-f16.gguf')
```

### dinov3
Currently only supports vit model
```py
from dinov23_cpp_py.dinov3 import run
import numpy as np
img = np.random.randint(0, 256, (8,224,224,3), 'B') #n,h,w,c where h,w<224 and h,w%patch_size=0
# patch_size is usually 16 for dinov3
clsvec, featmap = run(img, '../dinov3-vit7b16-pretrain-lvd1689m-f16.gguf')
```

## Weight
Conversion script provided in repo

### dinov2
name|size|quantization
---|---|---
[dinov2-base-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-base-f16.gguf)|176 MB|f16
[dinov2-base-imagenet1k-1-layer-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-base-imagenet1k-1-layer-f16.gguf)|179 MB|f16
[dinov2-giant-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-giant-f16.gguf)|2.28 GB|f16
[dinov2-giant-imagenet1k-1-layer-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-giant-imagenet1k-1-layer-f16.gguf)|2.29 GB|f16
[dinov2-large-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-large-f16.gguf)|612 MB|f16
[dinov2-large-imagenet1k-1-layer-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-large-imagenet1k-1-layer-f16.gguf)|616 MB|f16
[dinov2-small-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-small-f16.gguf)|45.3 MB|f16
[dinov2-small-imagenet1k-1-layer-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-small-imagenet1k-1-layer-f16.gguf)|46.9 MB|f16
[dinov2-with-registers-base-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-with-registers-base-f16.gguf)|176 MB|f16
[dinov2-with-registers-base-imagenet1k-1-layer-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-with-registers-base-imagenet1k-1-layer-f16.gguf)|179 MB|f16
[dinov2-with-registers-giant-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-with-registers-giant-f16.gguf)|2.28 GB|f16
[dinov2-with-registers-giant-imagenet1k-1-layer-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-with-registers-giant-imagenet1k-1-layer-f16.gguf)|2.29 GB|f16
[dinov2-with-registers-large-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-with-registers-large-f16.gguf)|612 MB|f16
[dinov2-with-registers-large-imagenet1k-1-layer-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-with-registers-large-imagenet1k-1-layer-f16.gguf)|616 MB|f16
[dinov2-with-registers-small-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-with-registers-small-f16.gguf)|45.3 MB|f16
[dinov2-with-registers-small-imagenet1k-1-layer-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov2-with-registers-small-imagenet1k-1-layer-f16.gguf)|46.9 MB|f16

### dinov3
name|size|quantization
---|---|---
[dinov3-convnext-base-pretrain-lvd1689m-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov3-convnext-base-pretrain-lvd1689m-f16.gguf)|175 MB|f16
[dinov3-convnext-large-pretrain-lvd1689m-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov3-convnext-large-pretrain-lvd1689m-f16.gguf)|393 MB|f16
[dinov3-convnext-small-pretrain-lvd1689m-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov3-convnext-small-pretrain-lvd1689m-f16.gguf)|99.2 MB|f16
[dinov3-convnext-tiny-pretrain-lvd1689m-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov3-convnext-tiny-pretrain-lvd1689m-f16.gguf)|55.8 MB|f16
[dinov3-vit7b16-pretrain-lvd1689m-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov3-vit7b16-pretrain-lvd1689m-f16.gguf)|13.4 GB|f16
[dinov3-vitb16-pretrain-lvd1689m-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov3-vitb16-pretrain-lvd1689m-f16.gguf)|172 MB|f16
[dinov3-vith16plus-pretrain-lvd1689m-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov3-vith16plus-pretrain-lvd1689m-f16.gguf)|1.68 GB|f16
[dinov3-vitl16-pretrain-lvd1689m-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov3-vitl16-pretrain-lvd1689m-f16.gguf)|607 MB|f16
[dinov3-vitl16-pretrain-sat493m-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov3-vitl16-pretrain-sat493m-f16.gguf)|607 MB|f16
[dinov3-vits16-pretrain-lvd1689m-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov3-vits16-pretrain-lvd1689m-f16.gguf)|43.3 MB|f16
[dinov3-vits16plus-pretrain-lvd1689m-f16.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov3-vits16plus-pretrain-lvd1689m-f16.gguf)|57.6 MB|f16

## Note
* dinov3 feature wired, maybe bug inside
* dinov3 currently only supports vit model
* gpu mode not tested
