# dinov23.cpp.py
A python ggml implementation of dinov2 and dinov3, heavily inspired by [dinov2.cpp](https://github.com/lavaman131/dinov2.cpp)

## Prerequisite
* python 3.12+ (for newer typing syntax)
* my ggml-python [fork](https://github.com/ZisIsNotZis/ggml-python) (since the original one is really outdated)
* pillow
* scikit-learn

## Usage
### dinov2
* `dino2.py <img>...`: Will generate `<img>dino2<C>f` for class vector, `<img>dino2.<H>,<W>,<C>f` for feature map, and `<img>dino2.vis` for PCA(3) RGB visulization (if env var `VIS` non empty)

### dinov3
* `dino3.py <img>...`: Will generate `<img>dino3<C>f` for class vector, `<img>dino3.<H>,<W>,<C>f` for feature map, and `<img>dino3.vis` for PCA(3) RGB visulization (if env var `VIS` non empty)

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
* `dino2.py` feature looks much more reasonable than `dino3.py`. Not sure why, maybe some kind of bug inside `dino3.py`
* gpu mode not tested
