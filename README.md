# dinov23.cpp.py
A python ggml implementation of dinov2 and dinov3, heavily inspired by [dinov2.cpp](https://github.com/lavaman131/dinov2.cpp)

## Prerequisite
* python 3.12+ (for newer typing syntax)
* my ggml-python [fork](https://github.com/ZisIsNotZis/ggml-python) (since the original one is really outdated)
* pillow
* scikit-learn

## Usage
* `dino2.py <img>...`: Will generate `<img>dino2<C>f` for class vector, `<img>dino2.<H>,<W>,<C>f` for feature map, and `<img>dino2.vis` for PCA(3) RGB visulization (if env var `VIS` non empty)
* `dino3.py <img>...`: Will generate `<img>dino3<C>f` for class vector, `<img>dino3.<H>,<W>,<C>f` for feature map, and `<img>dino3.vis` for PCA(3) RGB visulization (if env var `VIS` non empty)

## Weight
I'm uploading converted weight to huggingface

## Note
* `dino2.py` feature looks much more reasonable than `dino3.py`. Not sure why, maybe some kind of bug inside `dino3.py`
* gpu mode not tested