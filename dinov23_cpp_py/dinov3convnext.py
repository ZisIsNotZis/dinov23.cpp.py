#!/bin/env python
from sys import argv

from sklearn.decomposition import PCA
from PIL.Image import open, fromarray, Resampling

from .gg import *
from . import gg


def run(x: np.ndarray, model=getenv('MODEL', '../hf/dinov3-convnext-tiny-pretrain-lvd1689m-f16.gguf'), gpu=False) -> tuple[np.ndarray, np.ndarray]:
    global SHAPE, X, Y, Z
    gg.init(model, gpu)
    if x.shape != SHAPE:
        X = Y = Tensor(x.shape[::-1]).named('x')
        try:
            for i in range(99):
                if not i:
                    Y = Y.conv2dnhwc(f'stages.{i}.downsample_layers.0', 4, 4).normscale_(f'stages.{i}.downsample_layers.1')
                else:
                    Y = Y.normscale_(f'stages.{i}.downsample_layers.0').conv2dnhwc(f'stages.{i}.downsample_layers.1', 2, 2)
                try:
                    for j in range(99):
                        Y = Y.conv2dwnhwc(f'stages.{i}.layers.{j}.depthwise_conv', 1, 1, 3, 3).normscale_(f'stages.{i}.layers.{j}.layer_norm').mlp(f'stages.{i}.layers.{j}.pointwise_conv').mul_(Tensor(f'stages.{i}.layers.{j}.gamma')).add_(Y)
                except KeyError:
                    pass
        except KeyError:
            pass
        Z = Y.pool2d()
        Y = Y.normscale('layer_norm')
        initgraph(Y, Z)
    X.setnp((x - np.array([123.675, 116.28, 103.53], 'f')) / np.array([58.395, 57.12, 57.375], 'f'))
    gg.run()
    return Z.asnp()[0, 0], Y.asnp()


SHAPE = ()
PATCHSZ = 1
if __name__ == '__main__':
    X = [open(i) for i in argv[1:]]
    s = int(getenv('SZ', max(max(i.size) for i in X)))//PATCHSZ*PATCHSZ
    r = np.exp(sum(np.log(i.size[1] / i.size[0]) for i in X) / len(X))
    w, h = (int(s / r / PATCHSZ) * PATCHSZ, s) if r > 1 else (s, int(s * r / PATCHSZ) * PATCHSZ)
    X = np.stack([np.asarray(i.resize((w, h), Resampling.BOX))[..., :3] for i in X])
    Z, Y = run(X)
    for i, y, z in zip(argv[1:], Y, Z):
        y.tofile(f'{i}dinov3convnext.{','.join(map(str, y.shape))}f')
        z.tofile(f'{i}dinov3convnext.{','.join(map(str, z.shape))}f')
    if not getenv('VIS', ''):
        exit()
    Y = PCA(3).fit_transform(Y.reshape(-1, Y.shape[3])).reshape(*Y.shape[:3], -1)
    Y -= Y.min((1, 2), keepdims=True)
    Y *= 255.9 / Y.max((1, 2), keepdims=True)
    Y = Y.astype('B')
    for i, y in zip(argv[1:], Y):
        fromarray(y).resize((w, h), Resampling.LANCZOS).save(f'{i}dinov3convnext.vis', 'jpeg')
