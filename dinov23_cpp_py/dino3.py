#!/bin/env python
from sys import argv

from sklearn.decomposition import PCA
from PIL.Image import open, fromarray, Resampling

from .gg import *
from . import gg


def run(x: np.ndarray, model='dinov3-vits16-pretrain-lvd1689m-f16.gguf', gpu=False) -> tuple[np.ndarray, np.ndarray]:
    global SHAPE, X, Y, PATCHSZ, IMGSZ
    gg.init(model, gpu)
    if x.shape != SHAPE:
        layer = getkey(gg.GGUFCTX, 'num_hidden_layers')
        PATCHSZ = getkey(gg.GGUFCTX, 'patch_size')
        IMGSZ = getkey(gg.GGUFCTX, 'image_size')
        X = Tensor(x.transpose(0, 3, 1, 2).shape[::-1]).named('x')
        H = Tensor((x.shape[1] // PATCHSZ,), GGML_TYPE_I32).named('h')
        W = Tensor((x.shape[2] // PATCHSZ,), GGML_TYPE_I32).named('w')
        Y = X.conv(Tensor('embeddings.patch_embeddings.weight')).flatten(0, 1).T().add_(Tensor('embeddings.patch_embeddings.bias')).rcat(Tensor('embeddings.register_tokens').cat(Tensor('embeddings.cls_token'), 1), 1)
        for i in range(layer):
            Y = Y.norm(f'layer.{i}.norm1').atrope(f'layer.{i}', H, W, IMGSZ // PATCHSZ, IMGSZ // PATCHSZ, 1 + Tensor('embeddings.register_tokens').shape[1]).mul_(Tensor(f'layer.{i}.layer_scale1.lambda1')).add_(Y)
            Y = Y.norm(f'layer.{i}.norm2').mlpOrSwiglu(f'layer.{i}.mlp').mul_(Tensor(f'layer.{i}.layer_scale2.lambda1')).add_(Y)
        initgraph(Y.norm('norm').named('y'))
        H.setnp(np.arange(x.shape[1] // PATCHSZ, dtype='i'))
        W.setnp(np.arange(x.shape[2] // PATCHSZ, dtype='i'))
    X.setnp(((x - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]).astype('f').transpose(0, 3, 1, 2))
    gg.run()
    y = Y.asnp()
    return y[0, :, 4], y[0, :, 5:].reshape(y.shape[1], x.shape[1] // getkey(gg.GGUFCTX, 'patch_size'), -1, y.shape[3])


SHAPE = ()
IMGSZ = 224
PATCHSZ = 16
if __name__ == '__main__':
    del argv[0]
    x = [open(i) for i in argv]
    r = sum(i.size[1] / i.size[0] for i in x) / len(x)
    h, w = (IMGSZ, round(IMGSZ / r / PATCHSZ) * PATCHSZ) if r > 0 else (round(IMGSZ * r / PATCHSZ) * PATCHSZ, IMGSZ)
    x, z = run(np.stack([np.asarray(i.resize([w, h], Resampling.BOX))[..., :3] for i in x]))
    for i, x, y in zip(argv, x, z):
        x.tofile(f'{i}dino3.{','.join(map(str, x.shape))}f')
        y.tofile(f'{i}dino3.{','.join(map(str, y.shape))}f')
    if getenv('VIS', ''):
        z = PCA(3).fit_transform(z.reshape(-1, z.shape[3])).reshape(*z.shape[:3], 3)
        z -= z.min((1, 2), keepdims=True)
        z *= 255.9 / z.max((1, 2), keepdims=True)
        for i, z in zip(argv, z.astype('B')):
            fromarray(z).resize((z.shape[1] * PATCHSZ, z.shape[0] * PATCHSZ), Resampling.LANCZOS).save(f'{i}dino3.vis', 'jpeg')
