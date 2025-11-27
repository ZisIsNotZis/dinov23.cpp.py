#!/bin/env python
from sys import argv

from sklearn.decomposition import PCA
from PIL.Image import open, fromarray, Resampling

from .gg import *
from . import gg


def remb(emb: np.ndarray, h: int, w: int) -> np.ndarray:
    m = int((len(emb) - 1) ** .5)
    assert len(emb) == 1 + m * m
    return np.concat([emb[:1], np.stack([fromarray(i).resize((w, h), Resampling.BICUBIC) for i in emb[1:].reshape(m, m, -1).transpose(2, 0, 1)], 2).reshape(-1, emb.shape[1])])  # type:ignore


def run(x: np.ndarray, model=getenv('MODEL','dinov2-with-registers-small-imagenet1k-1-layer-f16.gguf'), gpu=False) -> tuple[np.ndarray, np.ndarray]:
    global SHAPE, X, Y, IMGSZ, PATCHSZ
    gg.init(model, gpu)
    if x.shape != SHAPE:
        layer = getkey(gg.GGUFCTX, 'num_hidden_layers')
        IMGSZ = getkey(gg.GGUFCTX, 'img_size')
        PATCHSZ = getkey(gg.GGUFCTX, 'patch_size')
        X = Tensor(x.transpose(0, 3, 1, 2).shape[::-1]).named('x')
        Y = X.conv(Tensor('embeddings.patch_embeddings.projection.weight'))
        Y = Y.add_(Tensor('embeddings.patch_embeddings.projection.bias'))
        Y = Y.flatten(0, 1).T().cont()
        EMB = Tensor((Y.shape[0], 1 + Y.shape[1])).named('emb')
        Y = Y.rcat(Tensor('embeddings.cls_token'), 1).add_(EMB)
        try:
            Y = Y.rcat(Tensor('embeddings.register_tokens'), 1)
        except KeyError:
            pass
        for i in range(layer):
            Y = Y.normscale(f'encoder.layer.{i}.norm1').at(f'encoder.layer.{i}.attention').mul_(Tensor(f'encoder.layer.{i}.layer_scale1.lambda1')).add_(Y)
            Y = (Tensor.mlp if layer != 40 else Tensor.swiglu)(Y.normscale(f'encoder.layer.{i}.norm2'), f'encoder.layer.{i}.mlp').mul_(Tensor(f'encoder.layer.{i}.layer_scale2.lambda1')).add_(Y)
        initgraph(Y.normscale('layernorm').named('y'))
        EMB.setnp(remb(Tensor('embeddings.position_embeddings').asnp()[0, 0], x.shape[1] // PATCHSZ, x.shape[2] // PATCHSZ))
    X.setnp((x - np.array([123.675, 116.28, 103.53], 'f')).transpose(0, 3, 1, 2) / np.array([58.395, 57.12, 57.375], 'f')[:, None, None])
    gg.run()
    y = Y.asnp()
    return y[0, :, -x.shape[1]*x.shape[2]//PATCHSZ**2-1], y[0, :, -x.shape[1]*x.shape[2]//PATCHSZ**2:].reshape(y.shape[1], x.shape[1] // PATCHSZ, -1, y.shape[3])


SHAPE = ()
IMGSZ = 504
PATCHSZ = 14
if __name__ == '__main__':
    del argv[0]
    x = [open(i) for i in argv]
    r = sum(i.size[1] / i.size[0] for i in x) / len(x)
    h, w = (IMGSZ, round(IMGSZ / r / PATCHSZ) * PATCHSZ) if r > 1 else (round(IMGSZ * r / PATCHSZ) * PATCHSZ, IMGSZ)
    x, z = run(np.stack([np.asarray(i.resize([w, h], Resampling.BOX))[..., :3] for i in x]))
    for i, x, y in zip(argv, x, z):
        x.tofile(f'{i}dino2.{','.join(map(str, x.shape))}f')
        y.tofile(f'{i}dino2.{','.join(map(str, y.shape))}f')
    if getenv('VIS', ''):
        z = PCA(3).fit_transform(z.reshape(-1, z.shape[3])).reshape(*z.shape[:3], 3)
        z -= z.min((1, 2), keepdims=True)
        z *= 255.9 / z.max((1, 2), keepdims=True)
        for i, z in zip(argv, z.astype('B')):
            fromarray(z).resize((z.shape[1] * PATCHSZ, z.shape[0] * PATCHSZ), Resampling.LANCZOS).save(f'{i}dino2.vis', 'jpeg')
