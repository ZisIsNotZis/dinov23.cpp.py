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


def run(x: np.ndarray, model=getenv('MODEL', '../hf/dinov2-with-registers-small-imagenet1k-1-layer-f16.gguf'), gpu=False) -> tuple[np.ndarray, np.ndarray]:
    global SHAPE, X, Y, PATCHSZ
    gg.init(model, gpu)
    if x.shape != SHAPE:
        layer = getint('num_hidden_layers')
        PATCHSZ = getint('patch_size')
        X = Tensor(x.transpose(0, 3, 1, 2).shape[::-1]).named('x')
        Y = X.conv2dskp0(Tensor('embeddings.patch_embeddings.projection.weight')).add_(Tensor('embeddings.patch_embeddings.projection.bias')).flatten(0, 1).T().cont()
        EMB = Tensor((Y.shape[0], 1 + Y.shape[1])).named('emb')
        Y = Y.rcat(Tensor('embeddings.cls_token'), 1).add_(EMB)
        try:
            Y = Y.rcat(Tensor('embeddings.register_tokens'), 1)
        except KeyError:
            pass
        for i in range(layer):
            Y = Y.normscale(f'encoder.layer.{i}.norm1').at(f'encoder.layer.{i}.attention').mul_(Tensor(f'encoder.layer.{i}.layer_scale1.lambda1')).add_(Y)
            Y = (Tensor.mlp if layer != 40 else lambda x, w: x.swiglu(w, 'silu'))(Y.normscale(f'encoder.layer.{i}.norm2'), f'encoder.layer.{i}.mlp.fc').mul_(Tensor(f'encoder.layer.{i}.layer_scale2.lambda1')).add_(Y)
        Y = Y.normscale_('layernorm').named('y')
        initgraph(Y)
        EMB.setnp(remb(Tensor('embeddings.position_embeddings').asnp()[0, 0], x.shape[1] // PATCHSZ, x.shape[2] // PATCHSZ))
    X.setnp((x - np.array([123.675, 116.28, 103.53], 'f')).transpose(0, 3, 1, 2) / np.array([58.395, 57.12, 57.375], 'f')[:, None, None])
    gg.run()
    y = Y.asnp()
    return y[0, :, -(x.shape[1] * x.shape[2] // PATCHSZ ** 2) - 1], y[0, :, -(x.shape[1] * x.shape[2] // PATCHSZ ** 2):].reshape(y.shape[1], x.shape[1] // PATCHSZ, -1, y.shape[3])


SHAPE = ()
PATCHSZ = 14
if __name__ == '__main__':
    X = [open(i) for i in argv[1:]]
    s = int(getenv('SZ', max(max(i.size)for i in X)))//PATCHSZ*PATCHSZ
    r = np.exp(sum(np.log(i.size[1] / i.size[0]) for i in X) / len(X))
    w, h = (int(s / r / PATCHSZ) * PATCHSZ, s) if r > 1 else (s, int(s * r / PATCHSZ) * PATCHSZ)
    X = np.stack([np.asarray(i.resize((w, h), Resampling.BOX))[..., :3] for i in X])
    Z, Y = run(X)
    for i, y, z in zip(argv[1:], Y, Z):
        y.tofile(f'{i}dinov2.{','.join(map(str, y.shape))}f')
        z.tofile(f'{i}dinov2.{','.join(map(str, z.shape))}f')
    if not getenv('VIS', ''):
        exit()
    Y = PCA(3).fit_transform(Y.reshape(-1, Y.shape[3])).reshape(*Y.shape[:3], -1)
    Y -= Y.min((1, 2), keepdims=True)
    Y *= 255.9 / Y.max((1, 2), keepdims=True)
    Y = Y.astype('B')
    for i, y in zip(argv[1:], Y):
        fromarray(y).resize((w, h), Resampling.LANCZOS).save(f'{i}dinov2.vis', 'jpeg')
