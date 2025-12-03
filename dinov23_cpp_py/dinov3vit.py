#!/bin/env python
from sys import argv

from sklearn.decomposition import PCA
from PIL.Image import open, fromarray, Resampling

from .gg import *
from . import gg


def run(x: np.ndarray, model=getenv('MODEL', '../hf/dinov3-vits16-pretrain-lvd1689m-f16.gguf'), gpu=False) -> tuple[np.ndarray, np.ndarray]:
    global SHAPE, X, Y, PATCHSZ, IMGSZ
    gg.init(model, gpu)
    if x.shape != SHAPE:
        layer = getint('num_hidden_layers')
        PATCHSZ = getint('patch_size')
        h = x.shape[1] // PATCHSZ
        w = x.shape[2] // PATCHSZ
        reg = getint('num_register_tokens')
        X = Tensor(x.shape[::-1]).named('x')
        H = Tensor((h,), GGML_TYPE_I32).named('h')
        W = Tensor((w,), GGML_TYPE_I32).named('w')
        Y = X.conv2dnhwc('embeddings.patch_embeddings', PATCHSZ, PATCHSZ).flatten(1, 2).rcat(Tensor('embeddings.register_tokens').cat(Tensor('embeddings.cls_token'), 1), 1)
        for i in range(layer):
            Y = Y.normscale(f'layer.{i}.norm1').atrope(f'layer.{i}', H, W, h, w, 1 + reg).mul_(Tensor(f'layer.{i}.layer_scale1.lambda1')).add_(Y)
            Y = Y.normscale(f'layer.{i}.norm2').mlpOrSwiglu(f'layer.{i}.mlp.').mul_(Tensor(f'layer.{i}.layer_scale2.lambda1')).add_(Y)
        Y = Y.normscale('norm').named('y')
        initgraph(Y)
        H.setnp(np.arange(h, dtype='i'))
        W.setnp(np.arange(w, dtype='i'))
    X.setnp((x - np.array([123.675, 116.28, 103.53], 'f')) / np.array([58.395, 57.12, 57.375], 'f'))
    gg.run()
    y = Y.asnp()
    return y[0, :, 4], y[0, :, 5:].reshape(y.shape[1], x.shape[1] // PATCHSZ, -1, y.shape[3])


SHAPE = ()
PATCHSZ = 16
if __name__ == '__main__':
    X = [open(i) for i in argv[1:]]
    s = int(getenv('SZ', max(max(i.size) for i in X)))//PATCHSZ*PATCHSZ
    r = np.exp(sum(np.log(i.size[1] / i.size[0]) for i in X) / len(X))
    w, h = (int(s / r / PATCHSZ) * PATCHSZ, s) if r > 1 else (s, int(s * r / PATCHSZ) * PATCHSZ)
    X = np.stack([np.asarray(i.resize((w, h), Resampling.BOX))[..., :3] for i in X])
    Z, Y = run(X)
    for i, y, z in zip(argv[1:], Y, Z):
        y.tofile(f'{i}dinov3vit.{','.join(map(str, y.shape))}f')
        z.tofile(f'{i}dinov3vit.{','.join(map(str, z.shape))}f')
    if not getenv('VIS', ''):
        exit()
    Y = PCA(3).fit_transform(Y.reshape(-1, Y.shape[3])).reshape(*Y.shape[:3], -1)
    Y -= Y.min((1, 2), keepdims=True)
    Y *= 255.9 / Y.max((1, 2), keepdims=True)
    Y = Y.astype('B')
    for i, y in zip(argv[1:], Y):
        fromarray(y).resize((w, h), Resampling.LANCZOS).save(f'{i}dinov3vit.vis', 'jpeg')
