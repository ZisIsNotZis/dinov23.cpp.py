from ctypes import POINTER, c_void_p, c_float, casserteq as ccasserteq, _Pointer
from math import prod
from logging import info, warning, basicConfig
from atexit import register
from os import getenv

from ggml import *
from ggml.utils import to_numpy, from_numpy
import numpy as np


def debug[**P, T](f: Callable[P, T], lv=[0]) -> Callable[P, T]:
    def debug(*_: P.args, **__: P.kwargs) -> T:
        info('%s%s %s', '\t' * lv[0], f.__name__, ' '.join(map(objstr, _)))
        lv[0] += 1
        try:
            o = f(*_, **__)
            info('%s%s', '\t' * (lv[0] - 1), objstr(o))
            return o
        finally:
            lv[0] -= 1

    return debug


def asserteq[T](a: T, b: T) -> None:
    assert a == b, (a, b)


@debug
def getkey(ctx_gguf: gguf_context_p, key: str) -> int:
    if (t := gguf_find_key(ctx_gguf, key.encode())) < 0:
        raise KeyError(key)
    return gguf_get_val_i32(ctx_gguf, t) if gguf_get_kv_type(ctx_gguf, t) == GGUF_TYPE_INT32 else gguf_get_val_u32(ctx_gguf, t)


def getensor(ctx: ggml_context_p, tensor: str | bytes) -> ggml_tensor_p:
    if not (t := ggml_get_tensor(ctx, (tensor.encode() if isinstance(tensor, str) else tensor))):
        raise KeyError(tensor)
    return t


def graphgetensor(gf: ggml_cgraph_p, tensor: str | bytes) -> ggml_tensor_p:
    if not (t := ggml_graph_get_tensor(gf, (tensor.encode() if isinstance(tensor, str) else tensor))):
        raise KeyError(tensor)
    return t


def typestr(type: int) -> str:
    return [i for i, j in globals().items() if i.startswith('GGML_TYPE_') and j == type][0][10:]


def objstr(_: object) -> str:
    return str((_.name, typestr(_.type), *_.shape) if isinstance(_, Tensor) else (_.dtype.char, *_.shape) if isinstance(_, np.ndarray) else _)


def ljust[T](a: tuple[T, ...] | list[T], l: int = 4, v=1) -> tuple[T, ...]:
    assert all(i == v for i in a[:-l])
    return (v,) * (l - len(a)) + tuple(a[-l:])


def rjust[T](a: tuple[T, ...] | list[T], l: int = 4, v=1) -> tuple[T, ...]:
    assert all(i == v for i in a[l:])
    return tuple(a[:l]) + (v,) * (l - len(a))


class Tensor:
    asdtype = {GGML_TYPE_I8: np.dtype('b'), GGML_TYPE_I16: np.dtype('h'), GGML_TYPE_I32: np.dtype('i'), GGML_TYPE_I64: np.dtype('l'), GGML_TYPE_F64: np.dtype('d'), GGML_TYPE_F32: np.dtype('f'), GGML_TYPE_F16: np.dtype('e')}

    def __init__(self, _: 'str | tuple[int,...] | ggml_tensor_p', type=GGML_TYPE_F32):
        self.base = getensor(CTX, _) if isinstance(_, str) else ggml_new_tensor_4d(GRAPHCTX, type, *_, *[1] * (4 - len(_))) if isinstance(_, (tuple, list)) else _
        try:
            TOUCH.remove(self.base.contents.name)
        except KeyError:
            pass

    @property
    def name(self) -> str:
        return self.base.contents.name.decode()

    def named(self, name: str) -> 'Tensor':
        if name:
            ggml_set_name(self.base, name.encode())
        return self

    @property
    def type(self) -> int:
        return self.base.contents.type

    @property
    def shape(self) -> tuple[int, ...]:
        return casserteq(tuple[int, ...], tuple(map(int, self.base.contents.ne)))

    @property
    def stride(self) -> tuple[int, ...]:
        return casserteq(tuple[int, ...], tuple(map(int, self.base.contents.nb)))

    @debug
    def flatten(self, i=0, j=3, s=(-1,)) -> 'Tensor':
        shape = list(self.shape)
        stride = list(self.stride)
        self = self if np.multiply(stride[i], shape[i:j]).tolist() == stride[i + 1:j + 1] else self.cont()
        stride = list(self.stride)
        if -1 in s:
            s = list(s)
            k = s.index(-1)
            s[k] = prod(shape[i:j + 1]) // prod(s[:k] + s[k + 1:])
        shape[i:j + 1] = s
        assert all(i == 1 for i in shape[4:])
        shape = shape[:4]
        shape += [1] * (4 - len(shape))
        stride[i:j + 1] = np.multiply(stride[i], [1, *s[:-1]])
        stride = stride[:4]
        stride += list(stride[-1:]) * (4 - len(stride))
        return Tensor(ggml_view_4d(GRAPHCTX, self.base, *shape, *stride[1:], 0)).named('flatten')

    @debug
    def permute(self, i=0, j=1, k=2, l=3) -> 'Tensor':
        return Tensor(ggml_permute(GRAPHCTX, self.base, i, j, k, l)).named('permute')

    def T(self, i=0, j=1) -> 'Tensor':
        dim = list(range(4))
        dim[i] = j
        dim[j] = i
        return self.permute(*dim).named('T')

    def expandshape(self, _: 'Tensor', ex=4) -> tuple[tuple[int, ...], tuple[int, ...]]:
        assert all(i == ex or a == 1 or b == 1 or a == b for i, (a, b) in enumerate(zip(self.shape, _.shape)))
        return tuple(a if i == ex else max(a, b) for i, (a, b) in enumerate(zip(self.shape, _.shape))), tuple(b if i == ex else max(a, b) for i, (a, b) in enumerate(zip(self.shape, _.shape)))

    def expand(self, _: 'Tensor', ex=4) -> tuple['Tensor', 'Tensor']:
        shape, shape_ = self.expandshape(_, ex)
        return self if shape == self.shape else Tensor(ggml_repeat(GRAPHCTX, self.base, Tensor(shape).base)).named('expand'), _ if shape_ == _.shape else Tensor(ggml_repeat(GRAPHCTX, _.base, Tensor(shape_).base)).named('expand2')

    def cont(self) -> 'Tensor':
        return self if ggml_is_contiguous(self.base) else Tensor(ggml_cont(GRAPHCTX, self.base)).named('cont')

    def untranspose(self) -> 'Tensor':
        return Tensor(ggml_cont(GRAPHCTX, self.base)).named('untranspose') if ggml_is_transposed(self.base) else self

    @debug
    def __getitem__(self, i: int | None | slice | tuple[int | None | slice]) -> 'Tensor':
        self = self.untranspose()
        shape = list(self.shape)
        stride = list(self.stride)
        offset = 0
        j = 0
        for i in (i if isinstance(i, tuple) else (i,)):
            if isinstance(i, int):
                offset += i * stride[j]
                del shape[j]
                del stride[j]
            elif i is None:
                shape[j:j] = 1,
                stride[j:j] = stride[j]
                j += 1
            else:
                start = 0 if i.start is None else min(max(i.start + (i.start < 0) * shape[j], 0), shape[j] - 1)
                stop = shape[j] if i.stop is None else min(max(i.stop + (i.stop < 0) * shape[j], 1), shape[j])
                step = i.step or 1
                offset += start * stride[j]
                shape[j] = (stop - start) // step
                stride[j] *= step
                j += 1
        return Tensor(ggml_view_4d(GRAPHCTX, self.base, *rjust(shape), *rjust(stride, v=stride[-1])[1:], offset)).named('getitem')

    @debug
    def cat(self, _: 'Tensor', i) -> 'Tensor':
        self, _ = self.expand(_, i)
        return Tensor(ggml_concat(GRAPHCTX, self.base, _.base, i)).named('cat')

    @debug
    def rcat(self, _: 'Tensor', i) -> 'Tensor':
        self, _ = self.expand(_, i)
        return Tensor(ggml_concat(GRAPHCTX, _.base, self.base, i)).named('rcat')

    def asnp(self) -> np.ndarray:
        return np.lib.stride_tricks.as_strided(np.asarray(ccasserteq(self.base.contents.data, POINTER(c_float * 0)).contents).view(self.asdtype[self.base.contents.type]), self.shape[::-1], self.stride[::-1])

    def tonp(self) -> np.ndarray:
        return to_numpy(self.base)

    def setnp(self, x: np.ndarray) -> None:
        asserteq(x.dtype, self.asdtype[self.base.contents.type])
        asserteq(self.shape[::-1], ljust(x.shape))
        assert ggml_is_contiguous(self.base)
        x = np.ascontiguousarray(x)
        ggml_backend_tensor_set(self.base, x.ctypes.data, 0, x.nbytes)

    @staticmethod
    def fromnp(x: np.ndarray) -> 'Tensor':
        return Tensor(from_numpy(x, GRAPHCTX)).named('fromnp')

    def gelu_(self) -> 'Tensor':
        return Tensor(ggml_gelu_inplace(GRAPHCTX, self.base)).named('gelu')

    def silu_(self) -> 'Tensor':
        return Tensor(ggml_silu_inplace(GRAPHCTX, self.base)).named('silu')

    def softmax(self, scale) -> 'Tensor':
        return Tensor(ggml_soft_max_ext(GRAPHCTX, self.base, None, scale, 0.)).named('softmax')

    @debug
    def add_(self, _: 'Tensor') -> 'Tensor':
        self.expandshape(_)
        return Tensor(ggml_add_inplace(GRAPHCTX, self.untranspose().base, _.untranspose().base)).named('add')

    @debug
    def add(self, _: 'Tensor') -> 'Tensor':
        self.expandshape(_)
        return Tensor(ggml_add(GRAPHCTX, self.untranspose().base, _.untranspose().base)).named('add')

    @debug
    def sub_(self, _: 'Tensor') -> 'Tensor':
        self.expandshape(_)
        return Tensor(ggml_sub_inplace(GRAPHCTX, self.untranspose().base, _.untranspose().base)).named('sub')

    @debug
    def sub(self, _: 'Tensor') -> 'Tensor':
        self.expandshape(_)
        return Tensor(ggml_sub(GRAPHCTX, self.untranspose().base, _.untranspose().base)).named('sub')

    @debug
    def mul_(self, _: 'Tensor') -> 'Tensor':
        self.expandshape(_)
        return Tensor(ggml_mul_inplace(GRAPHCTX, self.untranspose().base, _.untranspose().base)).named('mul')

    @debug
    def mul(self, _: 'Tensor') -> 'Tensor':
        self.expandshape(_)
        return Tensor(ggml_mul(GRAPHCTX, self.untranspose().base, _.untranspose().base)).named('mul')

    @debug
    def dot(self, _: 'Tensor') -> 'Tensor':
        self.expandshape(_, 1)
        return Tensor(ggml_mul_mat(GRAPHCTX, _.untranspose().base, self.untranspose().base)).named('dot')

    @debug
    def scale_(self, _: str) -> 'Tensor':
        return self.mul_(Tensor(_ + '.weight')).add_(Tensor(_ + '.bias')).named('scale')

    @debug
    def scale(self, _: str) -> 'Tensor':
        return self.mul(Tensor(_ + '.weight')).add_(Tensor(_ + '.bias')).named('scale')

    @debug
    def norm_(self, _: str) -> 'Tensor':
        return Tensor(ggml_norm_inplace(GRAPHCTX, self.base, 1e-6)).scale_(_).named('norm')

    @debug
    def norm(self, _: str) -> 'Tensor':
        return Tensor(ggml_norm(GRAPHCTX, self.base, 1e-6)).scale_(_).named('norm')

    @debug
    def lin(self, *_: str) -> 'Tensor':
        for nxt, _ in zip(range(len(_))[::-1], _):
            if _:
                try:
                    self = self.dot(Tensor(_ + '.weight'))
                except KeyError:
                    if not nxt:
                        raise
                    info('%s.weight no found, try next', _)
                    continue
                try:
                    self = self.add_(Tensor(_ + '.bias'))
                except KeyError:
                    warning('%s.bias no found, ignore', _)
            return self.named('lin')
        raise

    @debug
    def mlp(self, _: str) -> 'Tensor':
        return self.lin(_ + '.fc1', _ + '.weights_in', _ + '.up_proj').gelu_().lin(_ + '.fc2', _ + '.weights_out', _ + '.down_proj').named('mlp')

    @debug
    def swiglu(self, _: str) -> 'Tensor':
        x = self.lin(_ + '.fc1', _ + '.weights_in', _ + '.up_proj')
        a = x[:x.shape[0] // 2].silu_()
        b = x[x.shape[0] // 2:]
        return a.mul_(b).lin(_ + '.fc2', _ + '.weights_out', _ + '.down_proj').named('swiglu')

    @debug
    def mlpOrSwiglu(self, _: str):
        try:
            return self.mlp(_)
        except AssertionError:
            warning('mlp no found, try swiglu')
            return self.swiglu(_)

    @debug
    def conv(self, _: 'Tensor') -> 'Tensor':
        asserteq(self.shape[2], _.shape[2])
        return Tensor(ggml_conv_2d_sk_p0(GRAPHCTX, _.base, self.base)).named('conv')

    @debug
    def at(self, _: str) -> 'Tensor':
        try:
            x = self.lin(_ + '.attention.qkv').flatten(0, 0, (-1, 3))  # cg 3 wh n
            q = x[:, 0].named('q')  # cg m+wh n
            k = x[:, 1].named('k')  # cg m+wh n
            v = x[:, 2].named('v')  # cg m+wh n
        except KeyError:
            q = self.lin(_ + '.attention.q_proj').named('q')  # cg m+wh n
            k = self.lin(_ + '.attention.k_proj').named('k')  # cg m+wh n
            v = self.lin(_ + '.attention.v_proj').named('v')  # cg m+wh n
        q = q.flatten(0, 0, (-1, HEAD)).T(1, 2).named('q_')  # c m+wh0 g n
        k = k.flatten(0, 0, (-1, HEAD)).T(1, 2).named('k_')  # c m+wh1 g n
        qk = q.dot(k).softmax(1 / q.shape[0] ** .5).named('qk')  # m+wh1 m+wh0 g n
        v = v.flatten(0, 0, (-1, HEAD)).permute(1, 2, 0).named('v_')  # m+wh1 c g n
        return qk.dot(v).named('qkv').T(1, 2).flatten(0, 1).lin(_ + '.output.dense', _ + '.attention.o_proj').named('at')  # cg m+wh0 n

    @debug
    def rope(self, H: 'Tensor', h: int) -> 'Tensor':
        asserteq((self.shape[2], 1, 1, 1), H.shape)
        # return Tensor(ggml_rope(GRAPHCTX, self.base, H.base, self.shape[0] // HEAD, 0, h)).named('rope')
        return Tensor(ggml_rope_custom_inplace(GRAPHCTX, self.base, H.base, self.shape[0] // HEAD, 0, h, h, 10000., 1., 1., 1., 32., 1.)).named('rope')

    @debug
    def rope2(self, H: 'Tensor', W: 'Tensor', h: int, w: int) -> 'Tensor':
        return self[:self.shape[0] // 2].rope(H, h).cat(self[self.shape[0] // 2:].T(1, 2).rope(W, w).T(1, 2), 0).named('rope2')

    @debug
    def atrope(self, _: str, H: 'Tensor', W: 'Tensor', h: int, w: int, m: int) -> 'Tensor':
        try:
            x = self.lin(_ + '.attention.qkv').flatten(0, 0, (-1, 3))  # cg 3 m+wh n
            q = x[:, 0].named('q')  # cg m+wh n
            k = x[:, 1].named('k')  # cg m+wh n
            v = x[:, 2].named('v')  # cg m+wh n
        except KeyError:
            q = self.lin(_ + '.attention.q_proj').named('q')  # cg m+wh n
            k = self.lin(_ + '.attention.k_proj').named('k')  # cg m+wh n
            v = self.lin(_ + '.attention.v_proj').named('v')  # cg m+wh n
        q = q[:, m:].flatten(1, 1, (-1, H.shape[0])).rope2(H, W, h, w).flatten(1, 2).rcat(q[:, :m], 1)
        q = q.flatten(0, 0, (-1, HEAD)).T(1, 2).named('q_')  # c m+wh0 g n
        k = k[:, m:].flatten(1, 1, (-1, H.shape[0])).rope2(H, W, h, w).flatten(1, 2).rcat(k[:, :m], 1)
        k = k.flatten(0, 0, (-1, HEAD)).T(1, 2).named('k_')  # c m+wh1 g n
        qk = q.dot(k).softmax(1 / q.shape[0] ** .5).named('qk')  # m+wh1 m+wh0 g n
        v = v.flatten(0, 0, (-1, HEAD)).permute(1, 2, 0).named('v_')  # m+wh1 c g n
        return qk.dot(v).named('qkv').T(1, 2).flatten(0, 1).lin(_ + '.output.dense', _ + '.attention.o_proj').named('atrope')  # cg m+wh0 n


@debug
def init(model: str, gpu=False) -> None:
    global MODEL, GPU, GGUFCTX, CTX, BACKEND, ALLOC, HEAD, GRAPHCTX
    if model == MODEL and gpu == GPU:
        return
    deinit()
    MODEL = model
    GPU = gpu
    tmpctx: _Pointer[ggml_context_p] = POINTER(ggml_context_p_ctypes)(c_void_p(0))
    GGUFCTX = gguf_init_from_file(model.encode(), gguf_init_params(False, tmpctx))
    CTX = ggml_init(ggml_init_params(ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(), None, True))
    for i in range(gguf_get_n_tensors(GGUFCTX)):
        name = gguf_get_tensor_name(GGUFCTX, i)
        src = getensor(tmpctx.contents, name)
        info('%s', objstr(Tensor(src)))
        ggml_set_name(ggml_dup_tensor(CTX, src), name)
    BACKEND = ggml_backend_cuda_init() if gpu else ggml_backend_cpu_set_n_threads(_ := ggml_backend_cpu_init(), 28) or _
    ggml_backend_alloc_ctx_tensors(CTX, BACKEND)
    for i in range(gguf_get_n_tensors(GGUFCTX)):
        name = gguf_get_tensor_name(GGUFCTX, i)
        TOUCH.add(name)
        src = getensor(tmpctx.contents, name)
        dst = getensor(CTX, name)
        ggml_backend_tensor_set(dst, ggml_get_data(src), 0, ggml_nbytes(src))
    ggml_free(tmpctx.contents)
    ALLOC = ggml_gallocr_new(ggml_backend_get_default_buffer_type(BACKEND))
    GRAPHCTX = CTX  # ggml_init(ggml_init_params(ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(), None, True))
    HEAD = getkey(GGUFCTX, 'num_attention_heads')


@debug
def deinit() -> None:
    global GGUFCTX, CTX, BACKEND, ALLOC, GRAPHCTX
    if GGUFCTX is None:
        return
    warning('%s no touch', TOUCH)
    TOUCH.clear()
    # ggml_free(GRAPHCTX)
    ggml_gallocr_free(ALLOC)
    ggml_backend_free(BACKEND)
    ggml_free(CTX)
    gguf_free(GGUFCTX)
    GRAPHCTX = GGUFCTX = CTX = BACKEND = ALLOC = None


@debug
def initgraph(y: Tensor) -> None:
    global Y, GF
    if Y == y:
        return
    GF = ggml_new_graph(GRAPHCTX)
    ggml_build_forward_expand(GF, y.base)
    ggml_gallocr_alloc_graph(ALLOC, GF)


def run() -> None:
    assert ggml_backend_graph_compute(BACKEND, GF) == GGML_STATUS_SUCCESS


basicConfig(level='INFO')
if getenv('NOIMP'):
    ggml_add_inplace = ggml_add
    ggml_sub_inplace = ggml_sub
    ggml_mul_inplace = ggml_mul
    ggml_norm_inplace = ggml_norm
    ggml_silu_inplace = ggml_silu
    ggml_gelu_inplace = ggml_gelu
GGUFCTX: gguf_context_p = None
CTX: ggml_context_p = None
BACKEND: ggml_backend_t = None
ALLOC: ggml_gallocr = None
GRAPHCTX: ggml_context_p = None
TOUCH = set[bytes]()
HEAD = 0
MODEL = ''
GPU = False
Y: Tensor = None
register(deinit)
