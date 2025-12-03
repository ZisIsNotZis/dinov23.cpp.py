from ctypes import POINTER, c_void_p, c_float, cast as ccast, _Pointer
from math import prod
from logging import info, warning, basicConfig
from atexit import register
from functools import wraps, lru_cache
from os import getenv

from ggml import *
from ggml.utils import to_numpy, from_numpy
import numpy as np


def log[**P, T](f: Callable[P, T], lv=[0]) -> Callable[P, T]:
    @wraps(f)
    def log(*args: P.args, **kwargs: P.kwargs) -> T:
        info('%s%s(%s)', '\t' * lv[0], f.__name__, ', '.join(map(objstr, args)))
        lv[0] += 1
        try:
            t = f(*args, **kwargs)
            info(' %s%s', '\t' * (lv[0] - 1), objstr(t))
            return t
        finally:
            lv[0] -= 1

    return log


type Tape[*V] = tuple[Callable[[ggml_context_p, ggml_tensor_p, *V], ggml_tensor_p], ggml_tensor_p, *V]
type Tape2[*V] = Tape[ggml_tensor_p, *V]


def tape[**P, *V](f: Callable[P, Tape[*V]]) -> Callable[P, 'Tensor']:
    @wraps(f)
    def tape(*args: P.args, **kwargs: P.kwargs) -> 'Tensor':
        t = f(*args, **kwargs)
        a = t[0](GRAPHCTX, *t[1:])
        if getenv('TAPE'):
            print('_%s=%s(%s)' % (a.contents.data, t[0].__name__, ','.join(f'_{i.contents.data}' if isinstance(i, _Pointer) else str(i) for i in t[1:])))
        return Tensor(a)

    return tape


def id(_: ggml_context_p, a: ggml_tensor_p, *__) -> ggml_tensor_p:
    return a


def asserteq[T](a: T, b: T) -> None:
    assert a == b, (a, b)


@log
def getint(key: str, dft: int | None = None) -> int:
    if (t := gguf_find_key(GGUFCTX, key.encode())) < 0:
        if dft is not None:
            return dft
        raise KeyError(key)
    return gguf_get_val_i32(GGUFCTX, t) if gguf_get_kv_type(GGUFCTX, t) == GGUF_TYPE_INT32 else gguf_get_val_u32(GGUFCTX, t)


@log
def getfloat(key: str, dft: float | None = None) -> float:
    if (t := gguf_find_key(GGUFCTX, key.encode())) < 0:
        if dft is not None:
            return dft
        raise KeyError(key)
    return gguf_get_val_f32(GGUFCTX, t) if gguf_get_kv_type(GGUFCTX, t) == GGUF_TYPE_FLOAT32 else gguf_get_val_f64(GGUFCTX, t)


def getstr(key: str, dft: str | None = None) -> str:
    if (t := gguf_find_key(GGUFCTX, key.encode())) < 0:
        if dft is not None:
            return dft
        raise KeyError(key)
    return gguf_get_val_str(GGUFCTX, t).decode()


def getensor(ctx: ggml_context_p, tensor: str | bytes) -> ggml_tensor_p:
    if not (t := ggml_get_tensor(ctx, (tensor.encode() if isinstance(tensor, str) else tensor))):
        raise KeyError(tensor)
    return t


def typestr(type: int) -> str:
    return [i for i, j in globals().items() if i.startswith('GGML_TYPE_') and j == type][0][10:]


def objstr(a: object) -> str:
    return a.name + '[' + ','.join(map(str, a.shape)) + ']' if isinstance(a, Tensor) else a.dtype.char + '[' + ','.join(map(str, a.shape)) + ']' if isinstance(a, np.ndarray) else str(a)


def rjust(a: tuple[int, ...] | list[int], v=1) -> tuple[int, int, int, int]:
    return tuple(a[:4]) + (v,) * (4 - len(a))


class Tensor:
    asdtype = {GGML_TYPE_I8: np.dtype('b'), GGML_TYPE_I16: np.dtype('h'), GGML_TYPE_I32: np.dtype('i'), GGML_TYPE_I64: np.dtype('l'), GGML_TYPE_F64: np.dtype('d'), GGML_TYPE_F32: np.dtype('f'), GGML_TYPE_F16: np.dtype('e')}

    def __init__(self, a: 'str | tuple[int,...] | ggml_tensor_p', type=GGML_TYPE_F32):
        self.base = getensor(CTX, a) if isinstance(a, str) else ggml_new_tensor_4d(GRAPHCTX, type, *a, *[1] * (4 - len(a))) if isinstance(a, (tuple, list)) else a
        try:
            TOUCH.remove(self.base.contents.name)
        except KeyError:
            pass

    @property
    def name(self) -> str:
        return self.base.contents.name.decode()

    @property
    def type(self) -> int:
        return self.base.contents.type

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return tuple[int, int, int, int](map(int, self.base.contents.ne))

    @property
    def stride(self) -> tuple[int, int, int, int]:
        return tuple[int, int, int, int](map(int, self.base.contents.nb))

    def named(self, name: str) -> 'Tensor':
        return self

    @log
    @tape
    def flatten(self, i=0, j=3, s=(-1,)) -> Tape[int, int, int, int, int, int, int, int]:
        self = self.untranspose()
        shape = list(self.shape)
        stride = list(self.stride)
        s = list(s)
        if np.multiply(stride[i], shape[i:j]).tolist() == stride[i + 1:j + 1]:
            self = self.cont()
            stride = list(self.stride)
        if -1 in s:
            k = s.index(-1)
            s[k] = prod(shape[i:j + 1]) // prod(s[:k] + s[k + 1:])
        shape[i:j + 1] = s
        stride[i:j + 1] = np.multiply(stride[i], [1, *s[:-1]])
        return ggml_view_4d, self.base, *rjust(shape), *rjust(stride, stride[-1])[1:], 0

    @log
    @tape
    def permute(self, i=0, j=1, k=2, l=3) -> Tape[int, int, int, int]:
        return ggml_permute, self.base, i, j, k, l

    def T(self, i=0, j=1) -> 'Tensor':
        dim = list(range(4))
        dim[i] = j
        dim[j] = i
        return self.permute(*dim)

    def expandshape(self, a: 'Tensor', ex=4) -> tuple[int, int, int, int]:
        assert all(i == ex or a == 1 or b == 1 or a == b for i, (a, b) in enumerate(zip(self.shape, a.shape))), (self.shape, a.shape)
        return tuple[int, int, int, int](a if i == ex else max(a, b) for i, (a, b) in enumerate(zip(self.shape, a.shape)))

    @log
    @tape
    def expand(self, a: 'Tensor', ex=4) -> Tape2:
        shape = self.expandshape(a, ex)
        return (id, self.base, self.base) if shape == self.shape else (ggml_repeat, self.base, Tensor(shape).base)

    @tape
    def cont(self) -> Tape:
        return (id, self.base) if ggml_is_contiguous(self.base) else (ggml_cont, self.base)

    @tape
    def untranspose(self) -> Tape:
        return (ggml_cont, self.base) if ggml_is_transposed(self.base) else (id, self.base)

    @log
    @tape
    def get(self, *I: int | None | slice) -> Tape[int, int, int, int, int, int, int, int]:
        self = self.untranspose()
        shape = list(self.shape)
        stride = list(self.stride)
        offset = 0
        j = 0
        for i in I:
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
        return ggml_view_4d, self.base, *rjust(shape), *rjust(stride, v=stride[-1])[1:], offset

    def __getitem__(self, i: int | None | slice | tuple[int | None | slice]) -> 'Tensor':
        return self.get(*i if isinstance(i, tuple) else [i])

    @log
    @tape
    def cat(self, a: 'Tensor', i) -> Tape2[int]:
        return ggml_concat, self.expand(a, i).base, a.expand(self, i).base, i

    @log
    @tape
    def rcat(self, a: 'Tensor', i) -> Tape2[int]:
        return ggml_concat, a.expand(self, i).base, self.expand(a, i).base, i

    def asnp(self) -> np.ndarray:
        return np.lib.stride_tricks.as_strided(np.asarray(ccast(self.base.contents.data, POINTER(c_float * 0)).contents).view(self.asdtype[self.base.contents.type]), self.shape[::-1], self.stride[::-1])

    def tonp(self) -> np.ndarray:
        return to_numpy(self.base)

    def setnp(self, x: np.ndarray) -> None:
        asserteq(x.dtype, self.asdtype[self.base.contents.type])
        asserteq(self.shape, rjust(x.shape[::-1]))
        assert ggml_is_contiguous(self.base)
        x = np.ascontiguousarray(x)
        ggml_backend_tensor_set(self.base, x.ctypes.data, 0, x.nbytes)

    @staticmethod
    def fromnp(x: np.ndarray) -> 'Tensor':
        return Tensor(from_numpy(x, GRAPHCTX))

    @tape
    def gelu_(self) -> Tape:
        return ggml_gelu_inplace, self.base

    @tape
    def silu_(self) -> Tape:
        return ggml_silu_inplace, self.base

    @tape
    def gelu(self) -> Tape:
        return ggml_gelu, self.base

    @tape
    def silu(self) -> Tape:
        return ggml_silu, self.base

    def act(self, act='') -> 'Tensor':
        return getattr(self, act or ACT)()

    def act_(self, act='') -> 'Tensor':
        return getattr(self, act or ACT + '_')()

    @tape
    def softmax(self, scale) -> Tape2[float, float]:
        return ggml_soft_max_ext, self.base, None, scale, 0.

    @log
    @tape
    def add_(self, a: 'Tensor') -> Tape2:
        self.expandshape(a)
        return ggml_add_inplace, self.untranspose().base, a.untranspose().base

    @log
    @tape
    def add(self, a: 'Tensor') -> Tape2:
        self.expandshape(a)
        return ggml_add, self.untranspose().base, a.untranspose().base

    @log
    @tape
    def sub_(self, a: 'Tensor') -> Tape2:
        self.expandshape(a)
        return ggml_sub_inplace, self.untranspose().base, a.untranspose().base

    @log
    @tape
    def sub(self, a: 'Tensor') -> Tape2:
        self.expandshape(a)
        return ggml_sub, self.untranspose().base, a.untranspose().base

    @log
    @tape
    def mul_(self, a: 'Tensor') -> Tape2:
        self.expandshape(a)
        return ggml_mul_inplace, self.untranspose().base, a.untranspose().base

    @tape
    def neg(self) -> Tape:
        return ggml_neg, self.cont().base

    @log
    @tape
    def mul(self, a: 'Tensor') -> Tape2:
        self.expandshape(a)
        return ggml_mul, self.untranspose().base, a.untranspose().base

    @log
    @tape
    def dot(self, a: 'Tensor') -> Tape2:
        self.expandshape(a, 1)
        return ggml_mul_mat, a.untranspose().base, self.untranspose().base

    @log
    @tape
    def conv2dskp0(self, a: 'Tensor') -> Tape2:
        asserteq(self.shape[2], a.shape[2])
        return ggml_conv_2d_sk_p0, a.base, self.cont().base

    @log
    @tape
    def conv2ds1ph(self, a: 'Tensor') -> Tape2:
        asserteq(self.shape[2], a.shape[2])
        return ggml_conv_2d_s1_ph, a.base, self.cont().base

    @log
    @tape
    def conv2dcustom(self, a: 'Tensor', s0=1, s1=1, p0=1, p1=1, d0=1, d1=1) -> Tape2[int, int, int, int, int, int]:
        asserteq(self.shape[2], a.shape[2])
        return ggml_conv_2d, a.base, self.cont().base, s0, s1, p0, p1, d0, d1

    def conv2d(self, a: 'Tensor', s0=1, s1=1, p0=0, p1=0, d0=1, d1=1) -> 'Tensor':
        if s0 == a.shape[0] and s1 == a.shape[1] and p0 == p1 == 0 and d0 == d1 == 1:
            return self.conv2dskp0(a)
        if s0 == s1 == 1 and p0 == a.shape[0] // 2 and p1 == a.shape[1] // 2 and d0 == d1 == 1:
            return self.conv2ds1ph(a)
        return self.conv2dcustom(a, s0, s1, p0, p1, d0, d1)

    @log
    @tape
    def conv2dw(self, a: 'Tensor', s0=1, s1=1, p0=0, p1=0, d0=1, d1=1) -> Tape2[int, int, int, int, int, int]:
        asserteq(self.shape[2], a.shape[3])
        return ggml_conv_2d_dw, a.base, self.cont().base, s0, s1, p0, p1, d0, d1

    def conv2dnhwc(self, key: str, s0=1, s1=1, p0=0, p1=0, d0=1, d1=1) -> 'Tensor':
        self = self.permute(2, 0, 1).conv2d(Tensor(key + '.weight'), s0, s1, p0, p1, d0, d1).permute(1, 2, 0)
        try:
            self = self.add_(Tensor(key + '.bias'))
        except KeyError:
            warning('%s.bias no found, skip', key)
        return self.named('conv2dbias')

    def conv2dwnhwc(self, key: str, s0=1, s1=1, p0=0, p1=0, d0=1, d1=1) -> 'Tensor':
        self = self.permute(2, 0, 1).conv2dw(Tensor(key + '.weight'), s0, s1, p0, p1, d0, d1).permute(1, 2, 0)
        try:
            self = self.add_(Tensor(key + '.bias'))
        except KeyError:
            warning('%s.bias no found, skip', key)
        return self.named('conv2dwbias')

    @log
    @tape
    def pool2d(self) -> Tape[int, int, int, int, int, float, float]:
        return ggml_pool_2d, self.permute(2, 0, 1).base, GGML_OP_POOL_AVG, self.shape[1], self.shape[2], 1, 1, 0., 0.

    @tape
    def norm_(self) -> Tape[float]:
        return ggml_norm_inplace, self.base, 1e-6

    @tape
    def norm(self) -> Tape[float]:
        return ggml_norm, self.base, 1e-6

    @log
    def rotatehalf(self) -> 'Tensor':
        return self[self.shape[0] // 2:].neg().cat(self[:self.shape[0] // 2], 0)

    @tape
    def rope(self, H: 'Tensor', h: int) -> Tape2[int, int, int]:
        asserteq((self.shape[2], 1, 1, 1), H.shape)
        return ggml_rope_custom_inplace, self.base, H.base, self.shape[0] // HEAD, 0, h, h, 10000., 1., 1., 1., 32., 1.

    @log
    def rope2d(self, H: 'Tensor', W: 'Tensor', h: int, w: int) -> 'Tensor':
        return self[:self.shape[0] // 2].rope(H, h).cat(self[self.shape[0] // 2:].T(1, 2).rope(W, w).T(1, 2), 0).named('rope2')

    @log
    def rope2dcosin(self, COS: 'Tensor', SIN: 'Tensor') -> 'Tensor':
        return self.mul(COS).add_(self.rotatehalf().mul_(SIN))

    def scale_(self, key: str) -> 'Tensor':
        return self.mul_(Tensor(key + '.weight')).add_(Tensor(key + '.bias')).named('scale')

    def scale(self, key: str) -> 'Tensor':
        return self.mul(Tensor(key + '.weight')).add_(Tensor(key + '.bias')).named('scale')

    def normscale_(self, key: str) -> 'Tensor':
        return self.norm_().scale_(key).named('norm')

    def normscale(self, key: str) -> 'Tensor':
        return self.norm().scale(key).named('norm')

    @log
    def lin(self, *key: str) -> 'Tensor':
        for nxt, key in zip(range(len(key))[::-1], key):
            if key:
                try:
                    self = self.dot(Tensor(key + '.weight'))
                except KeyError:
                    if not nxt:
                        raise
                    info('%s.weight no found, try next', key)
                    continue
                try:
                    self = self.add_(Tensor(key + '.bias'))
                except KeyError:
                    warning('%s.bias no found, skip', key)
            return self.named('lin')
        raise

    @log
    def mlp(self, key: str, act='') -> 'Tensor':
        return self.lin(key + '1', key + 'in', key + 'up_proj').act_(act).lin(key + '2', key + 'out', key + 'down_proj').named('mlp')

    @log
    def swiglu(self, key: str, act='') -> 'Tensor':
        a = self.lin(key + '1', key + 'in', key + 'up_proj')
        try:
            b = self.lin(key + 'gate_proj')
        except KeyError:
            b = a[:a.shape[0] // 2].act_(act)
            a = a[a.shape[0] // 2:]
        return a.mul_(b).lin(key + '2', key + 'out', key + 'down_proj').named('swiglu')

    def mlpOrSwiglu(self, key: str):
        try:
            return self.swiglu(key)
        except AssertionError:
            return self.mlp(key)

    @log
    def at(self, key: str) -> 'Tensor':
        try:
            x = self.lin(key + '.attention.qkv').flatten(0, 0, (-1, 3))  # cg 3 wh n
            q = x[:, 0].named('q')  # cg m+wh n
            k = x[:, 1].named('k')  # cg m+wh n
            v = x[:, 2].named('v')  # cg m+wh n
            warning('%s.attention.qkv no found, try q/k/v_proj')
        except KeyError:
            q = self.lin(key + '.attention.q_proj').named('q')  # cg m+wh n
            k = self.lin(key + '.attention.k_proj').named('k')  # cg m+wh n
            v = self.lin(key + '.attention.v_proj').named('v')  # cg m+wh n
        q = q.flatten(0, 0, (-1, HEAD)).T(1, 2).named('q_')  # c m+wh0 g n
        k = k.flatten(0, 0, (-1, HEAD)).T(1, 2).named('k_')  # c m+wh1 g n
        qk = q.dot(k).softmax(1 / q.shape[0] ** .5).named('qk')  # m+wh1 m+wh0 g n
        v = v.flatten(0, 0, (-1, HEAD)).permute(1, 2, 0).named('v_')  # m+wh1 c g n
        return qk.dot(v).named('qkv').T(1, 2).flatten(0, 1).lin(key + '.output.dense', key + '.attention.o_proj').named('at')  # cg m+wh0 n

    @log
    def atrope(self, key: str, H: 'Tensor', W: 'Tensor', h: int, w: int, m: int) -> 'Tensor':
        try:
            x = self.lin(key + '.attention.qkv').flatten(0, 0, (-1, 3))  # cg 3 m+wh n
            q = x[:, 0].named('q')  # cg m+wh n
            k = x[:, 1].named('k')  # cg m+wh n
            v = x[:, 2].named('v')  # cg m+wh n
            warning('%s.attention.qkv no found, try q/k/v_proj')
        except KeyError:
            q = self.lin(key + '.attention.q_proj').named('q')  # cg m+wh n
            k = self.lin(key + '.attention.k_proj').named('k')  # cg m+wh n
            v = self.lin(key + '.attention.v_proj').named('v')  # cg m+wh n
        q = q[:, m:].flatten(1, 1, (-1, H.shape[0])).rope2d(H, W, h, w).flatten(1, 2).rcat(q[:, :m], 1)
        q = q.flatten(0, 0, (-1, HEAD)).T(1, 2).named('q_')  # c m+wh0 g n
        k = k[:, m:].flatten(1, 1, (-1, H.shape[0])).rope2d(H, W, h, w).flatten(1, 2).rcat(k[:, :m], 1)
        k = k.flatten(0, 0, (-1, HEAD)).T(1, 2).named('k_')  # c m+wh1 g n
        qk = q.dot(k).softmax(1 / q.shape[0] ** .5).named('qk')  # m+wh1 m+wh0 g n
        v = v.flatten(0, 0, (-1, HEAD)).permute(1, 2, 0).named('v_')  # m+wh1 c g n
        return qk.dot(v).named('qkv').T(1, 2).flatten(0, 1).lin(key + '.output.dense', key + '.attention.o_proj').named('atrope')  # cg m+wh0 n

    @log
    def atropecosin(self, key: str, COS: 'Tensor', SIN: 'Tensor') -> 'Tensor':
        try:
            x = self.lin(key + '.attention.qkv').flatten(0, 0, (-1, 3))  # cg 3 m+wh n
            q = x[:, 0].named('q')  # cg m+wh n
            k = x[:, 1].named('k')  # cg m+wh n
            v = x[:, 2].named('v')  # cg m+wh n
            warning('%s.attention.qkv no found, try q/k/v_proj')
        except KeyError:
            q = self.lin(key + '.attention.q_proj').named('q')  # cg m+wh n
            k = self.lin(key + '.attention.k_proj').named('k')  # cg m+wh n
            v = self.lin(key + '.attention.v_proj').named('v')  # cg m+wh n
        q = q.flatten(0, 0, (-1, HEAD)).T(1, 2)
        q = q[:, -COS.shape[1]:].rope2dcosin(COS, SIN).rcat(q[:, :-COS.shape[1]], 1).named('q_')  # c m+wh0 g n
        k = k.flatten(0, 0, (-1, HEAD)).T(1, 2)
        k = k[:, -COS.shape[1]:].rope2dcosin(COS, SIN).rcat(k[:, :-COS.shape[1]], 1).named('k_')  # c m+wh1 g n
        qk = q.dot(k).softmax(1 / q.shape[0] ** .5).named('qk')  # m+wh1 m+wh0 g n
        v = v.flatten(0, 0, (-1, HEAD)).permute(1, 2, 0).named('v_')  # m+wh1 c g n
        return qk.dot(v).named('qkv').T(1, 2).flatten(0, 1).lin(key + '.output.dense', key + '.attention.o_proj').named('atrope')  # cg m+wh0 n


def init(model: str, gpu=False) -> None:
    global MODEL, GPU, GGUFCTX, CTX, BACKEND, ALLOC, HEAD, ACT, GRAPHCTX
    if model == MODEL and gpu == GPU:
        return
    MODEL = model
    GPU = gpu
    deinit()
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
    HEAD = getint('num_attention_heads', 0)
    ACT = getstr('hidden_act', 'gelu').lower()


def deinit() -> None:
    global GGUFCTX, CTX, BACKEND, ALLOC, GRAPHCTX
    if GGUFCTX is None:
        return
    # ggml_free(GRAPHCTX)
    ggml_gallocr_free(ALLOC)
    ggml_backend_free(BACKEND)
    ggml_free(CTX)
    gguf_free(GGUFCTX)
    GRAPHCTX = GGUFCTX = CTX = BACKEND = ALLOC = None
    if TOUCH:
        warning('%s no use', TOUCH)
        TOUCH.clear()


def initgraph(*y: Tensor) -> None:
    global Y, GF
    if Y == y:
        return
    Y = y
    GF = ggml_new_graph(GRAPHCTX)
    for y in y:
        ggml_build_forward_expand(GF, y.base)
    ggml_gallocr_alloc_graph(ALLOC, GF)


def run() -> None:
    assert ggml_backend_graph_compute(BACKEND, GF) == GGML_STATUS_SUCCESS


GGUFCTX: gguf_context_p = None
CTX: ggml_context_p = None
BACKEND: ggml_backend_t = None
ALLOC: ggml_gallocr = None
GRAPHCTX: ggml_context_p = None
TOUCH = set[bytes]()
HEAD = -1
ACT = MODEL = 'undefined'
GPU = False
Y = tuple[Tensor]()
register(deinit)
basicConfig(level='INFO')
if getenv('NOINPLACE'):
    ggml_add_inplace = ggml_add
    ggml_sub_inplace = ggml_sub
    ggml_mul_inplace = ggml_mul
    ggml_norm_inplace = ggml_norm
    ggml_silu_inplace = ggml_silu
    ggml_gelu_inplace = ggml_gelu
