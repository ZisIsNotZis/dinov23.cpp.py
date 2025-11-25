#!/bin/env python
from sys import argv

from transformers import AutoModel, AutoConfig
from gguf import GGUFWriter, GGUFValueType

for arg in argv[1:]:
    TYPE = {str: GGUFValueType.STRING, int: GGUFValueType.INT32, float: GGUFValueType.FLOAT32, bool: GGUFValueType.BOOL, list: GGUFValueType.ARRAY, None: None}
    f = GGUFWriter(arg.rstrip('/') + '-f16.gguf', arg.split('-')[0])
    for i, j in AutoConfig.from_pretrained(arg).to_dict().items():
        j = list(j) if isinstance(j, dict) else j
        if not j and (j is None or isinstance(j, list)):
            continue
        print(i, j)
        f.add_key_value(i, j, TYPE[type(j)], TYPE[type(j[0]) if isinstance(j, list) else None])
    for i, j in AutoModel.from_pretrained(arg).named_parameters():
        j = j.half() if sum(k > 1 for k in j.shape) > 1 and 'weight' in i else j
        print(i, j.shape, j.dtype)
        f.add_tensor(i, j.data.numpy())
    f.write_header_to_file()
    f.write_kv_data_to_file()
    f.write_tensors_to_file()
