#!/bin/env python
from os import getenv
from sys import argv
from gguf import GGUFReader, GGUFWriter, GGUFValueType, ReaderField, GGMLQuantizationType, quantize, QuantError


def get(field: ReaderField):
    a = [field.parts[i].tobytes().decode() if field.types[-1] == GGUFValueType.STRING else field.parts[i].item() for i in field.data]
    return a if field.types[0] == GGUFValueType.ARRAY else a[0]


QUANT = [getattr(GGMLQuantizationType, i) for i in getenv('QUANT', 'Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q8_1 F16').split()]
for arg in argv[1:]:
    print(arg)
    in_ = GGUFReader(arg)
    out = GGUFWriter(f'{arg.rsplit('-', 1)[0]}-{QUANT[0].name.lower()}.gguf', get(in_.fields['general.architecture']))
    for field in in_.fields.values():
        if not field.name.startswith('general.'):
            print(field.name, get(field))
            out.add_key_value(field.name, get(field), *field.types)
    for field in in_.tensors:
        a = field.data.squeeze()
        if 'weight' in field.name and a.ndim >= 2:
            for quant in QUANT:
                if a.ndim == 2 or 'Q' not in quant.name:
                    try:
                        out.add_tensor(field.name, quantize(a, quant), None, quant)
                        print(field.name, a.shape, quant.name)
                        break
                    except QuantError:
                        pass
        else:
            out.add_tensor(field.name, field.data)
    out.write_header_to_file()
    out.write_kv_data_to_file()
    out.write_tensors_to_file()
    out.close()
