import warnings


def _get_c_extension():
    try:
        import __ash as _C

        if not hasattr(_C, "HashMap"):
            _C = None
    except Exception as e:
        print(e)
        _C = None

    if _C is None:
        warnings.warn("CUDA extension ash.HashMap could not be loaded!")
    return _C


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
