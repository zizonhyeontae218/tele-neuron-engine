"""Explicit 3T name for the historical Endophalon 3 single-head family.

Historical endophalon3 modules, serialized family strings and file paths remain
unchanged. This alias identifies those models without rewriting evidence.
"""
FAMILY='Teleneuron-Endophalon-3T'

def variant_name(level):
    return f'{FAMILY}-{level}'
