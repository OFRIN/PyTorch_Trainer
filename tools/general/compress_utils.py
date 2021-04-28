import numpy as np

def compress_for_confidence(data):
    return (data * 100).astype(np.int32).astype(np.uint8)

def decompress_for_confidence(data):
    return data.astype(np.float32) / 100.

def compress_for_one_hot(data):
    return np.packbits(data.astype(np.bool))

def decompress_for_one_hot(data, classes):
    return np.unpackbits(data, axis=None)[:classes].reshape(classes).astype(np.float32)