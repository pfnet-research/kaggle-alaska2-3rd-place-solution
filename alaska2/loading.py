import tempfile
from pathlib import Path
from typing import Union

import cv2
import jpegio as jio
import numpy as np
import pfio
from numpy.lib.stride_tricks import as_strided
from scipy import fftpack
from skimage.color import rgb2hed


def _read_with_jio(path: Union[str, Path], method: str) -> np.ndarray:
    path = str(path)

    if method == "DCT":
        jpeg_struct = jio.read(path)
        return np.stack(jpeg_struct.coef_arrays, axis=-1)
    elif method == "DCT_PE_L10":
        jpeg_struct = jio.read(path)
        L = 10
        encodes = [lambda x, l: np.sin(x / 2 ** l * np.pi), lambda x, l: 1 - np.cos(x / 2 ** l * np.pi)]
        arrays = [f(dct, i) for dct in jpeg_struct.coef_arrays for i in range(L) for f in encodes]
        return np.stack(arrays, axis=-1)
    elif method.startswith("DCT_BIN"):
        if method == "DCT_BIN":
            L = 10
        else:
            # method = DCT_BIN_L{n}
            assert method.startswith("DCT_BIN_L")
            L = int(method[len("DCT_BIN_L") :])
        jpeg_struct = jio.read(path)
        dct = np.stack(jpeg_struct.coef_arrays, axis=-1)
        ret = np.zeros(jpeg_struct.coef_arrays[0].shape + (3, 2, L), dtype=np.float32)
        for i in range(L - 1):
            ret[dct == i + 1, 0, i] = 1
            ret[dct == -(i + 1), 1, i] = 1
        ret[dct >= L, 0, L - 1] = 1
        ret[dct <= -L, 1, L - 1] = 1
        return ret.reshape(ret.shape[0], ret.shape[1], -1)
    elif method.startswith("DCT_TRI"):
        if method == "DCT_TRI":
            L = 10
        else:
            # method = DCT_TRI_L{n}
            assert method.startswith("DCT_TRI_L")
            L = int(method[len("DCT_TRI_L") :])

        jpeg_struct = jio.read(path)
        dct = np.stack(jpeg_struct.coef_arrays, axis=-1)
        ret = np.zeros(jpeg_struct.coef_arrays[0].shape + (3, L), dtype=np.float32)
        for i in range(L - 1):
            ret[dct == i + 1, i] = 1
            ret[dct == -(i + 1), i] = -1
        ret[dct >= L, L - 1] = 1
        ret[dct <= -L, L - 1] = -1
        return ret.reshape(ret.shape[0], ret.shape[1], -1)
    elif method == "DCT_LSB":
        dct = _read_with_jio(path, "DCT")
        return dct & 1
    else:
        raise ValueError(f"Unknown method: {method}")


def read_image_from_container(container: pfio.container.Container, path_in_container: str, method: str) -> np.ndarray:
    with container.open(path_in_container, "rb") as fp:
        buf = fp.read()

    if method == "RGB":
        return cv2.cvtColor(cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    elif method == "cv2_YUV" or method == "YUV":
        return cv2.cvtColor(cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2YUV)
    elif method == "XYZ":
        return cv2.cvtColor(cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2XYZ)
    elif method == "YCrCb":
        return cv2.cvtColor(cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2YCrCb)
    elif method == "HSV":
        return cv2.cvtColor(cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2HSV)
    elif method == "HLS":
        return cv2.cvtColor(cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2HLS)
    elif method == "Lab":
        return cv2.cvtColor(cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2Lab)
    elif method == "Luv":
        return cv2.cvtColor(cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2Luv)
    elif method == "HED":
        rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return rgb2hed(rgb.astype(np.float64))

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".jpg", dir="/tmp/ram/") as tmp_jpeg:
        tmp_jpeg.file.write(buf)
        tmp_jpeg.file.flush()

        if method.startswith("RGB_DCT"):
            rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            dct = _read_with_jio(tmp_jpeg.name, method[4:])
            return np.concatenate([rgb, dct], axis=-1)

        return _read_with_jio(tmp_jpeg.name, method)


def read_image(path: Union[str, Path], method: str) -> np.ndarray:
    """
    Args:
        path:
        method:
    Returns:
        img: Channel last. dtype depends on ``method``
    """

    if method == "RGB":
        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    elif method == "cv2_YUV" or method == "YUV":
        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2YUV)
    elif method == "XYZ":
        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2XYZ)
    elif method == "YCrCb":
        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2YCrCb)
    elif method == "HSV":
        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2HSV)
    elif method == "HLS":
        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2HLS)
    elif method == "Lab":
        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2Lab)
    elif method == "Luv":
        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2Luv)
    elif method == "HED":
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        return rgb2hed(rgb.astype(np.float64))
    else:
        if method.startswith("RGB_DCT"):
            rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            dct = _read_with_jio(path, method[4:])
            return np.concatenate([rgb, dct], axis=-1)
        return _read_with_jio(path, method)


# from https://www.kaggle.com/remicogranne/jpeg-explanations
def read_quality_factor(path: Union[str, Path]) -> int:
    jpeg_struct = jio.read(str(path))

    if jpeg_struct.quant_tables[0][0, 0] == 2:
        return 95
    elif jpeg_struct.quant_tables[0][0, 0] == 3:
        return 90
    elif jpeg_struct.quant_tables[0][0, 0] == 8:
        return 75
    else:
        raise ValueError("Unknown quant_tables")


def get_in_channels(method: str):
    if method in [
        "RGB",
        "cv2_YUV",
        "DCT",
        "DCT_LSB",
        "XYZ",
        "YCrCb",
        "HSV",
        "HLS",
        "Lab",
        "Luv",
        "YUV",
        "HED",
    ]:
        return 3
    if method == "DCT_PE_L10":
        return 60
    if method.startswith("DCT_BIN"):
        if method == "DCT_BIN":
            L = 10
        else:
            # method = DCT_BIN_L{n}
            assert method.startswith("DCT_BIN_L")
            L = int(method[len("DCT_BIN_L") :])
        return 3 * L * 2
    if method.startswith("DCT_TRI"):
        if method == "DCT_TRI":
            L = 10
        else:
            # method = DCT_TRI_L{n}
            assert method.startswith("DCT_TRI_L")
            L = int(method[len("DCT_TRI_L") :])
        return 3 * L
    if method.startswith("RGB_DCT"):
        return 3 + get_in_channels(method[4:])
    raise ValueError
