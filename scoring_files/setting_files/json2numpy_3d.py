import json
import numpy as np
from pathlib import Path

def json2numpy(jsonPath):
    """
    パラメータ:
        jsonPath (Path): 入力JSONファイルのパス

    戻り値:
        numpy.ndarray: 各フレームのキーポイント座標を含むNumPy配列
    """
    with open(jsonPath, "r") as f:
        jsonreader = json.load(f)

    frameCount = len(jsonreader["frames"])
    pointCount = len(jsonreader["frames"][0]["joints"])
    array = np.empty((frameCount, pointCount, 3), dtype=float)

    for i in range(frameCount):
        frame = jsonreader["frames"][i]
        for j in range(pointCount):
            joint = frame["joints"][j]
            coordinates = joint["coordinates"]
            x = coordinates.get("x", np.nan)
            y = coordinates.get("y", np.nan)
            z = coordinates.get("z", np.nan)

            array[i, j, :] = [x, y, z]

    return array