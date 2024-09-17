import os
import sys
import numpy as np

# 現在のスクリプトのディレクトリを取得
current_dir = os.path.dirname(os.path.abspath(__file__))

# calculation_files とその中の setting_files ディレクトリをパスに追加
sys.path.append(os.path.join(current_dir, 'calculation_files'))
sys.path.append(os.path.join(current_dir, 'calculation_files', 'setting_files'))

from scoring_files.setting_files.forward_posture_3d import compare_posture
from scoring_files.setting_files.json2numpy_3d import json2numpy


def sigmoid_score(change, k=1.0, x0=10):
    """
    シグモイド関数を使って得点を計算する関数。

    パラメータ:
    change (float): 距離のパーセンテージ変化
    k (float): シグモイド関数の急峻さを決定するパラメータ
    x0 (float): シグモイド関数の中心点

    戻り値:
    int: 0から100の得点
    """
    abs_change = abs(change)
    
    # シグモイド関数を使って得点を計算
    score = 100 / (1 + np.exp(k * (abs_change - x0)))
    
    return score / 100

if __name__ == "__main__":
    json_path = sys.argv[1]
    address_frame = int(sys.argv[2])
    impact_frame = int(sys.argv[3])
    
    keypoints_array = json2numpy(json_path)
    change = compare_posture(keypoints_array, address_frame, impact_frame)
    
    score = sigmoid_score(change, k=0.5, x0=15)
    
    print(f"アドレスからインパクトへの距離のパーセンテージ変化: {change:.2f}%")
    print(f"得点: {score}")