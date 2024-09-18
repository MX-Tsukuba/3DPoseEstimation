import os
import sys
import numpy as np

# 現在のスクリプトのディレクトリを取得
current_dir = os.path.dirname(os.path.abspath(__file__))

# calculation_files とその中の setting_files ディレクトリをパスに追加
sys.path.append(os.path.join(current_dir, 'calculation_files'))
sys.path.append(os.path.join(current_dir, 'calculation_files', 'setting_files'))

from setting_files.forward_posture_3d import compare_posture,avg_posture_angle
from setting_files.json2numpy_3d import json2numpy


def sigmoid_score_body_tilt(angle_avg, k=1.0, x0=10):
    """
    シグモイド関数を使って得点を計算する関数。

    パラメータ:
    angle_avg (float): 体の傾きの平均角度
    k (float): シグモイド関数の急峻さを決定するパラメータ
    x0 (float): シグモイド関数の中心点

    戻り値:
    float: 0から1の得点
    """
    abs_change = abs(angle_avg)
    
    # シグモイド関数を使って得点を計算
    score = 100 / (1 + np.exp(k * (abs_change - x0)))
    
    return score / 100

def score_henchedBack(angle_avg):
    """
    肩が丸まりすぎていないかどうかを判定する関数
    ダメな分だけこの値をマイナスするイメージ。90未満、200以上は警告出すなどの使い方でも可。

    パラメータ:
        angle_avg (float): 首・背中・腰の三点がなす角の角度の平均。

    戻り値:
        float: 0から1の値
    """
    if angle_avg <= 90:
        return 1
    elif 90 < angle_avg <= 135:
        # 90から135の範囲で1から0に向かってなだらかに減少 (コサインカーブで減少)
        return np.cos((angle_avg - 90) * (np.pi / (135 - 90))) / 2 + 0.5
    elif 135 < angle_avg <= 180:
        return 0
    elif 180 < angle_avg <= 200:
        # 180から200の範囲で0から1に向かってなだらかに増加 (コサインカーブで増加)
        return (1 - np.cos((angle_avg - 180) * (np.pi / (200 - 180)))) / 2
    else:
        return 1

if __name__ == "__main__":
    json_path = sys.argv[1]
    address_frame = int(sys.argv[2])
    impact_frame = int(sys.argv[3])
    
    keypoints_array = json2numpy(json_path)
    angle_avg = avg_posture_angle(keypoints_array)
    
    score = sigmoid_score_body_tilt(angle_avg, k=0.5, x0=20)
    # 体が丸まりすぎていないかどうかのスコア
    # henchedBacks = avg_henchedBacks(keypoints_array)
    # declease_score = score_henchedBack(henchBacks)
    # 0未満なら0にする
    # score = (score - declease_score) if (score - declease_score) > 0 else 0
    
    print(f"傾きの平均: {angle_avg:.2f}°")
    print(f"得点: {score}")