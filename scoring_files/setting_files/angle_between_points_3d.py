import math
import numpy as np

def angle_between_points(point1, point2):
    """
    2点間を結ぶ直線の角度(z軸に対する)を計算する関数

    パラメータ:
    point1 (array-like): 最初の点の座標 [x, y, z]
    point2 (array-like): 2番目の点の座標 [x, y, z]

    戻り値:
    float: 2点間を結ぶ直線に対する角度（度単位）。入力点のいずれかにNaNが含まれている場合はNaNを返す。
    """
    if np.isnan(point1).any() or np.isnan(point2).any():
        return np.nan  # データが不完全な場合はNaNを返す
    
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    
    # 2点間のベクトルを計算
    v = np.array([x2 - x1, y2 - y1, z2 - z1])
    # z平面と並行なベクトル
    z = np.array([0, 0, 1])
    
    # 内積を計算
    dot = np.dot(v, z)
    
    # ベクトルの長さを計算
    norm_v = np.linalg.norm(v)
    norm_z = np.linalg.norm(z)
    
    # 角度を計算
    t_cos = dot / (norm_v * norm_z)
    
    #arccosでラジアン角度を求める
    t_rad = np.arccos(np.clip(t_cos, -1.0, 1.0))
    
    # ラジアン角度を度に変換
    t_deg = math.degrees(t_rad)
    
    return t_deg