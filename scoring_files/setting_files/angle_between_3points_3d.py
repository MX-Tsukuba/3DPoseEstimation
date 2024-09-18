import math
import numpy as np

def angle_between_3points(point1, point2,middle_point):
    """
    middle_pointを頂点とする3点間を結ぶ角の角度を計算する関数

    パラメータ:
    point1 (array-like): 最初の点の座標 [x, y, z]
    point2 (array-like): 2番目の点の座標 [x, y, z]
    middle_point (array-like): 頂点の座標 [x, y, z]

    戻り値:
    float: middle_pointを頂点とする三点を結んでできる角の角度（度単位）。入力点のいずれかにNaNが含まれている場合はNaNを返す。
    """
    if np.isnan(point1).any() or np.isnan(point2).any() or np.isnan(middle_point).any():
        return np.nan  # データが不完全な場合はNaNを返す
    
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    x3, y3, z3 = middle_point
    
    # 2つのベクトルを計算
    v1 = np.array([x1 - x3, y1 - y3, z1 - z3])
    v2 = np.array([x2 - x3, y2 - y3, z2 - z3])
    
    # 内積を計算
    dot = np.dot(v1, v2)
    
    # ベクトルの長さを計算
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(z2)
    
    # 角度を計算
    t_cos = dot / (norm_v1 * norm_v2)
    
    #arccosでラジアン角度を求める
    t_rad = np.arccos(np.clip(t_cos, -1.0, 1.0))
    
    # ラジアン角度を度に変換
    t_deg = math.degrees(t_rad)
    
    return t_deg