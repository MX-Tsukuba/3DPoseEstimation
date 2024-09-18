import json
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from setting_files.json2numpy_3d import json2numpy
from setting_files.angle_between_points_3d import angle_between_points
from setting_files.angle_between_3points_3d import angle_between_3points


def posture_angle(keypoints,frame):
    """
    特定のフレームの姿勢の角度を計算する関数
    パラメータ:
        keypoints (array-like): フレームごとのキーポイントの配列
        frame (int): 何番目のフレームか

    返り値:
        float: 特定のフレームの姿勢の角度
    """
    # フレームごとのすべてのキーポイント座標を表示するfor文
    angle = angle_between_points(keypoints[frame][0], keypoints[frame][9])
    return angle

def posture_angles(keypoints):
    """
    姿勢の角度のリストを返す関数

    パラメータ:
        keypoints (array_like): フレームごとのキーポイントの配列

    戻り値:
        list: フレームごとの姿勢の角度のリスト
    """
    results = []
    
    for i in range(len(keypoints)):
        angle = posture_angle(keypoints,i)
        results.append(angle)
        
    return results

def plot_body_tilt(tilt_angles):
    """
    全フレームの体の傾きをグラフ化する。
    
    パラメータ:
    tilt_angles (list): 全フレームの体の傾きのリスト
    """
    plt.figure(figsize=(10, 5))
    plt.plot(tilt_angles, marker='o')
    plt.title("Body Tilt Over Frames")
    plt.xlabel("Frame")
    plt.ylabel("Tilt Angle (degrees)")
    plt.grid(True)
    plt.show()
    
def compare_posture(keypoints_array,frame1,frame2):
    """
    パラメータ:
        keypoints (array-like): フレームごとのキーポイントの配列
        frame1 (int): フレーム1
        frame2 (int): フレーム2

    戻り値:
        _type_: 二つのフレームの姿勢の角度の差
    """
    
    angles = posture_angles(keypoints_array)
    return abs(angles[frame1] - angles[frame2])

def avg_posture_angle(keypoints):
    """
    パラメータ:
        keypoints (array-like): フレームごとのキーポイントの配列

    戻り値:
        _type_: 全フレームの姿勢の角度の平均
    """
    
    angles = posture_angles(keypoints)
    return np.mean(angles)

def henchedBack(keypoints,frame):
    """
    パラメータ:
        keypoints (array-like): フレームごとのキーポイントの配列
        frame (int): 何番目のフレームか

    戻り値:
        _type_: 肩が丸まりすぎていないかどうかを判定する関数
    """
    angle = angle_between_3points(keypoints[frame][0], keypoints[frame][9],keypoints[frame][8])
    return angle

def henchedBacks(keypoints):
    """
    首・背中・腰の三点がなす角のリストを返す関数

    パラメータ:
        keypoints (array_like): フレームごとのキーポイントの配列

    戻り値:
        list: フレームごとの首・背中・腰の三点がなす角の角度のリスト
    """
    results = []
    
    for i in range(len(keypoints)):
        angle = henchedBack(keypoints,i)
        results.append(angle)
        
    return results

def avg_henchedBacks(keypoints):
    """
    パラメータ:
        keypoints (array-like): フレームごとのキーポイントの配列

    戻り値:
        _type_: 全フレームの首・背中・腰の三点がなす角の角度の平均
    """
    
    angles = henchedBacks(keypoints)
    return np.mean(angles)