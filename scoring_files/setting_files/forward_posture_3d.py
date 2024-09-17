import json
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from scoring_files.setting_files.json2numpy_3d import json2numpy
from scoring_files.setting_files.angle_between_points_3d import angle_between_points


def posture_angle(keypoints,frame):
    """特定のフレームを与えるとその時点での姿勢の角度を返す"""
    # フレームごとのすべてのキーポイント座標を表示するfor文
    angle = angle_between_points(keypoints[frame][0], keypoints[frame][9])
    return angle

def posture_angles(keypoints):
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
    
def compare_posture(keypoints,frame1,frame2):
    angles = posture_angles(keypoints)
    return abs(angles[frame1] - angles[frame2])
