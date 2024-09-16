import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import filedialog
from tools.color_edge import h36m_color_edge  # 色情報を取得

# 関節をつなぐ情報（vis_h36m.py から取得）
connections = [
    (0, 1), (1, 2), (2, 3),  # 右腕
    (0, 4), (4, 5), (5, 6),  # 左腕
    (0, 7), (7, 8), (8, 9), (9, 10),  # 胴体
    (8, 11), (11, 12), (12, 13),  # 右脚
    (8, 14), (14, 15), (15, 16)  # 左脚
]
# グローバル変数
skeleton_data = None

def load_skeleton_data(npz_file):
    data = np.load(npz_file)
    return data['reconstruction']

def visualize_skeleton(data, on_complete):
    print("Shape of skeleton data:", skeleton_data.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # ここで軸の範囲を固定する
    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([0, 2])

    for frame in skeleton_data[0]:
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 軸の範囲を固定する（再描画時に範囲がリセットされないように）
        ax.set_xlim3d([-1, 1])
        ax.set_ylim3d([-1, 1])
        ax.set_zlim3d([0, 2])

        x_data = frame[:, 0]
        y_data = frame[:, 1]
        z_data = frame[:, 2]
        ax.scatter(x_data, y_data, z_data, color='b')

        for (start, end) in connections:
            if start < frame.shape[0] and end < frame.shape[0]:
                print(f"Frame Start: {frame[start]}, Frame End: {frame[end]}")
                print(f"X Coordinates: {frame[start, 0], frame[end, 0]}")
                print(f"Y Coordinates: {frame[start, 1], frame[end, 1]}")
                print(f"Z Coordinates: {frame[start, 2], frame[end, 2]}")
                
                color = h36m_color_edge(start)
                ax.plot([frame[start, 0], frame[end, 0]], 
                        [frame[start, 1], frame[end, 1]], 
                        [frame[start, 2], frame[end, 2]], color=color)
            else:
                print(f"Invalid index: start={start}, end={end}, frame.shape={frame.shape}")

        plt.pause(0.1)
        # 再生が終わったらコールバックを呼び出す
        root.after(2000, on_complete)  # 2秒後に再生終了と仮定
    plt.show()

def open_file():
    global skeleton_data
    file_path = filedialog.askopenfilename(filetypes=[("NPZ files", "*.npz")])
    if file_path:
        skeleton_data = load_skeleton_data(file_path)
        visualize_skeleton(skeleton_data, on_playback_complete)
        
def reset_playback():
    if skeleton_data is not None and skeleton_data.size > 0:
        visualize_skeleton(skeleton_data, on_playback_complete)

def on_playback_complete():
    reset_button.pack(pady=10)

# tkinterを使ったシンプルなGUIの作成
root = tk.Tk()
root.title("3D Skeleton Viewer")
root.geometry("300x100")

open_button = tk.Button(root, text="Open NPZ File", command=open_file)
open_button.pack(pady=20)

reset_button = tk.Button(root, text="Reset Playback", command=reset_playback)
reset_button.pack_forget()  # 初期状態では非表示でOpen NPZ FIleボタンの下に出現

root.mainloop()
