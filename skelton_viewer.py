import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import filedialog, ttk
from tools.color_edge import h36m_color_edge  # 色情報を取得
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 関節をつなぐ情報
connections = [
    (0, 1), (1, 2), (2, 3),  # 右腕
    (0, 4), (4, 5), (5, 6),  # 左腕
    (0, 7), (7, 8), (8, 9), (9, 10),  # 胴体
    (8, 11), (11, 12), (12, 13),  # 右脚
    (8, 14), (14, 15), (15, 16)  # 左脚
]

# グローバル変数
skeleton_data = None
is_playing = False  # 再生中かどうかを示すフラグ
current_frame = 0  # 現在のフレーム
total_frames = 0  # フレームの総数
progress_bar_updating = False  # プログレスバーを更新中かどうかのフラグ

def load_skeleton_data(npz_file):
    data = np.load(npz_file)
    return data['reconstruction']

def visualize_skeleton(frame):
    global skeleton_data
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([0, 2])

    x_data = frame[:, 0]
    y_data = frame[:, 1]
    z_data = frame[:, 2]
    ax.scatter(x_data, y_data, z_data, color='b')

    for (start, end) in connections:
        if start < frame.shape[0] and end < frame.shape[0]:
            color = h36m_color_edge(start)
            ax.plot([frame[start, 0], frame[end, 0]], 
                    [frame[start, 1], frame[end, 1]], 
                    [frame[start, 2], frame[end, 2]], color=color)

    canvas.draw()

def play_skeleton():
    global is_playing, current_frame, total_frames, progress_bar_updating
    if is_playing and current_frame < total_frames:
        print(f"Playing frame {current_frame}/{total_frames}")  # デバッグ用出力
        visualize_skeleton(skeleton_data[0][current_frame])
        current_frame += 1

        # プログレスバー更新フラグを設定
        progress_bar_updating = True
        progress_bar.set(current_frame)
        progress_bar_updating = False

        if current_frame < total_frames:
            root.after(100, play_skeleton)  # 100ミリ秒後に次のフレームを描画
        else:
            print("Playback finished.")
            is_playing = False  # 再生が終了したのでフラグをFalseにする
    else:
        print("Playback stopped or reached the end of frames.")

def start_playback():
    global is_playing, current_frame  # グローバル変数として宣言
    if total_frames == 0:
        print("No data loaded or no frames to play.")
        return
    if current_frame >= total_frames:
        current_frame = 0  # 全フレーム再生後にリセット
    is_playing = True
    print("Playback started.")
    play_skeleton()  # 再生を開始

def stop_playback():
    global is_playing
    is_playing = False
    print("Playback stopped.")

def reset_playback():
    global current_frame
    stop_playback()
    current_frame = 0
    progress_bar.set(0)
    visualize_skeleton(skeleton_data[0][current_frame])

def on_progress_change(value):
    global current_frame, is_playing, progress_bar_updating
    # プログラムからの更新中は処理をスキップ
    if progress_bar_updating:
        return
    stop_playback()  # 再生中にスライダーを動かすと一旦再生を止める
    current_frame = int(float(value))
    print(f"Jumping to frame {current_frame}") 
    visualize_skeleton(skeleton_data[0][current_frame])

def open_file():
    global skeleton_data, total_frames, current_frame
    file_path = filedialog.askopenfilename(filetypes=[("NPZ files", "*.npz")])
    if file_path:
        skeleton_data = load_skeleton_data(file_path)
        total_frames = len(skeleton_data[0])
        current_frame = 0
        progress_bar.config(to=total_frames - 1)
        print(f"Loaded {total_frames} frames.")  # デバッグ用出力
        visualize_skeleton(skeleton_data[0][current_frame])

# tkinterを使ったシンプルなGUIの作成
root = tk.Tk()
root.title("3D Skeleton Viewer")
root.geometry("400x300")

open_button = tk.Button(root, text="Open NPZ File", command=open_file)
open_button.pack(pady=5)

progress_bar = ttk.Scale(root, from_=0, to=100, orient="horizontal", length=300, command=on_progress_change)
progress_bar.pack(pady=5)

play_button = tk.Button(root, text="Play", command=start_playback)
play_button.pack(pady=5)

stop_button = tk.Button(root, text="Stop", command=stop_playback)
stop_button.pack(pady=5)

reset_button = tk.Button(root, text="Reset", command=reset_playback)
reset_button.pack(pady=5)

# matplotlib Figureをtkinterに埋め込むための設定
fig = plt.Figure()
ax = fig.add_subplot(111, projection='3d')
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

root.mainloop()
