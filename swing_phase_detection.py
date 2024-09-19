import json
import numpy as np
import sys
import matplotlib.pyplot as plt
from joint_mappings import keypoint_indices  # Import joint mappings from separate file

def load_swing_data(json_file):
    """
    JSONファイルからスイングデータをロードする
    """
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data['frames']

def get_joint_coordinates(frames, joint_name, keypoints_mapping):
    """
    指定された関節名の座標を各フレームから抽出する
    """
    joint_index = keypoints_mapping[joint_name]
    coordinates = []
    for frame in frames:
        joint = next((j for j in frame['joints'] if j['joint_index'] == joint_index), None)
        if joint:
            coordinates.append([
                joint['coordinates']['x'],
                joint['coordinates']['y'],
                joint['coordinates']['z']
            ])
        else:
            coordinates.append([np.nan, np.nan, np.nan])
    return np.array(coordinates)

def find_crossing_points(z_coords, z_mid):
    """
    Z座標が中点を上昇して超えるポイントと、減少して超えるポイントを検出する
    """
    crossing_points = []
    for i in range(1, len(z_coords)):
        if z_coords[i-1] < z_mid and z_coords[i] >= z_mid:  # 上昇して中点を超える
            crossing_points.append((i, 'up'))
        elif z_coords[i-1] > z_mid and z_coords[i] <= z_mid:  # 減少して中点を超える
            crossing_points.append((i, 'down'))
    return crossing_points

def find_top_impact(crossing_points, z_coords):
    """
    中点を上昇・減少で超えるポイント間でトップとインパクトを検出する
    """
    top_frame = None
    impact_frame = None

    for i in range(1, len(crossing_points) - 1):
        if crossing_points[i-1][1] == 'up' and crossing_points[i][1] == 'down':
            start, end = crossing_points[i-1][0], crossing_points[i][0]
            top_frame = np.argmax(z_coords[start:end]) + start

        if crossing_points[i][1] == 'down' and crossing_points[i+1][1] == 'up':
            start, end = crossing_points[i][0], crossing_points[i+1][0]
            impact_frame = np.argmin(z_coords[start:end]) + start

    return top_frame, impact_frame

def find_finish(z_coords, impact_frame):
    """
    インパクト後でZ座標が最大となるフレームをフィニッシュとする
    """
    finish_frame = np.argmax(z_coords[impact_frame:]) + impact_frame
    return finish_frame

def detect_swing_phases(frames, keypoints_mapping):
    """
    スイングの各フェーズのframe_indexを検出する
    """
    # 左手の座標を取得（left_wrist）
    left_wrist_coords = get_joint_coordinates(frames, 'left_wrist', keypoints_mapping)
    z_coords = left_wrist_coords[:, 2]

    # Z座標の中点を計算
    z_mid = (min(z_coords) + max(z_coords)) / 2

    # 中点を超える上昇ポイントと減少ポイントを検出
    crossing_points = find_crossing_points(z_coords, z_mid)

    if len(crossing_points) < 3:
        print("十分な中点交差ポイントが見つかりませんでした")
        return None

    # トップとインパクトを検出
    top_frame, impact_frame = find_top_impact(crossing_points, z_coords)

    # フィニッシュはインパクト後の最大のZ座標を持つフレームとする
    finish_frame = find_finish(z_coords, impact_frame)

    if top_frame is None or impact_frame is None or finish_frame is None:
        print("トップ、インパクト、フィニッシュの検出に失敗しました")
        return None

    return {
        'address': frames[0]['frame_index'],  # アドレスはフレームの最初とする
        'top_of_swing': frames[top_frame]['frame_index'],
        'impact': frames[impact_frame]['frame_index'],
        'finish': frames[finish_frame]['frame_index']
    }

def visualize_phases(frames, z_coords, phases):
    """
    フェーズ検出結果を可視化する
    """
    frame_indices = np.array([frame['frame_index'] for frame in frames])

    plt.figure(figsize=(12, 6))
    plt.plot(frame_indices, z_coords, label='Left Wrist Z Coordinate')

    # フェーズポイントのプロット
    if phases:
        plt.scatter(phases['address'], z_coords[phases['address']], color='blue', label='Address', zorder=5)
        plt.scatter(phases['top_of_swing'], z_coords[phases['top_of_swing']], color='orange', label='Top of Swing', zorder=5)
        plt.scatter(phases['impact'], z_coords[phases['impact']], color='green', label='Impact', zorder=5)
        plt.scatter(phases['finish'], z_coords[phases['finish']], color='red', label='Finish', zorder=5)

    plt.xlabel('Frame Index')
    plt.ylabel('Left Wrist Z Coordinate')
    plt.title('Swing Phases Detection')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python detect_swing_phases.py <json_file>")
        sys.exit(1)

    # JSONデータのファイル名
    json_file = sys.argv[1]

    # スイングデータをロード
    frames = load_swing_data(json_file)

    # スイングフェーズを検出
    phases = detect_swing_phases(frames, keypoint_indices)

    if phases:
        # 結果を表示
        print("Swing Phases Frame Indices:")
        for phase, frame_index in phases.items():
            print(f"{phase.capitalize()}: Frame {frame_index}")

        # 左手のZ座標を取得
        left_wrist_coords = get_joint_coordinates(frames, 'left_wrist', keypoint_indices)
        z_coords = left_wrist_coords[:, 2]

        # フェーズを可視化
        visualize_phases(frames, z_coords, phases)
    else:
        print("フェーズの検出に失敗しました。")

if __name__ == "__main__":
    main()
