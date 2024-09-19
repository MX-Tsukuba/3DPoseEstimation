import json
import numpy as np
import matplotlib.pyplot as plt

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

def calculate_sway_score(frames, keypoints_mapping):
    """
    スウェーのスコアを計算する
    """
    # 骨盤の座標を取得（例: 'pelvis'）
    pelvis_coords = get_joint_coordinates(frames, 'pelvis', keypoints_mapping)
    pelvis_x_coords = pelvis_coords[:, 0]  # X座標（左右方向）

    # データの単位がメートルの場合、センチメートルに変換
    pelvis_x_coords_cm = pelvis_x_coords * 100

    # 基準位置（アドレス時の位置）
    address_pelvis_x = pelvis_x_coords_cm[0]

    # 横方向偏差の計算
    deviations = pelvis_x_coords_cm - address_pelvis_x

    # 最大偏差の計算
    max_deviation = np.nanmax(np.abs(deviations))

    # スコアの計算
    allowable_sway = 5.0  # 許容スウェー量（cm）
    max_sway = 20.0       # 最大スウェー量（この値以上は0点）
    if max_deviation <= allowable_sway:
        score = 100
    elif max_deviation >= max_sway:
        score = 0
    else:
        penalty_factor = 100 / (max_sway - allowable_sway)
        score = max(0, 100 - ((max_deviation - allowable_sway) * penalty_factor))

    return score, deviations

def visualize_sway(frames, deviations):
    """
    スウェーの偏差を可視化する
    """
    frame_indices = np.array([frame['frame_index'] for frame in frames])

    plt.figure(figsize=(12, 6))
    plt.plot(frame_indices, deviations, label='Pelvis X Deviation (cm)')
    plt.xlabel('Frame Index')
    plt.ylabel('Deviation from Address Position (cm)')
    plt.title('Sway Analysis')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # JSONデータのファイル名
    json_file = 'sample_output.json' 

    # 関節名とjoint_indexのマッピング
    keypoint_indices = {
        "pelvis": 0,
    }

    # スイングデータをロード
    frames = load_swing_data(json_file)

    # スウェースコアを計算
    score, deviations = calculate_sway_score(frames, keypoint_indices)

    # 結果を表示
    print(f"Sway Score: {score:.2f}/100")

    # スウェーを可視化
    visualize_sway(frames, deviations)

if __name__ == "__main__":
    main()
