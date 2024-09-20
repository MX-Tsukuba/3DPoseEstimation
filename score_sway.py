import json
import numpy as np
import sys
from joint_mappings import keypoint_indices

def load_swing_data(json_file):
    """
    JSONファイルからスイングデータをロードする
    """
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data['frames']

def get_joint_position(frames, frame_number, joint_name, keypoints_mapping):
    """
    指定されたフレーム番号と関節名に対応する3D座標を取得する
    """
    frame = next((f for f in frames if f['frame_index'] == frame_number), None)
    if frame is None:
        raise ValueError(f"Frame {frame_number} が見つかりません。")
    
    joint_index = keypoints_mapping[joint_name]
    joint = next((j for j in frame['joints'] if j['joint_index'] == joint_index), None)
    if joint is None:
        raise ValueError(f"Joint '{joint_name}' (index {joint_index}) がフレーム {frame_number} に見つかりません。")
    
    if 'coordinates' not in joint or any(coord not in joint['coordinates'] for coord in ['x', 'y', 'z']):
        raise ValueError(f"Joint '{joint_name}' の座標データが不完全です。")
    
    position = np.array([
        joint['coordinates']['x'],
        joint['coordinates']['y'],
        joint['coordinates']['z']
    ])
    return position

def calculate_rotation_matrix(axis_unit, y_unit):
    """
    基準軸に基づく回転行列を計算する
    """
    # Z軸は基準軸とY軸の外積で計算
    z_unit = np.cross(axis_unit, y_unit)
    z_norm = np.linalg.norm(z_unit)
    if z_norm == 0:
        raise ValueError("基準軸とY軸が平行です。スウェー方向の計算に失敗しました。")
    z_unit /= z_norm

    # 再度X軸を基準軸に設定（正規化済み）
    x_unit = axis_unit

    # 再計算したZ軸とY軸から再度Y軸を調整
    y_unit_corrected = np.cross(z_unit, x_unit)
    
    rotation_matrix = np.vstack([x_unit, y_unit_corrected, z_unit]).T  # 3x3行列
    return rotation_matrix

def rotate_and_translate(coord, translation, rotation_matrix):
    """
    座標を平行移動および回転させる
    """
    translated_coord = coord - translation
    local_coord = rotation_matrix.T @ translated_coord
    return local_coord

def calculate_sway_score(frames, keypoints_mapping, phase_frames):
    """
    スウェーのスコアを計算する
    """
    # アドレス時の足首の位置を取得
    left_ankle_address = get_joint_position(frames, phase_frames['address'], 'left_ankle', keypoints_mapping)
    right_ankle_address = get_joint_position(frames, phase_frames['address'], 'right_ankle', keypoints_mapping)
    
    print(f"Address フェーズの left_ankle 座標: {left_ankle_address}")
    print(f"Address フェーズの right_ankle 座標: {right_ankle_address}")
    
    # 足首の中点を基準点とする
    ankle_center_address = (left_ankle_address + right_ankle_address) / 2
    print(f"アドレス時の足首の中点（ankle_center_address）: {ankle_center_address}")
    
    # 左足首から右足首へのベクトルを基準軸とする
    ankle_axis = right_ankle_address - left_ankle_address
    axis_norm = np.linalg.norm(ankle_axis)
    if axis_norm == 0:
        raise ValueError("左足首と右足首の位置が同一です。")
    axis_unit = ankle_axis / axis_norm  # 正規化した基準軸
    print(f"基準軸（左足首から右足首へのベクトル）: {ankle_axis}")
    print(f"正規化された基準軸（axis_unit）: {axis_unit}")
    
    # Y軸を前後方向と仮定
    y_unit = np.array([0, 1, 0])
    
    # 回転行列の計算
    rotation_matrix = calculate_rotation_matrix(axis_unit, y_unit)
    print(f"回転行列（rotation_matrix）:\n{rotation_matrix}")
    
    deviations = {}
    max_deviation = 0
    allowable_sway = 5.0  # 許容スウェー量（cm）
    max_sway = 20.0        # 最大スウェー量（cm）
    
    for phase_name in ['top', 'impact', 'finish']:
        # 各フェーズでのcenter_hipの位置を取得
        center_hip_phase = get_joint_position(frames, phase_frames[phase_name], 'center_hip', keypoints_mapping)
        print(f"{phase_name.capitalize()} フェーズの center_hip 座標: {center_hip_phase}")
        
        # ローカル座標系に変換
        center_hip_local = rotate_and_translate(center_hip_phase, ankle_center_address, rotation_matrix)
        print(f"{phase_name.capitalize()} フェーズの center_hip ローカル座標系での位置: {center_hip_local}")
        
        # 左右移動量（X軸方向）
        lateral_movement_cm = center_hip_local[0] * 100  # メートルからセンチメートルに変換
        deviations[phase_name] = lateral_movement_cm
        max_deviation = max(max_deviation, abs(lateral_movement_cm))
        print(f"{phase_name.capitalize()} フェーズの左右移動量: {lateral_movement_cm:.2f} cm")
    
    # スコアの計算
    if max_deviation <= allowable_sway:
        score = 100
    elif max_deviation >= max_sway:
        score = 0
    else:
        penalty_factor = 100 / (max_sway - allowable_sway)
        score = max(0, 100 - ((max_deviation - allowable_sway) * penalty_factor))
    
    return score, deviations

def main():
    if len(sys.argv) != 6:
        print("Usage: python sway_score.py <json_file> <address_frame> <top_frame> <impact_frame> <finish_frame>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    try:
        address_frame = int(sys.argv[2])
        top_frame = int(sys.argv[3])
        impact_frame = int(sys.argv[4])
        finish_frame = int(sys.argv[5])
    except ValueError:
        print("フレーム番号は整数で指定してください。")
        sys.exit(1)
    
    try:
        frames = load_swing_data(json_file)
    except FileNotFoundError:
        print(f"ファイル '{json_file}' が見つかりません。")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"ファイル '{json_file}' のJSONデコードエラー。")
        sys.exit(1)
    
    phase_frames = {
        'address': address_frame,
        'top': top_frame,
        'impact': impact_frame,
        'finish': finish_frame,
    }
    
    try:
        score, deviations = calculate_sway_score(frames, keypoint_indices, phase_frames)
    except ValueError as e:
        print(e)
        sys.exit(1)
    
    print(f"\nスウェースコア: {score:.2f}/100")
    print("各フェーズでの骨盤の左右移動量（cm）:")
    for phase in ['top', 'impact', 'finish']:
        deviation_cm = deviations[phase]
        print(f"{phase.capitalize()}: {deviation_cm:.2f} cm")

if __name__ == "__main__":
    main()
