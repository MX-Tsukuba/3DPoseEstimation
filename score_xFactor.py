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
        raise ValueError(f"Joint '{joint_name}' がフレーム {frame_number} に見つかりません。")
    
    return np.array([joint['coordinates']['x'], joint['coordinates']['y'], joint['coordinates']['z']])

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("ゼロベクトルの正規化はできません。")
    return vector / norm

def calculate_rotation_angle(vector_ref, vector_target, axis):
    v1_proj = vector_ref - np.dot(vector_ref, axis) * axis
    v2_proj = vector_target - np.dot(vector_target, axis) * axis

    try:
        v1_norm = normalize_vector(v1_proj)
        v2_norm = normalize_vector(v2_proj)
    except ValueError:
        return 0.0

    orthonormal1 = v1_norm
    orthonormal2 = np.cross(axis, orthonormal1)

    x = np.dot(v2_norm, orthonormal1)
    y = np.dot(v2_norm, orthonormal2)

    angle_rad = np.arctan2(y, x)
    angle_deg = np.degrees(angle_rad)

    if angle_deg < 0:
        angle_deg += 360

    return angle_deg

def calculate_x_factor(shoulder_rotation, waist_rotation):
    """
    Xファクター (肩と腰の回転角度の差) を計算
    """
    return abs(shoulder_rotation - waist_rotation)

def evaluate_x_factor(x_factor):
    """
    Xファクターの評価スコアを計算
    理想の範囲は40〜50度とする
    """
    ideal_x_factor_min = 40
    ideal_x_factor_max = 50

    if ideal_x_factor_min <= x_factor <= ideal_x_factor_max:
        score = 100  # 理想的な範囲
    elif x_factor < ideal_x_factor_min:
        # 理想より小さい場合はスコアを低くする
        score = max(0, 100 - (ideal_x_factor_min - x_factor) * 2)
    else:
        # 理想より大きい場合もスコアを低くする
        score = max(0, 100 - (x_factor - ideal_x_factor_max) * 2)

    return score

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <json_file> <address_frame> <top_frame>")
        sys.exit(1)

    json_file = sys.argv[1]
    try:
        address_frame = int(sys.argv[2])
        top_frame = int(sys.argv[3])
    except ValueError:
        print("Frame numbers must be integers.")
        sys.exit(1)
    
    try:
        frames = load_swing_data(json_file)
    except FileNotFoundError:
        print(f"File '{json_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file '{json_file}'.")
        sys.exit(1)
    
    joints_needed = ["center_spine", "center_hip", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    
    try:
        address_positions = {joint: get_joint_position(frames, address_frame, joint, keypoint_indices) for joint in joints_needed}
        top_positions = {joint: get_joint_position(frames, top_frame, joint, keypoint_indices) for joint in joints_needed}
    except ValueError as e:
        print(e)
        sys.exit(1)
    
    address_spine_axis = address_positions["center_spine"] - address_positions["center_hip"]
    
    try:
        spine_axis = normalize_vector(address_spine_axis)
    except ValueError:
        print(f"Frame {address_frame} spine axis is a zero vector.")
        sys.exit(1)
    
    if spine_axis[2] < 0:
        spine_axis = -spine_axis
    
    address_waist_vector = address_positions["right_hip"] - address_positions["left_hip"]
    top_waist_vector = top_positions["right_hip"] - top_positions["left_hip"]
    
    address_shoulder_vector = address_positions["right_shoulder"] - address_positions["left_shoulder"]
    top_shoulder_vector = top_positions["right_shoulder"] - top_positions["left_shoulder"]
    
    waist_rotation_top = calculate_rotation_angle(top_waist_vector, address_waist_vector, spine_axis)
    shoulder_rotation_top = calculate_rotation_angle(top_shoulder_vector, address_shoulder_vector, spine_axis)
    
    waist_rotation_top = waist_rotation_top % 360
    shoulder_rotation_top = shoulder_rotation_top % 360
    
    # Xファクターを計算
    x_factor = calculate_x_factor(shoulder_rotation_top, waist_rotation_top)
    x_factor_score = evaluate_x_factor(x_factor)

    print(f"\nFrame {address_frame} to {top_frame} rotation angles:")
    print(f"Waist rotation at top: {waist_rotation_top:.2f} degrees")
    print(f"Shoulder rotation at top: {shoulder_rotation_top:.2f} degrees")
    print(f"X Factor: {x_factor:.2f} degrees")
    print(f"X Factor Score: {x_factor_score}/100")

if __name__ == "__main__":
    main()
