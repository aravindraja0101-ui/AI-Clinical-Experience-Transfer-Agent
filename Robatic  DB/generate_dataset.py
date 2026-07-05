import pandas as pd
import numpy as np
import datetime
import os

# Set seed for reproducibility
np.random.seed(42)

# Goal settings
total_rows = 186532
novice_target = 65286
intermediate_target = 65286
expert_target = 55960

assert novice_target + intermediate_target + expert_target == total_rows

# Helper function to generate trial sizes that sum to exact target totals
def generate_trial_sizes(target_total, min_size=100, max_size=300):
    sizes = []
    current_sum = 0
    while current_sum < target_total:
        remaining = target_total - current_sum
        if remaining <= max_size:
            sizes.append(remaining)
            current_sum += remaining
        else:
            size = np.random.randint(min_size, max_size + 1)
            if remaining - size < min_size:
                size = remaining - min_size
            sizes.append(size)
            current_sum += size
    return sizes

novice_sizes = generate_trial_sizes(novice_target)
intermediate_sizes = generate_trial_sizes(intermediate_target)
expert_sizes = generate_trial_sizes(expert_target)

all_sizes = novice_sizes + intermediate_sizes + expert_sizes
num_trials = len(all_sizes)

skill_levels = (
    ['Novice'] * len(novice_sizes) +
    ['Intermediate'] * len(intermediate_sizes) +
    ['Expert'] * len(expert_sizes)
)

print(f"Generated {num_trials} trials (Novice: {len(novice_sizes)}, Intermediate: {len(intermediate_sizes)}, Expert: {len(expert_sizes)})")

# Pre-generate unique pools
proc_pool = [f"P_{i:05d}" for i in range(10001, 10301)]
pat_pool = [f"PT_{i:05d}" for i in range(20001, 20151)]
or_pool = ["OR_1", "OR_2", "OR_3", "OR_4"]

procedure_types = ["General Surgery", "Urology", "Gynecology", "Cardiothoracic", "Colorectal"]
task_names = ["Suturing", "Needle Passing", "Knot Tying", "Peg Transfer", "Dissection", "Cutting", "Camera Navigation"]

task_base_times = {
    "Suturing": 240.0,
    "Needle Passing": 180.0,
    "Knot Tying": 150.0,
    "Peg Transfer": 120.0,
    "Dissection": 300.0,
    "Cutting": 100.0,
    "Camera Navigation": 60.0
}

task_base_pickups = {
    "Suturing": 8.0,
    "Needle Passing": 10.0,
    "Knot Tying": 4.0,
    "Peg Transfer": 6.0,
    "Dissection": 0.0,
    "Cutting": 0.0,
    "Camera Navigation": 0.0
}

# Generate trial-level static features
trial_data = []
for i in range(num_trials):
    skill = skill_levels[i]
    trial_id = f"TR_{10000 + i:05d}"
    proc_id = np.random.choice(proc_pool)
    pat_id = np.random.choice(pat_pool)
    proc_type = np.random.choice(procedure_types)
    task_name = np.random.choice(task_names)
    
    session_num = np.random.randint(1, 11)
    trial_num = np.random.randint(1, 6)
    or_id = np.random.choice(or_pool)
    
    # Latent performance score conditional on skill
    if skill == 'Expert':
        p_trial = np.random.beta(8, 2)
    elif skill == 'Intermediate':
        p_trial = np.random.beta(5, 5)
    else:
        p_trial = np.random.beta(2, 8)
        
    # Frame Rate and dimensions
    fps = int(np.random.choice([25, 30, 50, 60]))
    width = int(np.random.choice([1920, 1280, 640], p=[0.6, 0.3, 0.1]))
    height = 1080 if width == 1920 else (720 if width == 1280 else 480)
    
    # Durations
    base_time = task_base_times[task_name]
    skill_factor = 0.65 if skill == 'Expert' else (1.05 if skill == 'Intermediate' else 1.8)
    comp_time = base_time * skill_factor * np.exp(np.random.normal(0, 0.12))
    comp_time = np.clip(comp_time, 20.0, 900.0)
    
    video_dur = comp_time + np.random.uniform(5.0, 20.0)
    video_dur = np.clip(video_dur, 30.0, 900.0)
    total_frames = int(np.round(video_dur * fps))
    
    # Trial outcome performance scores
    precision = 100.0 * (0.15 + 0.8 * p_trial + np.random.normal(0, 0.03))
    consistency = 100.0 * (0.2 + 0.75 * p_trial + np.random.normal(0, 0.03))
    dexterity = 100.0 * (0.15 + 0.8 * p_trial + np.random.normal(0, 0.03))
    smoothness = 100.0 * (0.2 + 0.75 * p_trial + np.random.normal(0, 0.03))
    motion_econ = 100.0 * (0.15 + 0.8 * p_trial + np.random.normal(0, 0.03))
    overall_perf = 100.0 * (0.1 + 0.85 * p_trial + np.random.normal(0, 0.02))
    
    precision = np.clip(precision, 0.0, 100.0)
    consistency = np.clip(consistency, 0.0, 100.0)
    dexterity = np.clip(dexterity, 0.0, 100.0)
    smoothness = np.clip(smoothness, 0.0, 100.0)
    motion_econ = np.clip(motion_econ, 0.0, 100.0)
    overall_perf = np.clip(overall_perf, 0.0, 100.0)
    
    # Error counts
    lambda_error = 22.0 * (1.0 - p_trial) + 0.5
    err_cnt = int(np.random.poisson(lambda_error))
    
    lambda_critical = 4.0 * (1.0 - p_trial) + 0.1
    crit_err_cnt = int(np.random.poisson(lambda_critical))
    crit_err_cnt = min(crit_err_cnt, err_cnt)
    
    # Collisions
    lambda_tissue_col = 15.0 * (1.0 - p_trial) + 1.0
    tissue_col_cnt = int(np.random.poisson(lambda_tissue_col))
    
    lambda_inst_col = 8.0 * (1.0 - p_trial) + 0.5
    inst_col_cnt = int(np.random.poisson(lambda_inst_col))
    
    # Camera adjustments
    lambda_cam_adj = 12.0 * (1.0 - p_trial) + 3.0
    cam_adj_cnt = int(np.random.poisson(lambda_cam_adj))
    
    # Pickups & Drops
    base_pick = task_base_pickups[task_name]
    pickups = int(base_pick + np.random.poisson(3.0 * (1.0 - p_trial))) if base_pick > 0 else 0
    drops = int(np.random.poisson(4.0 * (1.0 - p_trial))) if base_pick > 0 else 0
    
    # Suture & Knot
    sutures = int(np.random.randint(6, 13)) if task_name in ['Suturing', 'Needle Passing'] else 0
    knots = int(np.random.randint(3, 7)) if task_name == 'Knot Tying' else (int(np.random.randint(2, 5)) if task_name == 'Suturing' else 0)
    
    # Contact counts
    base_contacts = {"Suturing": 50, "Needle Passing": 40, "Knot Tying": 30, "Peg Transfer": 30, "Dissection": 80, "Cutting": 40, "Camera Navigation": 5}
    contacts = int(base_contacts[task_name] + np.random.poisson(15.0 * (1.0 - p_trial) + 5.0))
    
    # Recovery Time
    recovery_time = float(err_cnt * np.random.uniform(5.0, 15.0) * (2.0 - p_trial))
    recovery_time = np.clip(recovery_time, 0.0, comp_time)
    
    # Success
    task_success = int(np.random.uniform(0, 1) < (0.45 + 0.53 * p_trial))
    
    trial_data.append({
        'Trial_ID': trial_id, 'Procedure_ID': proc_id, 'Patient_ID': pat_id,
        'Procedure_Type': proc_type, 'Task_Name': task_name, 'Session_Number': session_num,
        'Trial_Number': trial_num, 'Skill_Level': skill, 'Operating_Room_ID': or_id,
        'Video_Duration': video_dur, 'Frame_Rate': fps, 'Total_Frames': total_frames,
        'Frame_Width': width, 'Frame_Height': height, 'p_trial': p_trial,
        'Completion_Time': comp_time, 'Error_Count': err_cnt, 'Critical_Error_Count': crit_err_cnt,
        'Tissue_Collision_Count': tissue_col_cnt, 'Instrument_Collision_Count': inst_col_cnt,
        'Camera_Adjustments': cam_adj_cnt, 'Needle_Pickup_Count': pickups, 'Needle_Drop_Count': drops,
        'Suture_Pass_Count': sutures, 'Knot_Count': knots, 'Tissue_Contact_Count': contacts,
        'Precision_Score': precision, 'Consistency_Score': consistency, 'Dexterity_Index': dexterity,
        'Smoothness_Index': smoothness, 'Economy_of_Motion': motion_econ, 'Overall_Performance': overall_perf,
        'Recovery_Time': recovery_time, 'Task_Success': task_success
    })

df_trial = pd.DataFrame(trial_data)

# Repeat trial row details by their specified window sizes
df_rows = df_trial.loc[df_trial.index.repeat(all_sizes)].reset_index(drop=True)

# Generate temporal sequence ID for each row
t = df_rows.groupby('Trial_ID').cumcount().values

# Parameters for vectorization
W = df_rows['Frame_Width'].values
H = df_rows['Frame_Height'].values
p = df_rows['p_trial'].values
skill = df_rows['Skill_Level'].values
N = len(df_rows)

print("Synthesizing row-level frame details...")

# Base date times for each trial
start_ts = int(datetime.datetime(2024, 1, 1).timestamp())
end_ts = int(datetime.datetime(2025, 12, 31).timestamp())
trial_base_ts = np.random.randint(start_ts, end_ts, size=num_trials)
row_base_ts = trial_base_ts[df_trial.index.repeat(all_sizes)]
row_ts = row_base_ts + t
recording_dates = pd.to_datetime(row_ts, unit='s').strftime('%Y-%m-%d %H:%M:%S')

# Tool coordinates and movement (vectorized)
Tool1_center_X = 0.35 * W
Tool1_center_Y = 0.5 * H
Tool2_center_X = 0.65 * W
Tool2_center_Y = 0.5 * H

amp_coeff = np.where(skill == 'Expert', 0.05, np.where(skill == 'Intermediate', 0.10, 0.18))
noise_std = np.where(skill == 'Expert', 1.5, np.where(skill == 'Intermediate', 6.0, 16.0))

Tool1_X = Tool1_center_X + amp_coeff * W * np.sin(t / 18.0) + np.random.normal(0, noise_std, size=N)
Tool1_Y = Tool1_center_Y + amp_coeff * H * np.cos(t / 18.0) + np.random.normal(0, noise_std, size=N)
Tool2_X = Tool2_center_X + amp_coeff * W * np.sin(t / 24.0) + np.random.normal(0, noise_std, size=N)
Tool2_Y = Tool2_center_Y + amp_coeff * H * np.cos(t / 24.0) + np.random.normal(0, noise_std, size=N)

Tool1_Angle = (180.0 + 35.0 * np.sin(t / 12.0) + np.random.normal(0, 8.0, size=N)) % 360.0
Tool2_Angle = (180.0 + 35.0 * np.cos(t / 15.0) + np.random.normal(0, 8.0, size=N)) % 360.0

# Speeds and accelerations
speed_mean = np.where(skill == 'Expert', 35.0, np.where(skill == 'Intermediate', 55.0, 85.0))
speed_std = np.where(skill == 'Expert', 8.0, np.where(skill == 'Intermediate', 18.0, 35.0))
Tool1_Speed = np.abs(speed_mean + speed_std * np.sin(t / 9.0) + np.random.normal(0, speed_std, size=N))
Tool2_Speed = np.abs(speed_mean + speed_std * np.cos(t / 9.0) + np.random.normal(0, speed_std, size=N))

accel_mean = np.where(skill == 'Expert', 18.0, np.where(skill == 'Intermediate', 32.0, 52.0))
accel_std = np.where(skill == 'Expert', 4.0, np.where(skill == 'Intermediate', 12.0, 22.0))
Tool1_Acceleration = np.abs(accel_mean + accel_std * np.sin(t / 6.0) + np.random.normal(0, accel_std, size=N))
Tool2_Acceleration = np.abs(accel_mean + accel_std * np.cos(t / 6.0) + np.random.normal(0, accel_std, size=N))

Tool_Distance = np.sqrt((Tool1_X - Tool2_X)**2 + (Tool1_Y - Tool2_Y)**2) * 0.15
Tool_Overlap = 100.0 / (1.0 + 0.025 * Tool_Distance) + np.random.normal(0, 1.5, size=N)
Tool_Visibility = 0.88 + 0.11 * p + np.random.normal(0, 0.04, size=N)

# Assemble DataFrame
df_final = pd.DataFrame()
df_final['Trial_ID'] = df_rows['Trial_ID']
df_final['Procedure_ID'] = df_rows['Procedure_ID']
df_final['Patient_ID'] = df_rows['Patient_ID']
df_final['Procedure_Type'] = df_rows['Procedure_Type']
df_final['Task_Name'] = df_rows['Task_Name']
df_final['Session_Number'] = df_rows['Session_Number']
df_final['Trial_Number'] = df_rows['Trial_Number']
df_final['Skill_Level'] = df_rows['Skill_Level']
df_final['Operating_Room_ID'] = df_rows['Operating_Room_ID']
df_final['Recording_Date'] = recording_dates
df_final['Video_Duration'] = np.round(df_rows['Video_Duration'], 2)
df_final['Frame_Rate'] = df_rows['Frame_Rate']
df_final['Total_Frames'] = df_rows['Total_Frames']
df_final['Frame_Width'] = df_rows['Frame_Width']
df_final['Frame_Height'] = df_rows['Frame_Height']

# Visual features
df_final['Brightness_Mean'] = np.round(np.clip(125.0 + 18.0 * np.sin(t / 10.0) + np.random.normal(0, 4.0, size=N), 0.0, 255.0), 2)
df_final['Contrast_Mean'] = np.round(np.clip(85.0 + 12.0 * np.cos(t / 12.0) + np.random.normal(0, 4.0, size=N), 0.0, 255.0), 2)
df_final['Blur_Index'] = np.round(np.clip(14.0 + 4.0 * np.sin(t / 16.0) + np.random.normal(0, 1.5, size=N), 0.0, 100.0), 4)
df_final['Noise_Level'] = np.round(np.clip(4.5 + 1.8 * np.cos(t / 8.0) + np.random.normal(0, 0.8, size=N), 0.0, 50.0), 4)
df_final['Compression_Ratio'] = np.round(np.clip(14.5 + 2.5 * np.sin(t / 120.0) + np.random.normal(0, 0.4, size=N), 1.0, 100.0), 4)

df_final['Tool1_X'] = np.round(np.clip(Tool1_X, 0.0, W), 2)
df_final['Tool1_Y'] = np.round(np.clip(Tool1_Y, 0.0, H), 2)
df_final['Tool2_X'] = np.round(np.clip(Tool2_X, 0.0, W), 2)
df_final['Tool2_Y'] = np.round(np.clip(Tool2_Y, 0.0, H), 2)
df_final['Tool1_Angle'] = np.round(Tool1_Angle, 2)
df_final['Tool2_Angle'] = np.round(Tool2_Angle, 2)
df_final['Tool1_Length'] = np.round(np.clip(240.0 + 45.0 * np.sin(t / 32.0) + np.random.normal(0, 4.0, size=N), 0.0, 1000.0), 2)
df_final['Tool2_Length'] = np.round(np.clip(240.0 + 45.0 * np.cos(t / 36.0) + np.random.normal(0, 4.0, size=N), 0.0, 1000.0), 2)
df_final['Tool1_Speed'] = np.round(np.clip(Tool1_Speed, 0.0, 300.0), 2)
df_final['Tool2_Speed'] = np.round(np.clip(Tool2_Speed, 0.0, 300.0), 2)
df_final['Tool1_Acceleration'] = np.round(np.clip(Tool1_Acceleration, 0.0, 150.0), 2)
df_final['Tool2_Acceleration'] = np.round(np.clip(Tool2_Acceleration, 0.0, 150.0), 2)
df_final['Tool_Distance'] = np.round(np.clip(Tool_Distance, 0.0, 3000.0), 2)
df_final['Tool_Overlap'] = np.round(np.clip(Tool_Overlap, 0.0, 100.0), 2)
df_final['Tool_Visibility'] = np.round(np.clip(Tool_Visibility, 0.0, 1.0), 4)

# Path Efficiency and Velocity profiles
df_final['Path_Length'] = np.round(np.clip(df_rows['Completion_Time'] * (14.0 + 18.0 * (1.0 - p) + np.random.normal(0, 1.5, size=N)), 0.0, 50000.0), 2)
df_final['Straight_Path'] = np.round(np.clip(df_final['Path_Length'] * (0.82 * p + 0.12) + np.random.normal(0, 4.0, size=N), 0.0, df_final['Path_Length']), 2)
df_final['Efficiency_Ratio'] = np.round(np.clip(df_final['Straight_Path'] / (df_final['Path_Length'] + 1e-5), 0.0, 1.0), 4)

df_final['Average_Velocity'] = np.round(np.clip((df_final['Tool1_Speed'] + df_final['Tool2_Speed']) / 2.0 + np.random.normal(0, 1.5, size=N), 0.0, 300.0), 2)
df_final['Maximum_Velocity'] = np.round(np.clip(df_final['Average_Velocity'] * (1.2 + 1.4 * (1.0 - p) + np.random.normal(0, 0.08, size=N)), df_final['Average_Velocity'], 300.0), 2)
df_final['Minimum_Velocity'] = np.round(np.clip(df_final['Average_Velocity'] * (0.12 * p + np.random.normal(0, 0.01, size=N)), 0.0, df_final['Average_Velocity']), 2)
df_final['Velocity_SD'] = np.round(np.clip(df_final['Average_Velocity'] * (0.16 + 0.32 * (1.0 - p) + np.random.normal(0, 0.02, size=N)), 0.0, 150.0), 2)

df_final['Motion_Smoothness'] = np.round(np.clip(0.12 + 0.83 * p + np.random.normal(0, 0.015, size=N), 0.0, 1.0), 4)
df_final['Motion_Jerk'] = np.round(np.clip(48.0 * (1.0 - p) + np.random.normal(0, 1.5, size=N), 0.0, 50.0), 4)

df_final['Pause_Count'] = np.clip(np.round(4.0 * (1.0 - p) + np.random.poisson(1.8 * (1.0 - p), size=N)), 0, 100).astype(int)
df_final['Pause_Time'] = np.round(np.clip(df_final['Pause_Count'] * np.random.uniform(1.4, 2.8, size=N), 0.0, df_rows['Video_Duration']), 2)
df_final['Idle_Time'] = np.round(np.clip(8.0 * (1.0 - p) + np.random.uniform(4.0, 12.0, size=N) * (1.0 - p), 0.0, 200.0), 2)
df_final['Motion_Frequency'] = np.round(np.clip(1.6 + 2.4 * p + np.random.normal(0, 0.25, size=N), 0.1, 10.0), 4)
df_final['Direction_Changes'] = np.clip(np.round(14 + 32 * (1.0 - p) + np.random.normal(0, 4.0, size=N)), 0, 500).astype(int)
df_final['Trajectory_Entropy'] = np.round(np.clip(1.6 + 1.8 * (1.0 - p) + np.random.normal(0, 0.08, size=N), 0.0, 10.0), 4)

# Camera Shift and stability
df_final['Camera_X_Shift'] = np.round(np.random.normal(0, 1.2, size=N), 2)
df_final['Camera_Y_Shift'] = np.round(np.random.normal(0, 1.2, size=N), 2)
df_final['Camera_Rotation'] = np.round(np.clip(np.random.uniform(-12.0, 12.0, size=N) * (1.0 - p), -180.0, 180.0), 2)
df_final['Camera_Zoom'] = np.round(np.clip(1.4 + 0.4 * np.sin(t / 120.0) + np.random.normal(0, 0.04, size=N), 1.0, 5.0), 2)
df_final['Camera_Pan'] = np.round(np.random.normal(0, 4.5, size=N) * (1.0 - p), 2)
df_final['Camera_Tilt'] = np.round(np.random.normal(0, 4.5, size=N) * (1.0 - p), 2)
df_final['Camera_Stability'] = np.round(np.clip(0.52 + 0.43 * p + np.random.normal(0, 0.015, size=N), 0.0, 1.0), 4)
df_final['Camera_Jitter'] = np.round(np.clip(1.0 - df_final['Camera_Stability'] + np.random.normal(0, 0.008, size=N), 0.0, 1.0), 4)
df_final['Focus_Quality'] = np.round(np.clip(76.0 + 19.0 * p + np.random.normal(0, 1.8, size=N), 0.0, 100.0), 2)
df_final['View_Coverage'] = np.round(np.clip(82.0 + 14.0 * p + np.random.normal(0, 1.8, size=N), 0.0, 100.0), 2)

# Optical Flow features
df_final['Flow_Mean'] = np.round(np.clip(4.8 + 9.5 * (1.0 - p) + np.random.normal(0, 0.8, size=N), 0.0, 100.0), 4)
df_final['Flow_Max'] = np.round(np.clip(28.0 + 55.0 * (1.0 - p) + np.random.normal(0, 4.0, size=N), df_final['Flow_Mean'], 500.0), 4)
df_final['Flow_Min'] = np.round(np.clip(0.08 + 0.45 * p + np.random.normal(0, 0.08, size=N), 0.0, df_final['Flow_Mean']), 4)
df_final['Flow_SD'] = np.round(np.clip(2.8 + 6.5 * (1.0 - p) + np.random.normal(0, 0.4, size=N), 0.0, 100.0), 4)
df_final['Flow_Direction'] = np.round(np.random.uniform(0.0, 360.0, size=N), 2)
df_final['Flow_Entropy'] = np.round(np.clip(1.8 + 3.8 * (1.0 - p) + np.random.normal(0, 0.15, size=N), 0.0, 10.0), 4)
df_final['Flow_Density'] = np.round(np.clip(0.08 + 0.38 * (1.0 - p) + np.random.normal(0, 0.02, size=N), 0.0, 1.0), 4)
df_final['Flow_Curl'] = np.round(np.random.normal(0, 1.8, size=N), 4)
df_final['Flow_Divergence'] = np.round(np.random.normal(0, 1.8, size=N), 4)
df_final['Frame_Difference'] = np.round(np.clip(9.5 + 28.0 * (1.0 - p) + np.random.normal(0, 2.5, size=N), 0.0, 255.0), 4)

df_final['Needle_Pickup_Count'] = df_rows['Needle_Pickup_Count']
df_final['Needle_Drop_Count'] = df_rows['Needle_Drop_Count']
df_final['Suture_Pass_Count'] = df_rows['Suture_Pass_Count']
df_final['Knot_Count'] = df_rows['Knot_Count']
df_final['Tissue_Contact_Count'] = df_rows['Tissue_Contact_Count']
df_final['Tissue_Collision_Count'] = df_rows['Tissue_Collision_Count']
df_final['Instrument_Collision_Count'] = df_rows['Instrument_Collision_Count']
df_final['Camera_Adjustments'] = df_rows['Camera_Adjustments']
df_final['Error_Count'] = df_rows['Error_Count']
df_final['Critical_Error_Count'] = df_rows['Critical_Error_Count']

df_final['Completion_Time'] = np.round(df_rows['Completion_Time'], 2)
df_final['Economy_of_Motion'] = np.round(np.clip(df_rows['Economy_of_Motion'] + np.random.normal(0, 0.04, size=N), 0.0, 100.0), 2)
df_final['Precision_Score'] = np.round(np.clip(df_rows['Precision_Score'] + np.random.normal(0, 0.04, size=N), 0.0, 100.0), 2)
df_final['Consistency_Score'] = np.round(np.clip(df_rows['Consistency_Score'] + np.random.normal(0, 0.04, size=N), 0.0, 100.0), 2)
df_final['Dexterity_Index'] = np.round(np.clip(df_rows['Dexterity_Index'] + np.random.normal(0, 0.04, size=N), 0.0, 100.0), 2)
df_final['Smoothness_Index'] = np.round(np.clip(df_rows['Smoothness_Index'] + np.random.normal(0, 0.04, size=N), 0.0, 100.0), 2)
df_final['Error_Rate'] = np.round(df_final['Error_Count'] / (df_final['Completion_Time'] / 60.0), 4)
df_final['Recovery_Time'] = np.round(np.clip(df_rows['Recovery_Time'], 0, df_final['Completion_Time']), 2)
df_final['Task_Success'] = df_rows['Task_Success']
df_final['Overall_Performance'] = np.round(np.clip(df_rows['Overall_Performance'] + np.random.normal(0, 0.04, size=N), 0.0, 100.0), 2)

# CNN and Transformer centers
C_N = np.array([-2.0, 1.5, -1.0, 3.0, -0.5])
C_I = np.array([0.0, 0.0, 0.5, -1.0, 2.0])
C_E = np.array([2.0, -1.5, 2.5, -2.0, -1.5])

T_N = np.array([1.0, -2.0, 1.5, -0.5, 3.0])
T_I = np.array([-1.0, 0.5, -0.5, 2.0, -1.0])
T_E = np.array([-3.0, 2.5, -2.0, -1.5, -2.5])

cnn_centers = np.where(
    skill[:, None] == 'Expert', C_E,
    np.where(skill[:, None] == 'Intermediate', C_I, C_N)
)
transformer_centers = np.where(
    skill[:, None] == 'Expert', T_E,
    np.where(skill[:, None] == 'Intermediate', T_I, T_N)
)

cnn_features = np.clip(cnn_centers + np.random.normal(0, 0.75, size=(N, 5)), -5.0, 5.0)
transformer_features = np.clip(transformer_centers + np.random.normal(0, 0.75, size=(N, 5)), -5.0, 5.0)

for j in range(5):
    df_final[f'CNN_Feature_{j+1}'] = np.round(cnn_features[:, j], 6)
for j in range(5):
    df_final[f'Transformer_Feature_{j+1}'] = np.round(transformer_features[:, j], 6)

# Verify Column names and count
expected_cols = [
    "Trial_ID", "Procedure_ID", "Patient_ID", "Procedure_Type", "Task_Name",
    "Session_Number", "Trial_Number", "Skill_Level", "Operating_Room_ID", "Recording_Date",
    "Video_Duration", "Frame_Rate", "Total_Frames", "Frame_Width", "Frame_Height",
    "Brightness_Mean", "Contrast_Mean", "Blur_Index", "Noise_Level", "Compression_Ratio",
    "Tool1_X", "Tool1_Y", "Tool2_X", "Tool2_Y", "Tool1_Angle", "Tool2_Angle",
    "Tool1_Length", "Tool2_Length", "Tool1_Speed", "Tool2_Speed",
    "Tool1_Acceleration", "Tool2_Acceleration", "Tool_Distance", "Tool_Overlap", "Tool_Visibility",
    "Path_Length", "Straight_Path", "Efficiency_Ratio", "Average_Velocity", "Maximum_Velocity",
    "Minimum_Velocity", "Velocity_SD", "Motion_Smoothness", "Motion_Jerk", "Pause_Count",
    "Pause_Time", "Idle_Time", "Motion_Frequency", "Direction_Changes", "Trajectory_Entropy",
    "Camera_X_Shift", "Camera_Y_Shift", "Camera_Rotation", "Camera_Zoom", "Camera_Pan",
    "Camera_Tilt", "Camera_Stability", "Camera_Jitter", "Focus_Quality", "View_Coverage",
    "Flow_Mean", "Flow_Max", "Flow_Min", "Flow_SD", "Flow_Direction",
    "Flow_Entropy", "Flow_Density", "Flow_Curl", "Flow_Divergence", "Frame_Difference",
    "Needle_Pickup_Count", "Needle_Drop_Count", "Suture_Pass_Count", "Knot_Count", "Tissue_Contact_Count",
    "Tissue_Collision_Count", "Instrument_Collision_Count", "Camera_Adjustments", "Error_Count", "Critical_Error_Count",
    "Completion_Time", "Economy_of_Motion", "Precision_Score", "Consistency_Score", "Dexterity_Index",
    "Smoothness_Index", "Error_Rate", "Recovery_Time", "Task_Success", "Overall_Performance",
    "CNN_Feature_1", "CNN_Feature_2", "CNN_Feature_3", "CNN_Feature_4", "CNN_Feature_5",
    "Transformer_Feature_1", "Transformer_Feature_2", "Transformer_Feature_3", "Transformer_Feature_4", "Transformer_Feature_5"
]

assert len(df_final.columns) == 100, f"Expected 100 columns, got {len(df_final.columns)}"
assert list(df_final.columns) == expected_cols, "Column names do not match expected list!"
assert len(df_final) == total_rows, f"Expected exactly {total_rows} rows, got {len(df_final)}"

# Check missing and duplicates
missing_count = df_final.isnull().sum().sum()
assert missing_count == 0, f"Expected 0 missing values, got {missing_count}"

dup_count = df_final.duplicated().sum()
assert dup_count == 0, f"Expected 0 duplicate rows, got {dup_count}"

print("Column shape and structural integrity verified successfully!")

# Save primary dataset
output_csv = "robotic_surgery_dataset.csv"
df_final.to_csv(output_csv, index=False, encoding='utf-8')
print(f"Dataset successfully written to {output_csv}")

# Column details for metadata
col_info = {
    "Trial_ID": ("Unique alphanumeric code identifying each surgical training trial.", "None", "Categorical"),
    "Procedure_ID": ("Unique identifier mapping the trial to a simulated surgical procedure.", "None", "Categorical"),
    "Patient_ID": ("Unique identifier representing the patient/phantom model utilized.", "None", "Categorical"),
    "Procedure_Type": ("Specialty area of the procedure (General Surgery, Urology, etc.).", "None", "Categorical"),
    "Task_Name": ("Specific surgical task performed (e.g., Suturing, Knot Tying).", "None", "Categorical"),
    "Session_Number": ("Sequential training session number of the participant.", "None", "Integer"),
    "Trial_Number": ("Repetition number of the specific task in the session.", "None", "Integer"),
    "Skill_Level": ("Ground-truth expertise category of the operator (Novice, Intermediate, Expert).", "None", "Categorical"),
    "Operating_Room_ID": ("Identifier of the operating room / training simulator bay used.", "None", "Categorical"),
    "Recording_Date": ("Surgical recording timestamp.", "YYYY-MM-DD HH:MM:SS", "String/Datetime"),
    "Video_Duration": ("Total duration of the recorded surgical video.", "Seconds", "Float"),
    "Frame_Rate": ("Temporal capture speed of the camera.", "Frames/Sec (FPS)", "Integer"),
    "Total_Frames": ("Count of total image frames in the recording.", "Frames", "Integer"),
    "Frame_Width": ("Horizontal resolution of the video frames.", "Pixels", "Integer"),
    "Frame_Height": ("Vertical resolution of the video frames.", "Pixels", "Integer"),
    "Brightness_Mean": ("Mean luminosity level of the frame.", "Luma (0-255)", "Float"),
    "Contrast_Mean": ("Mean variance in luma values indicating contrast.", "Index (0-255)", "Float"),
    "Blur_Index": ("Quality metric indexing camera defocus blur.", "Index (0-100)", "Float"),
    "Noise_Level": ("Quantification of sensor noise variance in the frame.", "Index (0-50)", "Float"),
    "Compression_Ratio": ("Video spatial compression ratio.", "Ratio (1-100)", "Float"),
    "Tool1_X": ("Horizontal coordinate of left robotic tool tip in frame.", "Pixels", "Float"),
    "Tool1_Y": ("Vertical coordinate of left robotic tool tip in frame.", "Pixels", "Float"),
    "Tool2_X": ("Horizontal coordinate of right robotic tool tip in frame.", "Pixels", "Float"),
    "Tool2_Y": ("Vertical coordinate of right robotic tool tip in frame.", "Pixels", "Float"),
    "Tool1_Angle": ("Orientation angle of the left robotic tool shaft.", "Degrees", "Float"),
    "Tool2_Angle": ("Orientation angle of the right robotic tool shaft.", "Degrees", "Float"),
    "Tool1_Length": ("Visible length of left tool shaft inside frame.", "Pixels", "Float"),
    "Tool2_Length": ("Visible length of right tool shaft inside frame.", "Pixels", "Float"),
    "Tool1_Speed": ("Linear displacement velocity of left tool tip.", "mm/s", "Float"),
    "Tool2_Speed": ("Linear displacement velocity of right tool tip.", "mm/s", "Float"),
    "Tool1_Acceleration": ("Acceleration rate of left tool tip motion.", "mm/s²", "Float"),
    "Tool2_Acceleration": ("Acceleration rate of right tool tip motion.", "mm/s²", "Float"),
    "Tool_Distance": ("Euclidean separation distance between left and right tool tips.", "mm", "Float"),
    "Tool_Overlap": ("Percentage representing overlapping footprint of tools.", "Percentage (0-100)", "Float"),
    "Tool_Visibility": ("Continuous fraction of the time the tools are visible in frame.", "Fraction (0-1)", "Float"),
    "Path_Length": ("Cumulative path length of tool coordinates during the trial.", "mm", "Float"),
    "Straight_Path": ("Shortest straight-line distance required to complete task.", "mm", "Float"),
    "Efficiency_Ratio": ("Ratio of straight path to path length (straightness index).", "Ratio (0-1)", "Float"),
    "Average_Velocity": ("Combined mean velocity of both instrument tips.", "mm/s", "Float"),
    "Maximum_Velocity": ("Peak instantaneous velocity observed.", "mm/s", "Float"),
    "Minimum_Velocity": ("Lowest motion velocity observed.", "mm/s", "Float"),
    "Velocity_SD": ("Standard deviation of tool tip velocity.", "mm/s", "Float"),
    "Motion_Smoothness": ("Mean motion smoothness index (higher = smoother).", "Index (0-1)", "Float"),
    "Motion_Jerk": ("Kinematic jerk rate metric (derivative of acceleration).", "Index (0-50)", "Float"),
    "Pause_Count": ("Number of distinct pauses (zero motion) observed.", "Count", "Integer"),
    "Pause_Time": ("Cumulative duration of all tool pauses.", "Seconds", "Float"),
    "Idle_Time": ("Total duration in which no tools were active.", "Seconds", "Float"),
    "Motion_Frequency": ("Dominant frequency of instrument tremor/movements.", "Hz", "Float"),
    "Direction_Changes": ("Count of path trajectory direction changes.", "Count", "Integer"),
    "Trajectory_Entropy": ("Statistical entropy representing tool path complexity.", "Index", "Float"),
    "Camera_X_Shift": ("Horizontal displacement of camera frame.", "Pixels", "Float"),
    "Camera_Y_Shift": ("Vertical displacement of camera frame.", "Pixels", "Float"),
    "Camera_Rotation": ("Camera orientation roll angle.", "Degrees (-180 to 180)", "Float"),
    "Camera_Zoom": ("Zoom factor of the camera viewport.", "Magnification", "Float"),
    "Camera_Pan": ("Horizontal panning movement magnitude.", "Pixels", "Float"),
    "Camera_Tilt": ("Vertical tilting movement magnitude.", "Pixels", "Float"),
    "Camera_Stability": ("Mean stability score of the camera capture.", "Score (0-1)", "Float"),
    "Camera_Jitter": ("High-frequency motion deviation of camera.", "Score (0-1)", "Float"),
    "Focus_Quality": ("Quantified image sharpness/focus score.", "Score (0-100)", "Float"),
    "View_Coverage": ("Active percentage of workspace area captured in frame.", "Percentage (0-100)", "Float"),
    "Flow_Mean": ("Mean optical flow magnitude between frames.", "Pixels/Frame", "Float"),
    "Flow_Max": ("Peak optical flow magnitude between frames.", "Pixels/Frame", "Float"),
    "Flow_Min": ("Minimum optical flow magnitude between frames.", "Pixels/Frame", "Float"),
    "Flow_SD": ("Standard deviation of optical flow vectors.", "Pixels/Frame", "Float"),
    "Flow_Direction": ("Principal direction angle of optical flow vectors.", "Degrees", "Float"),
    "Flow_Entropy": ("Information entropy of spatial flow vectors.", "Index", "Float"),
    "Flow_Density": ("Fraction of pixels with active optical flow.", "Fraction (0-1)", "Float"),
    "Flow_Curl": ("Rotational curl magnitude of optical flow fields.", "Rad/Frame", "Float"),
    "Flow_Divergence": ("Divergence expansion magnitude of optical flow fields.", "Rad/Frame", "Float"),
    "Frame_Difference": ("Average absolute pixel change between successive frames.", "Intensity", "Float"),
    "Needle_Pickup_Count": ("Total count of successful needle grasp events.", "Count", "Integer"),
    "Needle_Drop_Count": ("Total count of accidental needle drops.", "Count", "Integer"),
    "Suture_Pass_Count": ("Number of completed suture tissue crossings.", "Count", "Integer"),
    "Knot_Count": ("Number of complete double/single suture knots tied.", "Count", "Integer"),
    "Tissue_Contact_Count": ("Number of intentional or functional tool-tissue interactions.", "Count", "Integer"),
    "Tissue_Collision_Count": ("Total counts of traumatic tool-tissue impacts.", "Count", "Integer"),
    "Instrument_Collision_Count": ("Counts of tool-to-tool crash events.", "Count", "Integer"),
    "Camera_Adjustments": ("Count of camera field-of-view repositioning events.", "Count", "Integer"),
    "Error_Count": ("Total frequency of minor or technical surgical errors.", "Count", "Integer"),
    "Critical_Error_Count": ("Frequency of major errors (e.g. vessel tearing).", "Count", "Integer"),
    "Completion_Time": ("Elapsed time to complete the surgical task.", "Seconds", "Float"),
    "Economy_of_Motion": ("Efficiency score penalizing redundant trajectories.", "Score (0-100)", "Float"),
    "Precision_Score": ("Metric grading tool path target accuracy.", "Score (0-100)", "Float"),
    "Consistency_Score": ("Score measuring velocity and path repeatability.", "Score (0-100)", "Float"),
    "Dexterity_Index": ("Index scoring hand coordination and dexterity.", "Score (0-100)", "Float"),
    "Smoothness_Index": ("Overall motion smoothness score.", "Score (0-100)", "Float"),
    "Error_Rate": ("Frequency of errors per minute.", "Errors/Min", "Float"),
    "Recovery_Time": ("Time spent correcting errors or collisions.", "Seconds", "Float"),
    "Task_Success": ("Binary indicator representing overall task success status.", "Binary (0/1)", "Integer"),
    "Overall_Performance": ("Global performance composite score.", "Score (0-100)", "Float"),
    "CNN_Feature_1": ("First latent visual feature extracted from frame.", "Embedding Value", "Float"),
    "CNN_Feature_2": ("Second latent visual feature extracted from frame.", "Embedding Value", "Float"),
    "CNN_Feature_3": ("Third latent visual feature extracted from frame.", "Embedding Value", "Float"),
    "CNN_Feature_4": ("Fourth latent visual feature extracted from frame.", "Embedding Value", "Float"),
    "CNN_Feature_5": ("Fifth latent visual feature extracted from frame.", "Embedding Value", "Float"),
    "Transformer_Feature_1": ("First temporal sequence context feature extracted.", "Embedding Value", "Float"),
    "Transformer_Feature_2": ("Second temporal sequence context feature extracted.", "Embedding Value", "Float"),
    "Transformer_Feature_3": ("Third temporal sequence context feature extracted.", "Embedding Value", "Float"),
    "Transformer_Feature_4": ("Fourth temporal sequence context feature extracted.", "Embedding Value", "Float"),
    "Transformer_Feature_5": ("Fifth temporal sequence context feature extracted.", "Embedding Value", "Float")
}

# Generate data dictionary CSV
data_dict = []
for col in expected_cols:
    desc, unit, dtype = col_info[col]
    vmin = df_final[col].min()
    vmax = df_final[col].max()
    
    # Format min/max for non-numeric columns
    if dtype in ["Categorical", "String/Datetime", "String"]:
        vmin_str = "N/A"
        vmax_str = "N/A"
    else:
        if "Float" in dtype:
            vmin_str = f"{vmin:.4f}"
            vmax_str = f"{vmax:.4f}"
        else:
            vmin_str = str(int(vmin))
            vmax_str = str(int(vmax))
            
    data_dict.append({
        "Column Name": col,
        "Description": desc,
        "Unit": unit,
        "Data Type": dtype,
        "Minimum": vmin_str,
        "Maximum": vmax_str
    })

df_dict = pd.DataFrame(data_dict)
df_dict.to_csv("data_dictionary.csv", index=False, encoding='utf-8')
print("Data dictionary successfully written to data_dictionary.csv")

# Print verification check correlations
df_final['Skill_Int'] = df_final['Skill_Level'].map({'Novice': 0, 'Intermediate': 1, 'Expert': 2})
corr_time = df_final['Skill_Int'].corr(df_final['Completion_Time'])
corr_smooth = df_final['Skill_Int'].corr(df_final['Motion_Smoothness'])
corr_prec = df_final['Skill_Int'].corr(df_final['Precision_Score'])
corr_err = df_final['Skill_Int'].corr(df_final['Error_Count'])

print("\n--- Key Statistical Correlations (Skill_Level vs Metric) ---")
print(f"Correlation with Completion_Time: {corr_time:.4f} (Expected: Negative)")
print(f"Correlation with Motion_Smoothness: {corr_smooth:.4f} (Expected: Positive)")
print(f"Correlation with Precision_Score: {corr_prec:.4f} (Expected: Positive)")
print(f"Correlation with Error_Count: {corr_err:.4f} (Expected: Negative)")

# Generate README.md
readme_content = f"""# Robot-Assisted Minimally Invasive Surgery (RAMIS) Synthetic Dataset

## Dataset Objective
This dataset contains simulated frame-level and time-window-level observations extracted from robotic surgery training sessions. It is designed to serve as a high-fidelity synthetic benchmark for research in surgical AI, biomedical computer vision, surgical skill evaluation, and machine learning/deep learning model training.

The dataset includes physical metrics, visual qualities, kinematics, optical flow estimates, task-specific event frequencies, and deep embeddings (CNN and Transformer spatial-temporal features) to evaluate surgical competency across three classes: **Novice**, **Intermediate**, and **Expert**.

---

## Dataset Summary Statistics
* **Total Observations (Rows)**: {total_rows:,}
* **Features (Columns)**: 100
* **Skill Level Class Distribution**:
  * **Novice**: {novice_target:,} rows (35.0%)
  * **Intermediate**: {intermediate_target:,} rows (35.0%)
  * **Expert**: {expert_target:,} rows (30.0%)
* **Missing Values**: 0
* **Duplicate Rows**: 0

---

## Generation Methodology
The dataset is generated via a hierarchical mathematical model:
1. **Trial-Level Static Attributes**: Trials (simulated surgical training runs) are established with static properties (e.g., patient ID, procedure type, operating room, base completion times, and overall performance metrics). An latent proficiency score ($p \\in [0,1]$) is assigned using a Beta distribution:
   * Experts: $p \\sim \\text{{Beta}}(8, 2)$
   * Intermediates: $p \\sim \\text{{Beta}}(5, 5)$
   * Novices: $p \\sim \\text{{Beta}}(2, 8)$
2. **Kinematics & Tool Positions**: Tool trajectories are generated using continuous sine/cosine components simulating smooth sweeps. Amplitudes and high-frequency noise parameters are governed by the operator's skill level (Novices exhibit larger spatial movements and high jitter; Experts maintain focused, low-noise movement).
3. **Temporal Sequence Realism**: Every observation has a continuous simulated time element. Recording dates are generated as sequential datetime timestamps by mapping rows to sequential elapsed frames relative to a base date.
4. **Multivariate Correlations**: Features like completion times, errors, tissue collisions, and smoothness parameters are mathematically coupled to the latent proficiency score, establishing realistic dependencies without introducing perfectly linear relationships.
5. **Feature Embeddings**: Spatial (CNN) and temporal (Transformer) features represent deep embeddings extracted from hypothetical video encoders. They are modeled around distinct Gaussian cluster centers for the three skill levels to facilitate cluster analysis, dimensionality reduction, and classification.

---

## Statistical Correlation Assumptions
The dataset embodies several key surgical validation assumptions verified during generation:
* **Novices** show longer completion times, lower movement efficiency (high path length relative to straight path), high jerk index, more frequent tool pauses, high error counts, and lower precision.
* **Experts** show short completion times, highly efficient motion paths, high precision, low jerk, minimal collisions, and high dexterity.
* **Intermediates** represent a transitional performance profile.

### Computed Correlation Coefficients (Pearson $r$)
* **Skill Level vs. Completion Time**: {corr_time:.4f} (Strong negative correlation, indicating faster completion by experts)
* **Skill Level vs. Motion Smoothness**: {corr_smooth:.4f} (Strong positive correlation, indicating smoother motions by experts)
* **Skill Level vs. Precision Score**: {corr_prec:.4f} (Strong positive correlation, indicating higher precision by experts)
* **Skill Level vs. Error Count**: {corr_err:.4f} (Strong negative correlation, indicating fewer errors by experts)

---

## Potential AI Applications
This dataset is suitable for training and evaluating several machine learning systems:
1. **Surgical Skill Classification**: Training classifiers (Random Forests, Gradient Boosting, SVMs, or Deep Neural Networks) to automatically grade surgeon performance from kinematic time series.
2. **Visual Embedding Analytics**: Utilizing the 5-dimensional CNN and Transformer features to perform unsupervised clustering (t-SNE, UMAP, PCA) representing video frame distributions.
3. **Event Detection & Safety Audits**: Regression models for predicting collision rates, or classifying safety risk categories based on tool distance and optical flow.
4. **Multi-Task Surgical Assessment**: Jointly predicting the success of tasks, overall performance scores, and error rates from time-window aggregates.

---

## Column Descriptions
For a full list of all 100 features, their descriptions, types, and ranges, please refer to the accompanying [data_dictionary.csv](file:///data_dictionary.csv).
"""

with open("README.md", "w", encoding='utf-8') as f:
    f.write(readme_content)
print("README.md successfully written")
