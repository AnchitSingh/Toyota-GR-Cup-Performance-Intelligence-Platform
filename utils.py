# utils.py - Updated with flexible parameter names

import pandas as pd
import numpy as np
from pathlib import Path

def get_lap_telemetry(telemetry_df, vehicle_id, lap_num):
    """Extract and pivot telemetry for one lap"""
    
    # Check if this is proper telemetry data
    if 'telemetry_name' not in telemetry_df.columns or 'telemetry_value' not in telemetry_df.columns:
        # Not a proper telemetry file
        return None
    
    lap_data = telemetry_df[
        (telemetry_df['vehicle_id'] == vehicle_id) & 
        (telemetry_df['lap'] == lap_num)
    ].copy()
    
    if len(lap_data) == 0:
        return None
    
    lap_wide = lap_data.pivot_table(
        index=['timestamp', 'vehicle_id', 'lap'],
        columns='telemetry_name',
        values='telemetry_value',
        aggfunc='first'
    ).reset_index()
    
    lap_wide = lap_wide.sort_values('timestamp').reset_index(drop=True)
    return lap_wide


def find_column(df, possible_names):
    """Find column by trying multiple possible names"""
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def detect_corners(lap_telemetry, throttle_threshold=80, min_corner_length=10):
    """Detect corners by throttle drops - flexible parameter names"""
    
    # Try to find throttle column
    throttle_col = find_column(lap_telemetry, ['ath', 'ATH', 'throttle', 'Throttle', 'TPS', 'tps', 'aps', 'APS'])

    if throttle_col is None:
        # No throttle data - skip this lap
        return []
    
    throttle = lap_telemetry[throttle_col].fillna(100)
    in_corner = throttle < throttle_threshold
    
    corners = []
    corner_start = None
    
    for i, is_corner in enumerate(in_corner):
        if is_corner and corner_start is None:
            corner_start = i
        elif not is_corner and corner_start is not None:
            if i - corner_start >= min_corner_length:
                corners.append({
                    'start': corner_start,
                    'end': i,
                    'apex': corner_start + throttle[corner_start:i].argmin()
                })
            corner_start = None
    
    return corners

def extract_corner_features(lap_telemetry, corners):
    """Extract physics-based features for each corner - flexible parameter names"""
    
    # Find column names flexibly
    throttle_col = find_column(lap_telemetry, ['ath', 'ATH', 'throttle', 'Throttle', 'TPS', 'tps', 'aps', 'APS'])
    brake_f_col = find_column(lap_telemetry, ['pbrake_f', 'brake_f', 'Brake_F', 'brake_front'])
    brake_r_col = find_column(lap_telemetry, ['pbrake_r', 'brake_r', 'Brake_R', 'brake_rear'])
    accy_col = find_column(lap_telemetry, ['accy_can', 'accy', 'lateral_accel', 'AccY'])
    steering_col = find_column(lap_telemetry, ['Steering_Angle', 'steering', 'Steering', 'steer'])
    
    features = []
    
    for i, corner in enumerate(corners, 1):
        start, apex, end = corner['start'], corner['apex'], corner['end']
        
        if start >= len(lap_telemetry) or end >= len(lap_telemetry):
            continue
        
        corner_data = lap_telemetry.iloc[start:end+1]
        
        # Extract features with safe defaults
        feature = {
            'corner_num': i,
            'start_idx': start,
            'apex_idx': apex,
            'end_idx': end,
            'corner_duration': end - start,
        }
        
        # Throttle features
        if throttle_col:
            feature['entry_throttle'] = lap_telemetry.iloc[start][throttle_col] if pd.notna(lap_telemetry.iloc[start][throttle_col]) else 0
            feature['apex_throttle'] = lap_telemetry.iloc[apex][throttle_col] if pd.notna(lap_telemetry.iloc[apex][throttle_col]) else 0
            feature['min_throttle'] = corner_data[throttle_col].min()
            feature['exit_throttle'] = lap_telemetry.iloc[end][throttle_col] if pd.notna(lap_telemetry.iloc[end][throttle_col]) else 0
        else:
            feature.update({'entry_throttle': 0, 'apex_throttle': 0, 'min_throttle': 0, 'exit_throttle': 0})
        
        # Brake features
        if brake_f_col:
            feature['max_brake'] = corner_data[brake_f_col].max()
            feature['brake_duration'] = (corner_data[brake_f_col] > 10).sum()
        else:
            feature.update({'max_brake': 0, 'brake_duration': 0})
        
        # Lateral G
        if accy_col:
            feature['apex_lateral_g'] = abs(lap_telemetry.iloc[apex][accy_col]) if pd.notna(lap_telemetry.iloc[apex][accy_col]) else 0
        else:
            feature['apex_lateral_g'] = 0
        
        # Steering
        if steering_col:
            feature['avg_steering_angle'] = abs(corner_data[steering_col]).mean()
        else:
            feature['avg_steering_angle'] = 0
        
        # Throttle application point
        if throttle_col and (corner_data[throttle_col] > 50).any():
            feature['throttle_application_point'] = (corner_data[throttle_col] > 50).idxmax()
        else:
            feature['throttle_application_point'] = end
        
        features.append(feature)
    
    return pd.DataFrame(features)
