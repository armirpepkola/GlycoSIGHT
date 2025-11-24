import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
from bs4 import BeautifulSoup
import numpy as np


def parse_xml_and_simulate_events(file_path):
    print(f"Parsing XML and simulating events for: {os.path.basename(file_path)}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    soup = BeautifulSoup(content, 'xml')
    glucose_level_tag = soup.find('glucose_level')
    if not glucose_level_tag: return pd.DataFrame()

    events = glucose_level_tag.find_all('event')

    data_list = []
    for event in events:
        data_list.append({
            'timestamp': pd.to_datetime(event.get('ts'), dayfirst=True, errors='coerce'),
            'glucose_value': float(event.get('value')) if event.get('value') else None
        })

    if not data_list: return pd.DataFrame()

    df = pd.DataFrame(data_list).dropna().set_index('timestamp').sort_index()
    if df.empty: return pd.DataFrame()

    df['bolus_value'] = 0.0
    df['meal_carbs'] = 0.0
    df['glucose_diff'] = df['glucose_value'].diff().fillna(0)

    meal_threshold = 20
    potential_meal_times = df.index[df['glucose_diff'].rolling(window=6).sum() > meal_threshold]

    if potential_meal_times.empty:
        return df.reset_index()

    last_meal_time = df.index[0]

    for meal_time in potential_meal_times:
        if (meal_time - last_meal_time).total_seconds() / 3600 > 3:
            try:
                meal_start_time_target = meal_time - pd.Timedelta(minutes=30)

                start_loc_pos = (np.abs(df.index - meal_start_time_target)).argmin()
                meal_start_ts = df.index[start_loc_pos]

                end_time_for_rise_target = meal_time + pd.Timedelta(hours=2)
                end_loc_pos = (np.abs(df.index - end_time_for_rise_target)).argmin()
                meal_end_ts_for_rise = df.index[end_loc_pos]

                rise_magnitude = df.loc[meal_time:meal_end_ts_for_rise, 'glucose_value'].max() - df.loc[
                    meal_start_ts, 'glucose_value']
                simulated_carbs = max(20, min(120, rise_magnitude * 0.5))
                simulated_bolus = simulated_carbs / 15.0

                df.loc[meal_start_ts, 'meal_carbs'] += simulated_carbs
                df.loc[meal_start_ts, 'bolus_value'] += simulated_bolus

                last_meal_time = meal_time
            except Exception:
                continue

    return df.reset_index()


def preprocess_patient_data(file_path, output_dir):
    df = parse_xml_and_simulate_events(file_path)
    if df.empty:
        print(f"  -> Warning: No valid data found in {os.path.basename(file_path)}. Skipping this file.")
        return
    df = df.set_index('timestamp')
    df_resampled = df.resample('5min').mean()
    df_resampled['glucose_value'] = df_resampled['glucose_value'].interpolate(method='linear', limit_direction='both')
    df_events = df[['bolus_value', 'meal_carbs']].resample('5min').sum()
    df_resampled['bolus_value'] = df_events['bolus_value']
    df_resampled['meal_carbs'] = df_events['meal_carbs']
    df_resampled.dropna(subset=['glucose_value'], inplace=True)
    df = df_resampled.rename(
        columns={'glucose_value': 'glucose', 'bolus_value': 'insulin_bolus', 'meal_carbs': 'manual_carb_input'})
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['glucose'] = scaler.fit_transform(df[['glucose']])
    patient_id = os.path.basename(file_path).split('-')[0]
    scaler_path = os.path.join(output_dir, f'scaler_patient_{patient_id}.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    df['hfa_c'] = 0.0
    df['hfa_f'] = 0.0
    df['hfa_p'] = 0.0
    output_path = os.path.join(output_dir, f'processed_patient_{patient_id}.csv')
    df.to_csv(output_path)
    print(f"Saved processed data with simulated events to {output_path}")


if __name__ == '__main__':
    raw_data_dir = './raw_data/OhioT1DM-training'
    processed_data_dir = './processed_data'
    os.makedirs(processed_data_dir, exist_ok=True)
    files_to_process = [f for f in os.listdir(raw_data_dir) if f.endswith('.xml')]
    if not files_to_process:
        print(f"Error: No XML files found in '{raw_data_dir}'.")
    else:
        for file_name in files_to_process:
            file_path = os.path.join(raw_data_dir, file_name)
            preprocess_patient_data(file_path, processed_data_dir)