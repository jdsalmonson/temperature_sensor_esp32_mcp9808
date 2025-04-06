# temperature_analysis/segments.py
from datetime import timedelta
import pandas as pd
from typing import List, Dict
from .config import AnalysisConfig
from astral import LocationInfo
from astral.sun import sun
from zoneinfo import ZoneInfo

def get_sunrise(date):
    """Get sunrise time for Livermore, CA"""
    livermore = LocationInfo('Livermore', 'California', 'US', 37.6819, -121.7680)
    s = sun(livermore.observer, date=date)
    return s['sunrise'].astimezone(ZoneInfo('America/Los_Angeles')).replace(tzinfo=None)

def find_cooling_segments(df_resampled1: pd.Series, 
                         df_resampled2: pd.Series, 
                         N_cooling: int = 10, 
                         N_warming: int = 4) -> List[Dict]:
    """Find cooling segments in temperature data"""
    temp_diff = df_resampled1.diff()
    not_increasing = temp_diff <= 0
    segments = []
    i = 0
    
    while i < len(not_increasing) - N_cooling:
        # Look for cooling sequence start
        run_length = 0
        while i < len(not_increasing) - N_cooling:
            if not_increasing.iloc[i]:
                run_length += 1
                if run_length == N_cooling:
                    start_idx = i - N_cooling + 1
                    break
            else:
                run_length = 0
            i += 1
            
        if i >= len(not_increasing) - N_cooling:
            break
            
        # Find end of sequence
        end_idx = start_idx + N_cooling
        while end_idx < len(temp_diff) - N_warming:
            if all(temp_diff.iloc[end_idx:end_idx + N_warming] > 0):
                break
            end_idx += 1
            
        # Get timestamps
        start_time = not_increasing.index[start_idx]
        end_time = not_increasing.index[end_idx]
        
        # Select data for this range
        mask = (df_resampled1.index >= start_time) & (df_resampled1.index < end_time)
        segment = {
            'start_time': start_time,
            'end_time': end_time,
            'temp1': df_resampled1[mask],
            'temp2': df_resampled2[mask]
        }
        segments.append(segment)
        
        i = end_idx + 1
    
    return segments

def add_segment_times(segment: Dict, delta_minutes: int = 60, before_sunrise_delta_minutes: int = 30) -> Dict:
    """Add delay_time and sunrise_time to segment"""
    start_time = segment['start_time'].to_pydatetime() if hasattr(segment['start_time'], 'to_pydatetime') else segment['start_time']
    end_time = segment['end_time'].to_pydatetime() if hasattr(segment['end_time'], 'to_pydatetime') else segment['end_time']
    
    segment['delay_time'] = start_time + timedelta(minutes=delta_minutes)
    last_date = end_time.date()
    segment['sunrise_time'] = get_sunrise(last_date) - timedelta(minutes=before_sunrise_delta_minutes)
    return segment

def is_sunrise_between(segment):
    """Check if sunrise_time falls between start_time and end_time for a segment"""
    start = segment['start_time'].to_pydatetime() if hasattr(segment['start_time'], 'to_pydatetime') else segment['start_time']
    end = segment['end_time'].to_pydatetime() if hasattr(segment['end_time'], 'to_pydatetime') else segment['end_time']
    sunrise = segment['sunrise_time']
    
    return start <= sunrise <= end

def process_segments(df_resampled1: pd.Series, 
                    df_resampled2: pd.Series, 
                    config: AnalysisConfig) -> List[Dict]:
    """Find and process cooling segments"""
    segments = find_cooling_segments(df_resampled1, df_resampled2)
    segments = [add_segment_times(segment, config.delay_time, config.before_sunrise_delta_minutes) for segment in segments]
    return [segment for segment in segments if is_sunrise_between(segment)]
