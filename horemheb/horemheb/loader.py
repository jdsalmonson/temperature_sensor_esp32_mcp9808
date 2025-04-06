import pandas as pd
from pandas.core.frame import DataFrame
from .config import AnalysisConfig
from typing import Optional, Tuple, Literal

def preprocess_temperature_file(
    df: DataFrame, 
    cutoff_time: Optional[str] = None,
    comparison: Literal['>=', '<='] = '>='
) -> DataFrame:
    """
    Preprocess temperature data file with optional time filtering
    
    Args:
        df: Input DataFrame
        cutoff_time: Optional timestamp to filter data (e.g., '2025-02-27 19:48:04')
        comparison: Direction of time filtering, either '>=' or '<='
    """
    # Apply time filtering if cutoff_time is provided
    if cutoff_time is not None:
        if comparison == '>=':
            df = df[df['Timestamp'] >= cutoff_time]
        elif comparison == '<=':
            df = df[df['Timestamp'] <= cutoff_time]
        else:
            raise ValueError("comparison must be '>=' or '<='")
    
    # Rename columns
    df = df.rename(columns={
        'Temperature Sensor 1._temperature._tcp.local.': 'temp1',
        'Temperature Sensor 2._temperature._tcp.local.': 'temp2'
    })
    df['timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.drop('Timestamp', axis=1)
    df['temp1'] = pd.to_numeric(df['temp1'], errors='coerce')
    df['temp2'] = pd.to_numeric(df['temp2'], errors='coerce')
    return df

def load_temperature_data(
    filename: str, 
    config: AnalysisConfig,
    cutoff_time: Optional[str] = None,
    comparison: Literal['>=', '<='] = '>='
) -> Tuple[DataFrame, pd.Series, pd.Series]:
    """
    Load and preprocess temperature data from CSV file
    
    Args:
        filename: Path to CSV file
        config: Analysis configuration
        cutoff_time: Optional timestamp to filter data (e.g., '2025-02-27 19:48:04')
        comparison: Direction of time filtering, either '>=' or '<='
    """
    df = pd.read_csv(filename)
    df = preprocess_temperature_file(df, cutoff_time, comparison)
    
    # Resample data
    df_resampled1 = df.set_index('timestamp').resample(f'{config.resample_minutes}min')['temp1'].mean()
    df_resampled2 = df.set_index('timestamp').resample(f'{config.resample_minutes}min')['temp2'].mean()
    
    return df, df_resampled1, df_resampled2
