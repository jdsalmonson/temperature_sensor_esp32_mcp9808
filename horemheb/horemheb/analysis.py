import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd
import matplotlib.dates as mdates
from .config import AnalysisConfig

@dataclass
class FitResults:
    A: float  # slope for y = Ax + b
    A_err: float  # error in slope
    b: float  # intercept
    b_err: float  # error in intercept
    A0: float  # slope for y = A0x
    A0_err: float  # error in slope for origin fit
    r_squared: float  # R-squared for y = Ax + b
    r_squared0: float  # R-squared for y = A0x

def linear_func(x, m, b):
    return m*x + b

def linear_func_through_origin(x, m):
    return m*x

def quadratic_func_through_origin(x, a, b):
    return a*x**2 + b*x

def r_squared_through_origin(x, y, m):
    y_pred = m * x
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum(y**2)
    return 1 - (ss_res / ss_tot)

def analyze_segments(segments: List[Dict[str, Any]], config: AnalysisConfig) -> FitResults:
    """Analyze temperature segments and perform linear fits"""
    x_all = []
    y_all = []
    
    for segment in segments:
        temp_diff = segment['temp1'] - segment['temp2']
        
        regular_time = pd.date_range(
            start=segment['temp1'].index[0],
            end=segment['temp1'].index[-1],
            freq='1min'
        )
        
        temp_interp = pd.Series(
            index=regular_time,
            data=np.interp(mdates.date2num(regular_time),
                          mdates.date2num(segment['temp1'].index),
                          segment['temp1'].values)
        )
        
        # Calculate derivative and smooth it
        window = 60
        dt = 1/60
        dT_dt = np.gradient(temp_interp.values) / dt
        dT_dt_smooth = pd.Series(
            index=regular_time,
            data=np.convolve(dT_dt, np.ones(window)/window, mode='same')
        )
        
        x_data = -dT_dt_smooth
        y_data = pd.Series(
            index=regular_time,
            data=np.interp(mdates.date2num(regular_time),
                          mdates.date2num(temp_diff.index),
                          temp_diff.values)
        )
        
        # Use only data between delay_time and sunrise
        mask_main = (regular_time >= segment['delay_time']) & (regular_time < segment['sunrise_time'])
        valid_data = ~np.isnan(x_data[mask_main]) & ~np.isnan(y_data[mask_main])
        
        if len(valid_data) > 0:
            x_section = x_data[mask_main][valid_data]
            y_section = y_data[mask_main][valid_data]
            x_all.extend(x_section)
            y_all.extend(y_section)
    
    x_all = np.array(x_all)
    y_all = np.array(y_all)
    
    # Perform fits and get parameter errors from covariance matrix
    popt, pcov = curve_fit(linear_func, x_all, y_all)
    # quick test of quadratic fit:
    #popt, pcov = curve_fit(quadratic_func_through_origin, x_all, y_all)
    A, b = popt
    A_err, b_err = np.sqrt(np.diag(pcov))
    
    popt0, pcov0 = curve_fit(linear_func_through_origin, x_all, y_all)
    A0 = popt0[0]
    A0_err = np.sqrt(pcov0[0][0])
    
    # Calculate R-squared values
    residuals = y_all - linear_func(x_all, A, b)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_all - np.mean(y_all))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    r_squared0 = r_squared_through_origin(x_all, y_all, A0)
    
    return FitResults(
        A=A, 
        A_err=A_err,
        b=b, 
        b_err=b_err,
        A0=A0,
        A0_err=A0_err,
        r_squared=r_squared, 
        r_squared0=r_squared0
    )


def time_dependent_func(X, A, b):
    """
    Fitting function of form A*x/(1 + b*t)
    X should be an array of [x, t] pairs
    """
    x, t = X
    return A * x / (1 + b * t)

def analyze_segments_time_dependent(segments: List[Dict[str, Any]], config: AnalysisConfig) -> FitResults:
    """Analyze temperature segments and perform fit with time dependence"""
    x_all = []
    y_all = []
    t_all = []  # to store time differences
    
    for segment in segments:
        temp_diff = segment['temp1'] - segment['temp2']
        
        regular_time = pd.date_range(
            start=segment['temp1'].index[0],
            end=segment['temp1'].index[-1],
            freq='1min'
        )
        
        temp_interp = pd.Series(
            index=regular_time,
            data=np.interp(mdates.date2num(regular_time),
                          mdates.date2num(segment['temp1'].index),
                          segment['temp1'].values)
        )
        
        # Calculate derivative and smooth it
        window = 60
        dt = 1/60
        dT_dt = np.gradient(temp_interp.values) / dt
        dT_dt_smooth = pd.Series(
            index=regular_time,
            data=np.convolve(dT_dt, np.ones(window)/window, mode='same')
        )
        
        x_data = -dT_dt_smooth
        y_data = pd.Series(
            index=regular_time,
            data=np.interp(mdates.date2num(regular_time),
                          mdates.date2num(temp_diff.index),
                          temp_diff.values)
        )
        
        # Use only data between delay_time and sunrise
        mask_main = (regular_time >= segment['delay_time']) & (regular_time < segment['sunrise_time'])
        valid_data = ~np.isnan(x_data[mask_main]) & ~np.isnan(y_data[mask_main])
        
        if len(valid_data) > 0:
            x_section = x_data[mask_main][valid_data]
            y_section = y_data[mask_main][valid_data]
            
            # Calculate time since delay_time in hours
            t_section = ((regular_time[mask_main][valid_data] - segment['delay_time'])
                        .total_seconds() / 3600)  # convert to hours
            #print(t_section)
            
            x_all.extend(x_section)
            y_all.extend(y_section)
            t_all.extend(t_section)
    
    x_all = np.array(x_all)
    y_all = np.array(y_all)
    t_all = np.array(t_all)
    
    # Perform fit with time dependence
    popt, pcov = curve_fit(lambda X, A, b: time_dependent_func((X[0], X[1]), A, b),
                          (x_all, t_all), y_all)
    
    A, b = popt
    A_err, b_err = np.sqrt(np.diag(pcov))
    
    # Calculate R-squared
    y_pred = time_dependent_func((x_all, t_all), A, b)
    residuals = y_all - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_all - np.mean(y_all))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Return results (note: some fields will be filled with zeros as they don't apply)
    return FitResults(
        A=A,
        A_err=A_err,
        b=b,
        b_err=b_err,
        A0=0.0,  # not applicable for this fit
        A0_err=0.0,  # not applicable for this fit
        r_squared=r_squared,
        r_squared0=0.0  # not applicable for this fit
    )