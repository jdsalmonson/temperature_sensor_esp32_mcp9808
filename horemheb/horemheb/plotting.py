import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#from scipy.optimize import curve_fit
from .analysis import linear_func, linear_func_through_origin


def plot_linear_fit(segments, results, sup_title, 
                    config=None, 
                    smoothed_window=60,
                    plot_Ax_b_fit: bool = True
                    ):
    """
    Plot linear fit analysis for a single door type.
    
    Args:
        segments: List of segments for the door
        results: Analysis results from analyze_segments()
        sup_title: Plot title
        config: AnalysisConfig (default: None)
        smoothed_window: Window size for smoothing (default: 60)
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), height_ratios=[2, 1])
    
    # Plot data and fit
    colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
    
    x_all = []
    for i, segment in enumerate(segments):
        # Get temperature difference data
        temp_diff = segment['temp1'] - segment['temp2']
        
        # Calculate cooling rate
        regular_time = pd.date_range(start=segment['temp1'].index[0], 
                                   end=segment['temp1'].index[-1],
                                   freq='1min')
        
        temp_interp = pd.Series(index=regular_time, 
                               data=np.interp(mdates.date2num(regular_time),
                                            mdates.date2num(segment['temp1'].index),
                                            segment['temp1'].values))
        
        # Calculate derivative and smooth it
        dt = 1/60  # time step in hours
        dT_dt = np.gradient(temp_interp.values) / dt
        dT_dt_smooth = pd.Series(index=regular_time,
                                data=np.convolve(dT_dt, np.ones(smoothed_window)/smoothed_window, 
                                               mode='same'))
        
        # Get x and y data
        x_data = -dT_dt_smooth
        y_data = pd.Series(index=regular_time,
                          data=np.interp(mdates.date2num(regular_time),
                                       mdates.date2num(temp_diff.index),
                                       temp_diff.values))
        
        # Create masks for different time periods
        mask_before_delay = regular_time < segment['delay_time']
        mask_main = (regular_time >= segment['delay_time']) & (regular_time < segment['sunrise_time'])
        mask_after_sunrise = regular_time >= segment['sunrise_time']
        
        # Plot each section with appropriate style
        for mask, style in [(mask_before_delay, '--'), (mask_main, '-'), (mask_after_sunrise, '--')]:
            valid_data = ~np.isnan(x_data[mask]) & ~np.isnan(y_data[mask])
            if len(valid_data) > 0:
                x_section = x_data[mask][valid_data]
                y_section = y_data[mask][valid_data]
                x_all.extend(x_section)
                if len(x_section) > 0:
                    # Plot data
                    ax1.plot(x_section, y_section, f'{style}', color=colors[i], 
                            label=f'Segment {i+1}' if mask is mask_main else "", 
                            alpha=0.7)
                    
                    # Plot residuals for main section only
                    if mask is mask_main:
                        residuals = y_section - linear_func(x_section, results.A, results.b)
                        ax2.plot(x_section, residuals, f'{style}', color=colors[i],
                                alpha=0.7)

    # Plot the fit lines using pre-calculated results
    # x_all = np.concatenate([seg['temp1'].values for seg in segments])
    x_fit = np.linspace(min(x_all), max(x_all), 100)

    fit_line_0, = ax1.plot(x_fit, linear_func_through_origin(x_fit, results.A0), 'k-', 
                            label=f'Fit: {results.A0:.3f}x')
    
    title_Ax_b = ''
    if plot_Ax_b_fit:
        fit_line, = ax1.plot(x_fit, linear_func(x_fit, results.A, results.b), 'k--', 
                            label=f'Fit: {results.A:.3f}x + {results.b:.3f}')
        
        title_Ax_b = f'temp_diff = {results.A:.3f}*cooling_rate + {results.b:.3f}, R² = {results.r_squared:.3f}\nb=0 fit: '

    
    if config is not None:
        delay_time = config.delay_time
        before_sunrise_delta_minutes = config.before_sunrise_delta_minutes
        title_suffix = f'(excluding first {delay_time} minutes and last {before_sunrise_delta_minutes} minutes before sunrise of each sequence)\n'
    else:
        title_suffix = ''

    # Customize plots
    ax1.set_xlabel('Cooling Rate -dT$_{i}$/dt (°C/hour)')
    ax1.set_ylabel('Temperature Difference T$_{i}$ - T$_{o}$ (°C)')
    ax1.set_title('Temperature Difference vs Cooling Rate\n' +
                  title_suffix + title_Ax_b +
                  '(T$_{i}$ - T$_{o}$) = ' + f'{results.A0:.3f}' + '*(dT$_{i}$/dt), ' + f'R² = {results.r_squared0:.3f}')
    ax1.set_ylim(0, 14.2)
    ax1.legend(loc='lower right')
    ax1.grid(True)

    ax2.set_xlabel('Cooling Rate -dT$_{i}$/dt (°C/hour)')
    ax2.set_ylabel('Residuals T$_{i}$ - T$_{o}$ - (A*dT$_{i}$/dt + b) (°C)')
    ax2.set_title('Residuals')
    ax2.grid(True)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    plt.suptitle(sup_title)
    plt.tight_layout()
    plt.show()

def plot_linear_fit_comparison(new_segments, old_segments, new_results, old_results, sup_title, 
                               smoothed_window=60, 
                               plot_Ax_b_fit: bool = True):
    """
    Plot and compare fits for both new and old door segments.
    
    Args:
        new_segments: List of segments for new door
        old_segments: List of segments for old door
        new_results: Analysis results for new door from analyze_segments()
        old_results: Analysis results for old door from analyze_segments()
        sup_title: Plot title
        smoothed_window: Window size for smoothing (default: 60)
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), height_ratios=[2, 1])
    
    new_handles = []
    old_handles = []
    new_labels = []
    old_labels = []
    
    # Process each door type
    for segments, results, door_type, marker, color in [
        (new_segments, new_results, 'New Door', 'o', 'blue'),
        (old_segments, old_results, 'Old Door', 's', 'red')
    ]:
        # Generate unique colors for each door type's segments
        segment_colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
        
        x_all = []
        for i, segment in enumerate(segments):
            # Get temperature difference data
            temp_diff = segment['temp1'] - segment['temp2']
            
            # Calculate cooling rate
            regular_time = pd.date_range(start=segment['temp1'].index[0], 
                                       end=segment['temp1'].index[-1],
                                       freq='1min')
            
            temp_interp = pd.Series(index=regular_time, 
                                   data=np.interp(mdates.date2num(regular_time),
                                                mdates.date2num(segment['temp1'].index),
                                                segment['temp1'].values))
            
            # Calculate derivative and smooth it
            dt = 1/60  # time step in hours
            dT_dt = np.gradient(temp_interp.values) / dt
            dT_dt_smooth = pd.Series(index=regular_time,
                                   data=np.convolve(dT_dt, np.ones(smoothed_window)/smoothed_window, 
                                                  mode='same'))
            
            # Get x and y data
            x_data = -dT_dt_smooth
            y_data = pd.Series(index=regular_time,
                             data=np.interp(mdates.date2num(regular_time),
                                          mdates.date2num(temp_diff.index),
                                          temp_diff.values))
            
            # Only plot between delay_time and sunrise_time
            mask_main = (regular_time >= segment['delay_time']) & (regular_time < segment['sunrise_time'])
            
            # Remove any NaN values from main section
            valid_data = ~np.isnan(x_data[mask_main]) & ~np.isnan(y_data[mask_main])
            if len(valid_data) > 0:
                x_section = x_data[mask_main][valid_data]
                y_section = y_data[mask_main][valid_data]
                x_all.extend(x_section)
                # Plot data points
                #label = f'{door_type} Segment {i+1}' if i == 0 else f'Segment {i+1}'
                label = f'Segment {i+1}'
                #scatter = ax1.scatter(x_section, y_section, marker=marker, color=segment_colors[i], 
                #                    label=label, alpha=0.7, s=20)
                style = '-' if "New" in door_type else '--'
                line_handle, = ax1.plot(x_section, y_section, f'{style}', color=segment_colors[i], 
                            label=label, alpha=0.7) #, markersize=3)
                
                if door_type == 'New Door':
                    new_handles.append(line_handle)
                    new_labels.append(label)
                else:
                    old_handles.append(line_handle)
                    old_labels.append(label)
                
                # Plot residuals
                residuals = y_section - linear_func(x_section, results.A, results.b)
                #ax2.scatter(x_section, residuals, marker=marker, color=segment_colors[i],
                #          alpha=0.7, s=20)
                ax2.plot(x_section, residuals, f'{style}', color=segment_colors[i], 
                            label=label, alpha=0.7) #, markersize=3)

        # Plot fit lines
        #x_all = np.concatenate([seg['temp1'].values for seg in segments])
        x_fit = np.linspace(min(x_all), max(x_all), 100)
        
        fit_line_0, = ax1.plot(x_fit, linear_func_through_origin(x_fit, results.A0), '-', 
                              color=color, label=f'Fit: {results.A0:.3f}x, R² = {results.r_squared0:.3f}')
        if plot_Ax_b_fit:
            fit_line, = ax1.plot(x_fit, linear_func(x_fit, results.A, results.b), '--', 
                                color=color, label=f'Ax+bFit: {results.A:.3f}x + {results.b:.3f}, R² = {results.r_squared:.3f}')                
        
        if door_type == 'New Door':
            new_handles.extend([fit_line_0])
            new_labels.extend([fit_line_0.get_label()])
            if plot_Ax_b_fit:
                new_handles.append(fit_line)
                new_labels.append(fit_line.get_label())
        else:
            old_handles.extend([fit_line_0])
            old_labels.extend([fit_line_0.get_label()])
            if plot_Ax_b_fit:
                old_handles.append(fit_line)
                old_labels.append(fit_line.get_label())
    
    # Create separate legends for new and old door data
    # new_labels = [h.get_label() for h in new_handles]
    # old_labels = [h.get_label() for h in old_handles]
    
    # ax1.legend(new_handles, new_labels, loc='upper left', title='New Door')
    ax1.add_artist(ax1.legend(new_handles, new_labels, loc='upper left', title='New Door'))
    ax1.add_artist(ax1.legend(old_handles, old_labels, loc='lower right', title='Old Door'))
    
    # Customize plots
    ax1.set_xlabel('Cooling Rate -dT$_{i}$/dt (°C/hour)')
    ax1.set_ylabel('Temperature Difference T$_{i}$ - T$_{o}$ (°C)')
    ax1.set_title('Temperature Difference vs Cooling Rate\n(between delay_time and sunrise_time)')
    ax1.set_ylim(0, 14.2)
    ax1.set_xlim(0, None)
    ax1.grid(True)

    ax2.set_xlabel('Cooling Rate -dT$_{i}$/dt (°C/hour)')
    ax2.set_ylabel('Residuals T$_{i}$ - T$_{o}$ - (A*dT$_{i}$/dt) (°C)')
    ax2.set_title('Residuals')
    ax2.grid(True)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    plt.suptitle(sup_title)
    plt.tight_layout()
    plt.show()


def plot_all_segments(segments, results, sup_title, config=None, smoothed_window=60):
    A, b = results.A, results.b
    if config is not None:
        suffix = f" ({config.resample_minutes}min avg)"
    else:
        suffix = ""

    fig, axes = plt.subplots(len(segments), 2, figsize=(15, 6*len(segments)))
    if len(segments) == 1:
        axes = [axes]

    for i, (ax1, ax2) in enumerate(axes):
        segment = segments[i]
            
        # Create secondary y-axes
        ax1_f = ax1.twinx()  # Fahrenheit axis
        ax2_rate = ax2.twinx()  # Cooling rate axis
        
        # Plot temperatures on first subplot (keeping original style)
        ax1.plot(segment['temp1'].index, segment['temp1'].values, '-r', 
                linewidth=1.5, label='Sensor 1: T$_{i}$' + suffix)
        ax1.plot(segment['temp2'].index, segment['temp2'].values, '-b', 
                linewidth=1.5, label='Sensor 2: T$_{o}$' + suffix)
        
        # Calculate temperature difference
        temp_diff = segment['temp1'] - segment['temp2']
        
        # Split temp_diff into three periods
        mask_before_delay = temp_diff.index < segment['delay_time']
        mask_main = (temp_diff.index >= segment['delay_time']) & (temp_diff.index < segment['sunrise_time'])
        mask_after_sunrise = temp_diff.index >= segment['sunrise_time']
        
        # Plot temperature difference in three parts
        ax2.plot(temp_diff[mask_before_delay].index, temp_diff[mask_before_delay].values, '--k', 
                linewidth=1.5, label='T$_{i}$ - T$_{o}$ (before delay)')
        ax2.plot(temp_diff[mask_main].index, temp_diff[mask_main].values, '-k', 
                linewidth=1.5, label='T$_{i}$ - T$_{o}$ (main)')
        ax2.plot(temp_diff[mask_after_sunrise].index, temp_diff[mask_after_sunrise].values, '--k', 
                linewidth=1.5, label='T$_{i}$ - T$_{o}$ (after sunrise)')
        
        # Calculate cooling rate
        regular_time = pd.date_range(start=segment['temp1'].index[0], 
                                end=segment['temp1'].index[-1],
                                freq='1min')
        
        temp_interp = pd.Series(index=regular_time, 
                            data=np.interp(mdates.date2num(regular_time),
                                            mdates.date2num(segment['temp1'].index),
                                            segment['temp1'].values))
        
        # Calculate derivative and smooth it
        window = smoothed_window #60  # 1 hour window for smoothing
        dt = 1/60  # time step in hours
        dT_dt = np.gradient(temp_interp.values) / dt  # degrees per hour
        dT_dt_smooth = pd.Series(index=regular_time,
                                data=np.convolve(dT_dt, np.ones(window)/window, mode='same'))
        
        # Split cooling rate into three periods
        mask_before_delay = regular_time < segment['delay_time']
        mask_main = (regular_time >= segment['delay_time']) & (regular_time < segment['sunrise_time'])
        mask_after_sunrise = regular_time >= segment['sunrise_time']
        
        # Plot scaled cooling rate in three parts
        scaled_cooling = A*(-dT_dt_smooth) + b
        ax2.plot(regular_time[mask_before_delay], scaled_cooling[mask_before_delay], '--m', alpha=0.7,
                label='Cooling Rate -dT$_{i}$/dt (before delay)')
        ax2.plot(regular_time[mask_main], scaled_cooling[mask_main], '-m', alpha=0.7,
                label='Cooling Rate -dT$_{i}$/dt (main)')
        ax2.plot(regular_time[mask_after_sunrise], scaled_cooling[mask_after_sunrise], '--m', alpha=0.7,
                label='Cooling Rate -dT$_{i}$/dt (after sunrise)')
        
        # Rest of the plotting code remains the same...
        # [Previous axis formatting, limits, labels, etc.]

        # Find the ranges needed for both datasets
        diff_min = temp_diff.min()
        diff_max = temp_diff.max()
        
        # Set y1 limits to include all temperature difference data
        y1_min = min(0, diff_min)
        y1_max = max(25, diff_max)
        ax2.set_ylim(y1_min, y1_max)
        
        # Calculate corresponding y2 limits using y2 = (y1 - b)/A
        y2_min = (y1_min - b)/A
        y2_max = (y1_max - b)/A
        ax2_rate.set_ylim(y2_min, y2_max)
        
        # Customize the plots
        ax1.set_title(f'Cooling Sequence {i+1}\n'
                    f'Start: {segment["start_time"]}\n'
                    f'End: {segment["end_time"]}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Temperature (°C)')
        ax1_f.set_ylabel('Temperature (°F)')
        
        # Set up Fahrenheit axis limits
        c_min, c_max = ax1.get_ylim()
        ax1_f.set_ylim((c_min * 9/5 + 32), (c_max * 9/5 + 32))
        
        # Customize the difference/rate plot
        ax2.set_title(f'Temperature Difference and Scaled Cooling Rate\n'
                    f'(A = {A:.3f}, b = {b:.3f})')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Temperature Difference T$_{i}$ - T$_{o}$ (°C)')
        ax2_rate.set_ylabel('Cooling Rate -dT$_{i}$/dt (°C/hour)', color='m')
        
        # Set up grid for both plots
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
            ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:00'))
            ax.grid(True, which='major', axis='x', linestyle='-', color='gray', alpha=0.5)
            ax.grid(True, which='minor', axis='x', linestyle='-', color='gray', alpha=0.2)
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Color cooling rate ticks magenta
        ax2_rate.tick_params(axis='y', colors='m')
        
        # Add legends
        ax1.legend(loc='center left')
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines2, labels2, loc='upper right')

    # Add suptitle to the figure
    plt.suptitle(sup_title, fontsize=16, y=1.0)

    plt.tight_layout()
    plt.show()