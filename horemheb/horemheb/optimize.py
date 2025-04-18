import pandas as pd
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def get_time_temps(seg: dict):
    """Convert timestamps to hours from start
    Args:
        seg: (dict) cooling segment
    Returns:
        time_hours1: (np.array) time in hours
        temp1: (np.array) temperature 1
        time_hours2: (np.array) time in hours
        temp2: (np.array) temperature 2
    """

    delay_time = pd.Timestamp(seg['delay_time'])
    sunrise = pd.Timestamp(seg['sunrise_time'])
    mask1 = (seg['temp1'].index >= delay_time) & (seg['temp1'].index <= sunrise)

    def _one_and_two(temp_name: str):
        time_series = seg[temp_name][mask1]
        start_time = time_series.index[0]
        time_hours = [(t - start_time).total_seconds() / 60 /60 for t in time_series.index]
        temperatures = time_series.values
        return np.array(time_hours), np.array(temperatures)
    
    time_hours1, temp1 = _one_and_two("temp1")
    time_hours2, temp2 = _one_and_two("temp2")

    return time_hours1, temp1, time_hours2, temp2

def cooling_system(t, y, k1, k2, k3, k4, To_func):
    """
    System of ODEs:
    Ta' = k1*(Tw - Ta) + k2*(To - Ta)
    Tw' = k3*(To - Tw) + k4*(Ta - Tw)
    
    where
    Ta: temperature of air inside the house
    Tw: temperature of walls of the house
    To: temperature of outside air

    the k# coupling constants are in units of 1/hours
    and describe:

    k1: How fast the walls heat the inside air
    k2: How fast the outside air cools the inside air.
        This is due to leakage or porosity of the house.
    k3: How fast the walls are cooled by the outside air.
        The main cooling mechanism.
    k4: How fast the inside air heats the wall.
        This term is negligible (k4 = 0), since the heat capacity of the 
        walls is much larger than the inside air, but is included
        for completeness.

    Args:
        t: time point
        y: array of [Ta, Tw]
        k1, k2, k3, k4: coupling constants
        To_func: function that returns To at any time t
    """
    Ta, Tw = y
    To = To_func(t)
    
    dTa_dt = k1*(Tw - Ta) + k2*(To - Ta)
    dTw_dt = k3*(To - Tw) + k4*(Ta - Tw)
    
    return [dTa_dt, dTw_dt]

def precompute_data(tm1: np.array, 
                    t1: np.array,
                    tm2: np.array,                    
                    t2: np.array,
                    smoothing_factor1: float = 0.001,
                    smoothing_factor2: float = 0.01) -> tuple:
    """ 
    Precompute the data for the cooling system

    Args:
        tm1: (np.array) time in hours
        tm2: (np.array) time in hours
        t1: (np.array) temperature 1
        t2: (np.array) temperature 2
    """
    mask1 = ~np.isnan(t1)
    t1_interp = UnivariateSpline(tm1[mask1], t1[mask1], s=len(tm1)*smoothing_factor1)
    mask2 = ~np.isnan(t2)
    t2_interp = UnivariateSpline(tm2[mask2], t2[mask2], s=len(tm2)*smoothing_factor2)

    Ta0, Tw0 = t1_interp(0), t1_interp(0)  # initial conditions

    return (tm1, t1, t1_interp, t2_interp, Ta0, Tw0)

def solve_cooling_system(precomputed_data: tuple,
                         k_values: np.array = np.array([10.*0.5/24., 1.4/24., 0.5/24., 0.]),
                         ) -> scipy.integrate._ivp.ivp.OdeSolution:
    """
    Solve the cooling system using solve_ivp
    """
    tm1, t1, t1_interp, t2_interp, Ta0, Tw0 = precomputed_data

    t_span = (tm1[0], tm1[-1])
    #t_eval = np.linspace(0, tm1[-1], 1000)  # points at which to evaluate solution

    To_func = t2_interp

    k1, k2, k3, k4 = k_values

    # Solve the system
    solution = solve_ivp(
        cooling_system, 
        t_span, 
        [Ta0, Tw0], 
        args=(k1, k2, k3, k4, To_func),
        t_eval=tm1, #t_eval,
        method='RK45'  # Runge-Kutta 4(5) method
    )

    return solution


def plot_cooling_system_solution(solution: scipy.integrate._ivp.ivp.OdeSolution,
                        precomputed_data: tuple,
                        plot: bool = False) -> scipy.integrate._ivp.ivp.OdeSolution:
    """
    Plot the cooling system
    """
    tm1, t1, _, t2_interp, _, _ = precomputed_data

    plt.figure(figsize=(10, 6))
    plt.plot(solution.t, solution.y[0], 'b-', label='Ta(t) optimized')
    plt.plot(solution.t, solution.y[1], 'r-', label='Tw(t) optimized')
    plt.plot(tm1, t2_interp(tm1), 'g--', label='To(t)')
    plt.plot(tm1, t1, 'b.', alpha=0.3, label='Raw Ta data')
    plt.title('Cooling Segment')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid(True)
    plt.show()

    return solution


def objective(params, segments, precomputed_data=None, fixed_params=None):
    """
    Objective function that evaluates the fit across multiple segments
    params: list of parameters to optimize
    fixed_params: dict of fixed parameters {k1: value, k2: value, etc.}
    """
    # Combine fixed and optimized parameters
    k1, k2, k3, k4 = 0.0, 0.0, 0.0, 0.0  # default values
    if fixed_params:
        for key, value in fixed_params.items():
            if key == 'k1': k1 = value
            elif key == 'k2': k2 = value
            elif key == 'k3': k3 = value
            elif key == 'k4': k4 = value
    
    # Fill in optimized parameters
    param_idx = 0
    for i in range(4):
        param_name = f'k{i+1}'
        if not fixed_params or param_name not in fixed_params:
            if param_name == 'k1': k1 = params[param_idx]
            elif param_name == 'k2': k2 = params[param_idx]
            elif param_name == 'k3': k3 = params[param_idx]
            elif param_name == 'k4': k4 = params[param_idx]
            param_idx += 1
    
    total_error = 0
    total_points = 0
    
    for i, segment in enumerate(segments):
        if precomputed_data is None:
            precomputed_data_segment = precompute_data(*get_time_temps(segment))
        else:
            precomputed_data_segment = precomputed_data[i]

        solution = solve_cooling_system(precomputed_data_segment,
                                        k_values=(k1, k2, k3, k4))
        
        tm1, t1, _, _, _, _ = precomputed_data_segment

        segment_error = np.mean((solution.y[0] - t1)**2)
        total_error += segment_error * len(tm1)
        total_points += len(tm1)
    
    return total_error / total_points


def optimize_cooling_system(segments_to_optimize, sup_title, fixed_params: dict = None, verbose: bool = True):
    """
    Optimize cooling system parameters
    fixed_params: dict of fixed parameters {k1: value, k2: value, etc.}
    
    """
    # Precompute data for all segments
    if verbose:
        print("Precomputing data for segments...")
    precomputed_data = []
    for segment in segments_to_optimize:
        #tm1, t1, tm2, t2 = get_time_temps(segment)
        #mask1 = ~np.isnan(t1)
        #t1_interp = UnivariateSpline(tm1[mask1], t1[mask1], s=len(tm1)*0.001)
        #mask2 = ~np.isnan(t2)
        #t2_interp = UnivariateSpline(tm2[mask2], t2[mask2], s=len(tm2)*0.01)
        #Ta0, Tw0 = t1_interp(0), t1_interp(0)
        precomputed_data.append(precompute_data(*get_time_temps(segment)))

    # Determine which parameters to optimize
    n_params = 4 - (len(fixed_params) if fixed_params else 0)
    if n_params == 0:
        print("No parameters to optimize!")
        return

    # Initial guess for parameters (scaled to be closer to 1)
    #initial_guess = [10.*0.5/24., 1.4/24., 0.5/24., 0.0][:n_params]

    # Define bounds for parameters
    bounds = [(0, 1.0)] * n_params

    # Run multiple optimizations with different starting points
    if verbose:
        print("Starting optimization with multiple initial points...")
    n_trials = 5
    best_result = None
    best_error = float('inf')

    for i in range(n_trials):
        # Generate random initial guess within bounds
        x0 = np.random.uniform(0, 1.0, n_params)
        
        result = minimize(
            objective,
            x0,
            args=(segments_to_optimize, precomputed_data, fixed_params),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        if result.fun < best_error:
            best_error = result.fun
            best_result = result

    result = best_result

    # Get optimized parameters
    optimized_params = result.x
    param_idx = 0
    k1, k2, k3, k4 = 0.0, 0.0, 0.0, 0.0
    
    # Combine fixed and optimized parameters
    if fixed_params:
        for key, value in fixed_params.items():
            if key == 'k1': k1 = value
            elif key == 'k2': k2 = value
            elif key == 'k3': k3 = value
            elif key == 'k4': k4 = value
    
    # Fill in optimized parameters
    for i in range(4):
        param_name = f'k{i+1}'
        if not fixed_params or param_name not in fixed_params:
            if param_name == 'k1': k1 = optimized_params[param_idx]
            elif param_name == 'k2': k2 = optimized_params[param_idx]
            elif param_name == 'k3': k3 = optimized_params[param_idx]
            elif param_name == 'k4': k4 = optimized_params[param_idx]
            param_idx += 1

    # Calculate confidence intervals using the Hessian matrix
    if verbose:
        print("Calculating confidence intervals...")
    hessian = result.hess_inv.todense()
    covariance = np.linalg.inv(hessian)
    std_errors = np.sqrt(np.diag(covariance))
    confidence_intervals = 1.96 * std_errors

    result_dict = {}

    if verbose:
        print("\n---" + sup_title + "---")
        print(f"\nParameters:")
    param_idx = 0
    for i in range(4):
        param_name = f'k{i+1}'
        if fixed_params and param_name in fixed_params:
            if verbose:
                print(f"{param_name} = {fixed_params[param_name]:.6f} (fixed)")
            result_dict[param_name] = fixed_params[param_name]
        else:
            if verbose:
                print(f"{param_name} = {optimized_params[param_idx]:.6f} ± {confidence_intervals[param_idx]:.6f}")
            result_dict[param_name] = float(optimized_params[param_idx])
            param_idx += 1

    # Calculate R-squared value
    total_error = objective(result.x, segments_to_optimize, precomputed_data, fixed_params)
    total_variance = np.var(np.concatenate([seg['temp1'].values for seg in segments_to_optimize]))
    r_squared = 1 - (total_error / total_variance)

    if verbose:
        print(f"\nGoodness of fit:")
        print(f"R-squared = {r_squared:.4f}")

    result_dict['R-squared'] = float(r_squared)

    return result_dict


def plot_parametric_curves(segments, result_dict, sup_title=""):
    """
    Plot parametric curves of temperature difference vs rate of change for multiple segments.
    
    Args:
        segments (list): List of segments to analyze
        result_dict (dict): Dictionary containing optimization results, must include 'k3'
        precomputed_data (list): List of tuples containing (tm1, t1, t1_interp, t2_interp, Ta0, Tw0)
        sup_title (str, optional): Title for the plot. Defaults to empty string.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import solve_ivp
    
    # Extract parameters from result_dict
    k1 = result_dict['k1']
    k2 = result_dict['k2']
    k3 = result_dict['k3']
    k4 = result_dict['k4']
    
    # Generate colors for different segments
    colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))

    plt.figure()
    
    for i, segment in enumerate(segments):
        tm1, t1, t1_interp, t2_interp, Ta0, Tw0 = precompute_data(*get_time_temps(segment))
        
        solution_opt = solve_ivp(
            cooling_system,
            (tm1[0], tm1[-1]),
            [Ta0, Tw0],
            args=(k1, k2, k3, k4, t2_interp),
            t_eval=tm1,
            method='RK45'
        )
        
        # Calculate derivative of Ta(t)
        dt = tm1[1] - tm1[0]  # time step
        dTa_dt = np.gradient(solution_opt.y[0], dt)
        
        # Calculate temperature difference
        temp_diff = solution_opt.y[0] - t2_interp(tm1)
        
        # Plot parametric curve
        plt.plot(-dTa_dt[1:], temp_diff[1:], '-', color=colors[i], 
                label=f'Segment {i+1}',
                alpha=0.7)

    # Plot the theoretical line y = A0*x
    x = np.linspace(0.0*np.min(-dTa_dt[1:]), 1.0*np.max(-dTa_dt[1:]), 100)
    A0 = 1./result_dict['k3']
    plt.plot(x, A0*x, '--', color='black', alpha=0.4, label=f'{A0:.3f}*x')
    
    # Set plot properties
    plt.xlim(0, None)
    plt.ylim(0, None)
    plt.xlabel('-dT$_{i}$/dt (°C/hour)')
    plt.ylabel('T$_{i}$ - T$_{o}$ (°C)')
    plt.title(sup_title + ':\n Parametric Curves: Temperature Difference vs Rate of Change')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_combined_parametric_curves(segments1, result_dict1, sup_title1,
                                  segments2, result_dict2, sup_title2):
    """
    Plot parametric curves of temperature difference vs rate of change for two sets of segments.
    
    Args:
        segments1 (list): First list of segments to analyze
        result_dict1 (dict): Dictionary containing optimization results for first set
        sup_title1 (str): Title for first set of segments
        segments2 (list): Second list of segments to analyze
        result_dict2 (dict): Dictionary containing optimization results for second set
        sup_title2 (str): Title for second set of segments
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import solve_ivp
    
    plt.figure(figsize=(12, 8))
    
    handles1, handles2 = [], []
    labels1, labels2 = [], []
    
    # Process each set of segments
    for segments, results, sup_title, linestyle, linewidth, handles, labels in [
        (segments1, result_dict1, sup_title1, '-', 2, handles1, labels1),
        (segments2, result_dict2, sup_title2, '--', 3, handles2, labels2)
    ]:
        # Extract parameters from result_dict
        k1 = results['k1']
        k2 = results['k2']
        k3 = results['k3']
        k4 = results['k4']
        
        # Generate colors for different segments
        colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
        
        #dTa_dt_all = []
        dTa_dt_all_min = 0.0
        
        for i, segment in enumerate(segments):
            tm1, t1, t1_interp, t2_interp, Ta0, Tw0 = precompute_data(*get_time_temps(segment))
            
            solution_opt = solve_ivp(
                cooling_system,
                (tm1[0], tm1[-1]),
                [Ta0, Tw0],
                args=(k1, k2, k3, k4, t2_interp),
                t_eval=tm1,
                method='RK45'
            )
            
            # Calculate derivative of Ta(t)
            dt = tm1[1] - tm1[0]  # time step
            dTa_dt = np.gradient(solution_opt.y[0], dt)
            #dTa_dt_all.extend(dTa_dt[1:])
            dTa_dt_all_min = max(dTa_dt_all_min, np.min(-dTa_dt[1:]))
            
            # Calculate temperature difference
            temp_diff = solution_opt.y[0] - t2_interp(tm1)
            
            # Plot parametric curve
            line, = plt.plot(-dTa_dt[1:], temp_diff[1:], 
                           linestyle=linestyle, 
                           linewidth=linewidth,
                           color=colors[i],
                           label=f'Segment {i+1}',
                           alpha=0.7)
            
            handles.append(line)
            labels.append(f'Segment {i+1}')
        
        # Plot the theoretical line y = A0*x
        #x = np.linspace(0.0*np.min(-np.array(dTa_dt_all)), 
        #               1.0*np.max(-np.array(dTa_dt_all)), 100)
        x = np.linspace(0.0, dTa_dt_all_min, 100)
        A0 = 1./results['k3']
        theory_line, = plt.plot(x, A0*x, 
                              linestyle=linestyle, 
                              linewidth=linewidth,
                              color='black',
                              alpha=0.3,
                              label=f'{A0:.3f}*x')
        
        handles.append(theory_line)
        labels.append(f'{A0:.3f}*x, R² = {results["R-squared"]:.3f}')
    
    # Set plot properties
    plt.xlim(0, None)
    plt.ylim(0, None)
    plt.xlabel('-dT$_{i}$/dt (°C/hour)')
    plt.ylabel('T$_{i}$ - T$_{o}$ (°C)')
    plt.title(f'Parametric Curves: Temperature Difference vs Rate of Change\n{sup_title1} vs {sup_title2}')
    plt.grid(True, alpha=0.3)
    
    # Create separate legends for each set
    ax1 = plt.gca()
    ax1.add_artist(ax1.legend(handles1, labels1, loc='upper left', title=sup_title1))
    ax1.add_artist(ax1.legend(handles2, labels2, loc='lower right', title=sup_title2))
    
    plt.tight_layout()
    plt.show()


def plot_optimize_k3_over_k1_k2(segments, sup_title, 
                               k1_values=None, k2_values=None,
                               mark_best=True):
    """
    Perform grid search optimization over k1 and k2 values, plotting k3 and R-squared results.
    
    Args:
        segments (list): List of segments to analyze
        sup_title (str): Title for the plots
        k1_values (list, optional): List of k1 values to try. Defaults to [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        k2_values (list, optional): List of k2 values to try. Defaults to [0.1, 0.09, 0.08, 0.07, 0.06, 0.05]
        mark_best (bool, optional): Whether to mark the best value point on plots. Defaults to True
    
    Returns:
        dict: Dictionary containing the best parameters and their R-squared value
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Use default values if none provided
    if k1_values is None:
        k1_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    if k2_values is None:
        k2_values = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05]
    
    # Initialize results list and tracking variables
    results = []
    max_r_squared = 0.0
    best_params = None
    
    # Run optimization for each combination
    print("Running grid search...")
    for k1 in k1_values:
        print('.', end='')
        row_results = []
        for k2 in k2_values:
            fixed_params = {'k1': k1, 'k2': k2, 'k4': 0.0}
            result = optimize_cooling_system(segments, sup_title, fixed_params=fixed_params, verbose=False)
            row_results.append(result)
            if result['R-squared'] > max_r_squared:
                max_r_squared = result['R-squared']
                best_params = fixed_params
        results.append(row_results)
    print('')

    # Extract k3 and R-squared values into grids
    k3_grid = np.array([[result['k3'] for result in row] for row in results])
    r_squared_grid = np.array([[result['R-squared'] for result in row] for row in results])
    
    # Find the best combination
    best_idx = np.unravel_index(np.argmax(r_squared_grid), r_squared_grid.shape)
    best_k1 = k1_values[best_idx[0]]
    best_k2 = k2_values[best_idx[1]]
    best_k3 = k3_grid[best_idx]
    best_r_squared = r_squared_grid[best_idx]
    
    # Create contour plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create meshgrid for plotting
    k1_mesh, k2_mesh = np.meshgrid(k1_values, k2_values, indexing='ij')
    
    # Plot k3 contour
    contour1 = ax1.contourf(k1_mesh, k2_mesh, k3_grid, levels=25)
    ax1.set_xlabel('K$_{1}$')
    ax1.set_ylabel('K$_{2}$')
    ax1.set_title('Optimized K$_{3}$ values')
    plt.colorbar(contour1, ax=ax1, label='K$_{3}$')
    
    # Add best point marker and label for k3 plot if requested
    if mark_best:
        ax1.plot(best_k1, best_k2, 'w*', markersize=10, label=f'Best: K$_{3}$={best_k3:.4f}, 1/K$_{3}$={1./best_k3:.3f} hr')
        #ax1.annotate(f'k3={best_k3:.4f}', 
        #            xy=(best_k1, best_k2), 
        #            xytext=(10, 10), 
        #            textcoords='offset points',
        #            color='white',
        #            bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
        ax1.legend()
    
    # Plot R-squared contour
    contour2 = ax2.contourf(k1_mesh, k2_mesh, r_squared_grid, levels=25)
    ax2.set_xlabel('K$_{1}$')
    ax2.set_ylabel('K$_{2}$')
    ax2.set_title('R$^{2}$ values')
    plt.colorbar(contour2, ax=ax2, label='R$^{2}$')
    
    # Add best point marker and label for R-squared plot if requested
    if mark_best:
        ax2.plot(best_k1, best_k2, 'w*', markersize=10, label=f'Best: R²={best_r_squared:.3f}')
        #ax2.annotate(f'R²={best_r_squared:.3f}', 
        #            xy=(best_k1, best_k2), 
        #            xytext=(10, 10), 
        #            textcoords='offset points',
        #            color='white',
        #            bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
        ax2.legend()
    

    
    # Print the results
    print(f"\n{sup_title} Best parameter combination:")
    print(f"K$_{1}$ = {best_k1:.6f}")
    print(f"K$_{2}$ = {best_k2:.6f}")
    print(f"K$_{3}$ = {best_k3:.6f}")
    print(f"R$^{2}$ = {best_r_squared:.6f}")
    
    print(f"\n{sup_title} Best parameters: {best_params}")

    plt.suptitle(f'{sup_title}: Parameter Optimization Results')
    plt.tight_layout()
    plt.show()

    return {
        'k1': best_k1,
        'k2': best_k2,
        'k3': best_k3,
        'k4': 0.0,
        'R-squared': best_r_squared
    }


if __name__ == "__main__":
    from horemheb.config import AnalysisConfig
    from horemheb.loader import load_temperature_data
    from horemheb.segments import process_segments
    from horemheb.paths import data_dir

    # Configure analysis parameters
    config = AnalysisConfig(
        delay_time=0, #80, #300, #240, #80, #240, #180, #120,
        before_sunrise_delta_minutes=0,
        resample_minutes=5
    )

    temperature_data_file_old_door = 'temperature_logB_12.csv'
    temperature_data_file = 'temperature_logC_2025-03-12.csv'

    # Filter New Door data after the specified timestamp
    cutoff_time = '2025-02-27 19:48:04'

    # Old Door
    dfo, dfo_r1, dfo_r2 = load_temperature_data(
        data_dir / temperature_data_file,
        config,
        cutoff_time=cutoff_time,
        comparison='<='
    )

    segments_old = process_segments(dfo_r1, dfo_r2, config)

    #optimize_cooling_system(segments_old, "Old Door")

    results = optimize_cooling_system(segments_old, "Old Door", fixed_params={'k1': 0.7,'k2': 0.07, 'k4': 0.0}) #'k1': 0.7, 'k2': 0.07, 'k4': 0.0})

    #print(results)

    #plot_parametric_curves(segments_old, results, "Old Door")

    grid_best_result = plot_optimize_k3_over_k1_k2(segments_old, "Old Door")

'''
def plot_all_segments_with_solution(segments, result_dict, sup_title, config=None, smoothed_window=60):
    """
    Plot all segments with both raw data and ODE solution overlaid.
    
    Args:
        segments (list): List of segments to analyze
        results: Results object containing A and b values
        result_dict (dict): Dictionary containing k1, k2, k3, k4 values
        sup_title (str): Super title for the plot
        config (optional): Configuration object containing resample_minutes
        smoothed_window (int, optional): Window size for smoothing. Defaults to 60
    """
    A, b = 1./result_dict['k3'], 0.0
    if config is not None:
        suffix = f" ({config.resample_minutes}min avg)"
    else:
        suffix = ""

    fig, axes = plt.subplots(len(segments), 2, figsize=(15, 6*len(segments)))
    if len(segments) == 1:
        axes = [axes]

    # Extract k parameters
    k1 = result_dict['k1']
    k2 = result_dict['k2']
    k3 = result_dict['k3']
    k4 = result_dict['k4']

    for i, (ax1, ax2) in enumerate(axes):
        segment = segments[i]
            
        # Create secondary y-axes
        ax1_f = ax1.twinx()  # Fahrenheit axis
        ax2_rate = ax2.twinx()  # Cooling rate axis
        
        # Plot original temperature data
        ax1.plot(segment['temp1'].index, segment['temp1'].values, '.r', 
                alpha=0.3, label='Sensor 1 (raw)' + suffix)
        ax1.plot(segment['temp2'].index, segment['temp2'].values, '.b', 
                alpha=0.3, label='Sensor 2 (raw)' + suffix)
        
        # Get ODE solution
        precomputed = precompute_data(*get_time_temps(segment))
        tm1, t1, t1_interp, t2_interp, Ta0, Tw0 = precomputed
        
        solution = solve_ivp(
            cooling_system,
            (tm1[0], tm1[-1]),
            [Ta0, Tw0],
            args=(k1, k2, k3, k4, t2_interp),
            t_eval=tm1,
            method='RK45'
        )
        
        # Convert solution time to datetime
        start_time = segment['temp1'].index[0]
        solution_times = [start_time + pd.Timedelta(hours=t) for t in solution.t]
        
        # Plot ODE solutions
        ax1.plot(solution_times, solution.y[0], '-r', 
                linewidth=2, label='Ta(t) solution')
        ax1.plot(solution_times, solution.y[1], '-g', 
                linewidth=2, label='Tw(t) solution')
        ax1.plot(solution_times, t2_interp(solution.t), '--b', 
                linewidth=2, label='To(t) interpolated')
        
        # Calculate temperature difference
        temp_diff = segment['temp1'] - segment['temp2']
        temp_diff_solution = solution.y[0] - t2_interp(solution.t)
        
        # Split temp_diff into three periods
        mask_before_delay = temp_diff.index < segment['delay_time']
        mask_main = (temp_diff.index >= segment['delay_time']) & (temp_diff.index < segment['sunrise_time'])
        mask_after_sunrise = temp_diff.index >= segment['sunrise_time']
        
        # Plot temperature difference data
        ax2.plot(temp_diff[mask_before_delay].index, temp_diff[mask_before_delay].values, '.k', 
                alpha=0.3, label='T1 - T2 (before delay)')
        ax2.plot(temp_diff[mask_main].index, temp_diff[mask_main].values, '.k', 
                alpha=0.3, label='T1 - T2 (main)')
        ax2.plot(temp_diff[mask_after_sunrise].index, temp_diff[mask_after_sunrise].values, '.k', 
                alpha=0.3, label='T1 - T2 (after sunrise)')
        
        # Plot solution temperature difference
        ax2.plot(solution_times, temp_diff_solution, '-k', 
                linewidth=2, label='Ta - To solution')
        
        # Calculate and plot cooling rate
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
        
        # Split cooling rate into periods
        mask_before_delay = regular_time < segment['delay_time']
        mask_main = (regular_time >= segment['delay_time']) & (regular_time < segment['sunrise_time'])
        mask_after_sunrise = regular_time >= segment['sunrise_time']
        
        # Plot scaled cooling rate
        scaled_cooling = A*(-dT_dt_smooth) + b
        ax2.plot(regular_time[mask_before_delay], scaled_cooling[mask_before_delay], '--r', alpha=0.7,
                label='Scaled Cooling Rate (before delay)')
        ax2.plot(regular_time[mask_main], scaled_cooling[mask_main], '-r', alpha=0.7,
                label='Scaled Cooling Rate (main)')
        ax2.plot(regular_time[mask_after_sunrise], scaled_cooling[mask_after_sunrise], '--r', alpha=0.7,
                label='Scaled Cooling Rate (after sunrise)')
        
        # Set axis limits and labels
        diff_min = min(temp_diff.min(), temp_diff_solution.min())
        diff_max = max(temp_diff.max(), temp_diff_solution.max())
        
        y1_min = min(0, diff_min)
        y1_max = max(22, diff_max)
        ax2.set_ylim(y1_min, y1_max)
        
        y2_min = (y1_min - b)/A
        y2_max = (y1_max - b)/A
        ax2_rate.set_ylim(y2_min, y2_max)
        
        # Customize plots
        ax1.set_title(f'Cooling Sequence {i+1}\n'
                     f'Start: {segment["start_time"]}\n'
                     f'End: {segment["end_time"]}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Temperature (°C)')
        ax1_f.set_ylabel('Temperature (°F)')
        
        c_min, c_max = ax1.get_ylim()
        ax1_f.set_ylim((c_min * 9/5 + 32), (c_max * 9/5 + 32))
        
        ax2.set_title(f'Temperature Difference and Scaled Cooling Rate\n'
                     f'(A = {A:.3f}, b = {b:.3f})')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Temperature Difference (°C)')
        ax2_rate.set_ylabel('Cooling Rate (°C/hour)', color='r')
        
        # Set up grid
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
            ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:00'))
            ax.grid(True, which='major', axis='x', linestyle='-', color='gray', alpha=0.5)
            ax.grid(True, which='minor', axis='x', linestyle='-', color='gray', alpha=0.2)
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        ax2_rate.tick_params(axis='y', colors='r')
        
        # Add legends
        ax1.legend(loc='upper left')
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines2, labels2, loc='upper right')

    plt.suptitle(sup_title, fontsize=16, y=1.0)
    plt.tight_layout()
    plt.show()
'''

def plot_all_segments_with_solution(segments, result_dict, sup_title, config=None, smoothed_window=60):
    """
    Plot all segments with both raw data and ODE solution overlaid.
    
    Args:
        segments (list): List of segments to analyze
        results: Results object containing A and b values
        result_dict (dict): Dictionary containing k1, k2, k3, k4 values
        sup_title (str): Super title for the plot
        config (optional): Configuration object containing resample_minutes
        smoothed_window (int, optional): Window size for smoothing. Defaults to 60
    """
    A, b = 1./result_dict['k3'], 0.0
    if config is not None:
        suffix = f" ({config.resample_minutes}min avg)"
    else:
        suffix = ""

    fig, axes = plt.subplots(len(segments), 2, figsize=(15, 6*len(segments)))
    if len(segments) == 1:
        axes = [axes]

    # Extract k parameters
    k1 = result_dict['k1']
    k2 = result_dict['k2']
    k3 = result_dict['k3']
    k4 = result_dict['k4']

    for i, (ax1, ax2) in enumerate(axes):
        segment = segments[i]
            
        # Create secondary y-axes
        ax1_f = ax1.twinx()  # Fahrenheit axis
        ax2_rate = ax2.twinx()  # Cooling rate axis
        
        # Plot original temperature data
        ax1.plot(segment['temp1'].index, segment['temp1'].values, '.r', 
                alpha=0.3, label='Sensor 1 T$_{i}$ (raw)' + suffix)
        ax1.plot(segment['temp2'].index, segment['temp2'].values, '.b', 
                alpha=0.3, label='Sensor 2 T$_{o}$ (raw)' + suffix)
        
        # Get ODE solution
        precomputed = precompute_data(*get_time_temps(segment))
        tm1, t1, t1_interp, t2_interp, Ta0, Tw0 = precomputed
        
        solution = solve_ivp(
            cooling_system,
            (tm1[0], tm1[-1]),
            [Ta0, Tw0],
            args=(k1, k2, k3, k4, t2_interp),
            t_eval=tm1,
            method='RK45'
        )
        
        # Convert solution time to datetime
        start_time = segment['temp1'].index[0]
        solution_times = [start_time + pd.Timedelta(hours=t) for t in solution.t]
        
        # Plot ODE solutions
        ax1.plot(solution_times, solution.y[0], '-r', 
                linewidth=2, label='T$_{i}$(t) solution')
        ax1.plot(solution_times, solution.y[1], '-g', 
                linewidth=2, label='T$_{w}$(t) solution')
        ax1.plot(solution_times, t2_interp(solution.t), '--b', 
                linewidth=2, label='T$_{o}$ interpolated')
        
        # Calculate temperature difference
        temp_diff = segment['temp1'] - segment['temp2']
        temp_diff_solution = solution.y[0] - t2_interp(solution.t)
        
        # Split temp_diff into three periods
        mask_before_delay = temp_diff.index < segment['delay_time']
        mask_main = (temp_diff.index >= segment['delay_time']) & (temp_diff.index < segment['sunrise_time'])
        mask_after_sunrise = temp_diff.index >= segment['sunrise_time']
        
        # Plot temperature difference data
        ax2.plot(temp_diff[mask_before_delay].index, temp_diff[mask_before_delay].values, '.k', 
                alpha=0.3, label='T$_{i}$ - T$_{o}$ (before delay)')
        ax2.plot(temp_diff[mask_main].index, temp_diff[mask_main].values, '.k', 
                alpha=0.3, label='T$_{i}$ - T$_{o}$ (main)')
        ax2.plot(temp_diff[mask_after_sunrise].index, temp_diff[mask_after_sunrise].values, '.k', 
                alpha=0.3, label='T$_{i}$ - T$_{o}$ (after sunrise)')
        
        # Plot solution temperature difference
        ax2.plot(solution_times, temp_diff_solution, '-k', 
                linewidth=2, label='T$_{i}$ - T$_{o}$ solution')
        
        # Calculate and plot cooling rate from data
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
        
        # Calculate derivative of ODE solution Ta(t)
        dt_solution = solution.t[1] - solution.t[0]  # time step in hours
        dTa_dt_solution = np.gradient(solution.y[0], dt_solution)  # degrees per hour
        
        # Split cooling rate into periods
        mask_before_delay = regular_time < segment['delay_time']
        mask_main = (regular_time >= segment['delay_time']) & (regular_time < segment['sunrise_time'])
        mask_after_sunrise = regular_time >= segment['sunrise_time']
        
        # Plot scaled cooling rates
        scaled_cooling = A*(-dT_dt_smooth) + b
        scaled_cooling_solution = A*(-dTa_dt_solution) + b
        
        # Plot data-based cooling rate
        ax2.plot(regular_time[mask_before_delay], scaled_cooling[mask_before_delay], '--m', alpha=0.7,
        ) #label='Cooling Rate dTa/dt (data, before delay)')
        ax2.plot(regular_time[mask_main], scaled_cooling[mask_main], '-m', alpha=0.7,
                label='Cooling Rate dT$_{i}$/dt (data, main)')
        ax2.plot(regular_time[mask_after_sunrise], scaled_cooling[mask_after_sunrise], '--m', alpha=0.7,
                label='Cooling Rate dT$_{i}$/dt (data, after sunrise)')
        
        # Plot solution-based cooling rate
        ax2.plot(solution_times, scaled_cooling_solution, '-m', alpha=0.4, linewidth=3,
                label='Cooling Rate dT$_{i}$/dt (solution)')
        
        # Set axis limits and labels
        diff_min = min(temp_diff.min(), temp_diff_solution.min())
        diff_max = max(temp_diff.max(), temp_diff_solution.max())
        
        y1_min = min(0, diff_min)
        y1_max = max(25, diff_max)
        ax2.set_ylim(y1_min, y1_max)
        
        y2_min = (y1_min - b)/A
        y2_max = (y1_max - b)/A
        ax2_rate.set_ylim(y2_min, y2_max)
        
        # Customize plots
        ax1.set_title(f'Cooling Sequence {i+1}\n'
                     f'Start: {segment["start_time"]}\n'
                     f'End: {segment["end_time"]}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Temperature (°C)')
        ax1_f.set_ylabel('Temperature (°F)')
        
        c_min, c_max = ax1.get_ylim()
        ax1_f.set_ylim((c_min * 9/5 + 32), (c_max * 9/5 + 32))
        
        ax2.set_title(f'Temperature Difference and Scaled Cooling Rate\n'
                     f'(A = {A:.3f}, b = {b:.3f})')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Temperature Difference (°C)')
        ax2_rate.set_ylabel('Cooling Rate (°C/hour)', color='m')
        
        # Set up grid
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
            ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:00'))
            ax.grid(True, which='major', axis='x', linestyle='-', color='gray', alpha=0.5)
            ax.grid(True, which='minor', axis='x', linestyle='-', color='gray', alpha=0.2)
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        ax2_rate.tick_params(axis='y', colors='m')
        
        # Add legends
        ax1.legend(loc='center left')
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines2, labels2, loc='upper right')

    plt.suptitle(sup_title, fontsize=16, y=1.0)
    plt.tight_layout()
    plt.show()