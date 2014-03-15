import nanosim
import numpy as np
from itertools import repeat, izip, chain
import scipy.ndimage
from scipy.interpolate import griddata
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
from sets import Set
from collections import namedtuple
from utilities import *
import numexpr as ne

smoothing_factor = 0.01 # Smoothing width (gaussian std. dev.) as a percentage of time-domain
m = 1000 # Number of points in the smoothed line
y_paddding_percentage = .3 # Percentage of blank space around the y-range of the data

plt = nanosim.plt

ncolours = 12
colourmap_positions = np.linspace(0, 1.0, int(np.ceil(ncolours * 1.5))) # Generate (1.5 * ncolours) evenly-spaced colourmap positions
colourmap_positions = np.array([p for i, p in enumerate(colourmap_positions) if ((i+1) % 3) == 0]) # Reduce to (ncolours) positions by dropping every third
# Drpping every third colour ensures that there are pairs of similar colours and that those pairs are distinct from other pairs
line_colours = list(plt.get_cmap('jet')(colourmap_positions))

image_colours = ('Blues', 'Reds', 'RdPu')

SweepData = namedtuple('SweepData', 'forward, backward')
ComponentData = namedtuple('ComponentData', 'means, time_means, vars_p, vars_m, label')

def voltage_sweep(inputs, v_ranges, steptime, all_probes, v_probes, i_probes, q_probes, V, I, Q):
    '''
        Recursivley (for each dimension) set voltage values, then simulate and grab the data.
    '''
    
    if len(inputs) == 0:
        # Clear logs
        nanosim.reset_logs()

        # Run sim
        nanosim.simulate(steptime, logging = all_probes)
        
        # Extract data from component logs
        for k, p in enumerate(v_probes):
            V[k] = p.voltages
        for k, p in enumerate(i_probes):
            I[k] = p.current
        for k, p in enumerate(q_probes):
            Q[k] = p.charges

    else:
        input_component = inputs[0]
        v_range = v_ranges[0]
        inputs = inputs[1:]
        v_ranges = v_ranges[1:]

        for j, v in enumerate(v_range):
            input_component.set_V(v)
            voltage_sweep(inputs, v_ranges, steptime, all_probes, v_probes, i_probes, q_probes, V[:, j], I[:, j], Q[:, j])

def vars_and_means(raw_data, steptime, probes, data_array_shape, flip = False):
    '''
        Calculate vars and means over both time and repetitions from raw sim data.
    '''
    if raw_data.size == 0: return [], [], [], []
    
    # The mean/variance values over time for each component, at each voltage level combination, and at each repetition
    time_means = np.zeros([len(probes)] + data_array_shape)
    time_vars_plus = np.zeros([len(probes)] + data_array_shape)
    time_vars_minus = np.zeros([len(probes)] + data_array_shape)
    # Index order for these arrays: component, v1, (v2), repetition

    time_vars = time_vars_plus, time_vars_minus

    # Fill array of means by averaging over time
    it = np.nditer(raw_data, flags=['multi_index', 'refs_ok'])
    while not it.finished:
        events = it[0][()] # Numpy's yields a 0-d array, not the actual object, so this strange getitem call is required to extract it
        times, values = zip(*events)
        
        # Weight each value by the period of time until the next measurement
        times += (steptime,) # End time
        times = np.diff(times)
        
        mean = ne.evaluate('sum(times * values)') / steptime
        time_means[it.multi_index] = mean # Assign mean to array

        residuals = values - mean # Residual at each event
        plus_indicies, minus_indicies = (residuals > 0), (residuals < 0)
        residuals = ne.evaluate('(times * (residuals ** 2)) / steptime') # Scaled (Note: square *before* scaling)
        residuals = residuals[plus_indicies], residuals[minus_indicies] # +, - residuals

        # Assign variance to arrays
        time_vars_plus[it.multi_index] = np.sum(residuals[0])
        time_vars_minus[it.multi_index] = np.sum(residuals[1])
        
        it.iternext()
    
    # The mean values over repetition for each component, at each voltage level combination
    means = np.mean(time_means, axis = -1)
    # Index order for these arrays: component, v1, (v2)

    # The variance values over repetition for each component, at each voltage level combination
    extended_means = means.reshape(means.shape + (1,)) # Extend with a single index in the repetitions-dimension - allows numpy to broadcast properly in the next step
    repetition_residuals = time_means - extended_means # Residual at each repetition
    
    # Splot into positive and negative residuals
    res_plus, res_minus = np.zeros(repetition_residuals.shape), np.zeros(repetition_residuals.shape)
    plus_indicies, minus_indicies = (repetition_residuals > 0), (repetition_residuals < 0)
    res_plus[plus_indicies] = repetition_residuals[plus_indicies]
    res_minus[minus_indicies] = repetition_residuals[minus_indicies]
    
    repetition_residuals = res_plus**2, res_minus**2
    
    # Add mean variance over time, to mean variance between repetitions, then take the sqrt to get the std. dev.
    variances_plus, variances_minus = [np.sqrt(np.mean(t_vars, axis = -1) + np.mean(rep_vars, axis = -1)) for t_vars, rep_vars in zip(time_vars, repetition_residuals)]
    
    # Index order for these arrays: component, v1, (v2)

    # Undo backwards-voltage sweep ordering
    if flip:
        if len(means.shape) == 2:  # For dim-1 plots
            means = means[:, ::-1]
            time_means = time_means[:, ::-1, :]
            variances_plus = variances_plus[:, ::-1]
            variances_minus = variances_minus[:, ::-1]
        elif len(means.shape) == 3:  # For dim-2 plots
            means = means[:, ::-1, ::-1]
            time_means = time_means[:, ::-1, ::-1, :]
            variances_plus = variances_plus[:, ::-1, ::-1]
            variances_minus = variances_minus[:, ::-1, ::-1]
            
            means = np.transpose(means, (0, 2, 1))
            time_means = np.transpose(time_means, (0, 2, 1, 3))
            variances_plus = np.transpose(variances_plus, (0, 2, 1))
            variances_minus = np.transpose(variances_minus, (0, 2, 1))

    return means, time_means, variances_plus, variances_minus

def presmooth_current(I, steptime):
    '''
        Calculate vars and means for currents over repetitions from raw sim data.

        Currents are different to voltages and charges, since they are all instantaneous events; measuring current variance over time has no meaning*.
        * Technically you could estimate variance in frequency of events over time, and probably do some Poisson distribution stuff to characterise the
            time-variance, but for now, I think it's probably fine the way it is.
    '''
    if I.size == 0: return [], []
    
    it = np.nditer(I, flags=['multi_index', 'refs_ok'])
    while not it.finished:
        events = it[0][()] # Numpy's yields a 0-d array, not the actual object, so this strange getitem call is required to extract it
        times, currents = zip(*events)
        
        times, currents = spread_signal_spikes(times, currents, x_range = (0, steptime), m = m)
        
        smoothing_window = int(m * 0.4) # Smoothing radius for softening varaiance computation
        times = smooth1D(times, k = smoothing_window)
        currents = smooth1D(currents, k = smoothing_window)
        
        I[it.multi_index] = zip(times, currents)
        it.iternext()

def characterise(runtime, inputs, input_ranges, steps = 200, v_probes = [], i_probes = [], q_probes = [], v_diffs = [], repetitions = 10, raw_data = True, mean_data = True, variance = True, show = True, fig = False, axs = False, colours = None):
    '''
    Characterise a circuit over two, independant input dimensions.
        runtime: time, in seconds that each simulation is to be run for
        inputs: list of reservoirs to voltage-sweep with
        input_ranges: a 2-tuple for each input that has min and max voltages to sweep over
        steps: number of voltage-steps
        v_probes: components to be monitored for voltage-levels
        i_probes: components to be monitored for current
        q_probes: components to be monitored for charge-levels
        v_diffs: pairs (tuples) of components whose relative voltage is to be monitored (the plotted value will be: pair[0].V - pair[1].V)
        variance: claculate and display variances in behaviour (for stability analysis)
        repetitions: run the sim this many times and average the results
        raw_data: plot the raw data-points (for dim-1 plots only)
        mean_data: plot the mean data-points (for dim-1 plots only)
        show: call pyplot.show() at the end. Us ethis if you want to modify the graph externally before showing.
        fig, axs: provide fig and axs objects for the plot to work with (if you want external control over the graph). Both must be provided if either is, and axs MUST have one subplot axis for each of voltage, current and charge if they are to be monitored.
        colours: change the default list of colours for plotting.

    inputs and input_ranges must be at most length two.
    At least one of v_probes, i_probes , q_probes or v_diffs should be filled or there will just be an empty plot.
    '''

    dim = len(inputs) # 2 control voltages, or 1? (only dim 1 and dim 2 are supported)
    steptime = runtime/float(steps)

    if not colours:
        if dim == 1:
            colours = line_colours
        elif dim == 2:
            colours = image_colours
    
    dont_plot = [] # Components that need to monitored for v_diff, but whose voltage shouldn't be independantly plotted.
    for pair in v_diffs:
        for c in pair:
            if c not in v_probes:
                dont_plot.append(c)
    v_probes += dont_plot # Add the not-for-plotting nodes to the monitoring-list

    dont_plot = Set(dont_plot) # For quick membership tests
    v_probe_indicies = dict(zip(v_probes, range(len(v_probes)))) # For quick-lookup of data when plotting v_diff pairs

    all_probes = v_probes + i_probes + q_probes
    all_probe_lists = (v_probes, i_probes, q_probes)
    
    if (fig is False) or (axs is False): # If fig and axs aren't provided as args
        ncolumns = 1
        if dim == 1:
            nplots = len(nonempty(v_probes, i_probes, q_probes))
        elif dim == 2:
            nplots = len(all_probes) - len(dont_plot)
            if variance: ncolumns = 2
        fig, axs = plt.subplots(nplots, ncolumns, figsize=(14, 10)) # Plot size

    
    # The data for each component will be stored in these lists in the same order as the components are listed in the v/i/q_probes arguments
    data_array_shape = [steps] * dim + [repetitions] # voltages (* voltages) * repetitions
    V = np.empty([len(v_probes)] + data_array_shape, dtype = 'object')
    I = np.empty([len(i_probes)] + data_array_shape, dtype = 'object')
    Q = np.empty([len(q_probes)] + data_array_shape, dtype = 'object')
    # Index order for these arrays: component, v1, (v2,) repetition
    # Each element of these arrays will be a list of logged time-value tuples
    
    all_data = (V, I, Q)


    print 'Calculating...',
    
    #  Main computation for forward sweep
    for j in xrange(repetitions):
        nanosim.reset_sim() # Set time to 0 and clear logs on all components

        v_ranges = [np.linspace(r[0], r[1], steps) for r in input_ranges]

        # Step over all voltage combinations, simulate, then log the results.
        voltage_sweep(inputs, v_ranges, steptime, all_probes, v_probes, i_probes, q_probes, V[..., j], I[..., j], Q[..., j])

        print '%.1f%%' % (50 * float(j+1)/repetitions)


    v_components, i_components, q_components = [], [], []
    all_component_lists = (v_components, i_components, q_components)

    presmooth_current(I, steptime)

    # Temporarily sort forward-data
    for data, probes, component_list in zip(all_data, all_probe_lists, all_component_lists):
        means, time_means, vars_p, vars_m = vars_and_means(data, steptime, probes, data_array_shape)
        for i, probe in enumerate(probes): component_list.append(ComponentData(means[i], time_means[i], vars_p[i], vars_m[i], probe.label))


    inputs.reverse() # Reverse component order
    input_ranges = [tuple(reversed(r)) for r in reversed(input_ranges)] # Reverse both range and component order

    #  Main computation for backward sweep
    for j in xrange(repetitions):
        nanosim.reset_sim() # Set time to 0 and clear logs on all components

        v_ranges = [np.linspace(r[0], r[1], steps) for r in input_ranges]

        # Step over all voltage combinations, simulate, then log the results.
        voltage_sweep(inputs, v_ranges, steptime, all_probes, v_probes, i_probes, q_probes, V[..., j], I[..., j], Q[..., j])

        print '%.1f%%' % (50 + 50 * float(j+1)/repetitions)

    # Undo reversal
    inputs.reverse()
    input_ranges = [tuple(reversed(r)) for r in reversed(input_ranges)]


    presmooth_current(I, steptime)

    for data, probes, component_list in zip(all_data, all_probe_lists, all_component_lists):           
        means, time_means, vars_p, vars_m = vars_and_means(data, steptime, probes, data_array_shape, flip = True)
        for i, probe in enumerate(probes):
            sweep_mean = SweepData(component_list[i].means, means[i])
            sweep_time_mean = SweepData(component_list[i].time_means, time_means[i])
            sweep_vars_p = SweepData(component_list[i].vars_p, vars_p[i])
            sweep_vars_m = SweepData(component_list[i].vars_m, vars_m[i])
            component_list[i] = ComponentData(sweep_mean, sweep_time_mean, sweep_vars_p, sweep_vars_m, probe.label)

    # Calculate voltage-difference data
    for pair in v_diffs:
        i, j = v_probe_indicies[pair[0]], v_probe_indicies[pair[1]]
        means = SweepData((v_components[i].means.forward - v_components[j].means.forward), (v_components[i].means.backward - v_components[j].means.backward))
        time_means = SweepData((v_components[i].time_means.forward - v_components[j].time_means.forward), (v_components[i].time_means.backward - v_components[j].time_means.backward))
        vars_p = SweepData((v_components[i].vars_p.forward - v_components[j].vars_p.forward), (v_components[i].vars_p.backward - v_components[j].vars_p.backward))
        vars_m = SweepData((v_components[i].vars_m.forward - v_components[j].vars_m.forward), (v_components[i].vars_m.backward - v_components[j].vars_m.backward))
        v_components.append(ComponentData(means, time_means, vars_p, vars_m, 'V[%s - %s]' % (pair[0].label, pair[1].label)))

    # Remove v_probes that shouldn't be plotted on their own (ie. that were only recorded for voltage-difference calculations)
    v_components = [data for i, data in enumerate(v_components[:len(v_probes)]) if v_probes[i] not in dont_plot] + v_components[len(v_probes):]
        

    # Now for the actual plotting. The logic is totally different for 1d or 2d data, so this big if-statement does the branching
    if dim == 1:
        v_range = input_ranges[0]
        X = np.linspace(v_range[0], v_range[1], steps) # Input voltage data
        smoothing_window = int(steps * smoothing_factor) # Smoothing width in array-indicies
        
        for ax, components, axlabel in zip(axs, all_component_lists, ('Voltage (V)', 'Current Avg. (e/s)', 'Charge (e)')): # For axis/plot
            # Labels
            ax.set_title('%s - Voltage Performance' % axlabel.split()[0])
            ax.set_xlabel('Probe Voltage (V)')
            ax.set_ylabel(axlabel)
            
            for component, colourpair in zip(components, pairwise(colours)): # For each component
                for means, time_means, vars_p, vars_m, colour, direction in zip(component.means, component.time_means, component.vars_p, component.vars_m, colourpair, ('->', '<-')): # For forwards and backwards datasets
                    # Raw Data
                    if raw_data:
                        raw_data_alpha = max(1.0/repetitions, 0.01)
                        for i in range(repetitions):
                            ax.scatter(X, time_means[..., i], marker = '.', color = colour, alpha = raw_data_alpha)

                    # Smoothed Mean Curve
                    ax.plot(smooth1D(X, k = smoothing_window), smooth1D(means, k = smoothing_window), color = colour, alpha = 0.7, lw = 2, label = '%s (%s sweep)' % (component.label, direction))

                    # Means
                    if mean_data:
                        ax.scatter(X, means, color = colour, alpha = 0.4)

                    # Variances
                    if variance:
                        ax.fill_between(X, means - vars_m, means + vars_p, facecolor = colour, alpha=0.05)

            # Legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)

            # Bounds            
            ax.set_xlim(v_range[0], v_range[1])
            all_data = sum(([c.means.forward, c.means.backward] for c in components), [])

            # Check that there is any data
            if any(len(data) > 0 for data in all_data): ymin, ymax = min(chain(*all_data)), max(chain(*all_data))
            else: ymin, ymax = 0, 0
            
            ypadding = y_paddding_percentage * (ymax - ymin)
            ax.set_ylim(ymin - ypadding, ymax + ypadding)
                
        # For title
        dVdt = '%.1g' % ((v_range[1] - v_range[0])/float(runtime))

    if dim == 2:
        x_range = input_ranges[0]
        y_range = input_ranges[1]
        X = np.linspace(x_range[0], x_range[1], steps)
        Y = np.linspace(y_range[0], y_range[1], steps)
        #X_GRID, Y_GRID = np.meshgrid(X, Y) # Input voltage grid

        # Smoothing Params
        order = 3
        zlevel = 3
        smoothing_window = smoothing_factor * steps
        #X_GRID = zoom(X_GRID, zlevel, order = order)
        #Y_GRID = zoom(Y_GRID, zlevel, order = order)
        #Z_GRID = zoom(Z_GRID, zlevel, order = order)

        ax_counter = 0
        for components, axlabel in zip(all_component_lists, ('Voltage (V)', 'Current Avg. (e/s)', 'Charge (e)')): # For quantity (V, I, Q)
            for ax, component in zip(axs[ax_counter:], components): # For each component/plot
                ax_counter += 1
                
                if variance:
                    var_ax = ax[1]
                    ax = ax[0]

                    var_ax.set_title(axlabel.split()[0] + ' Variance ' + axlabel.split()[-1])
                    var_ax.set_xlabel('%s Voltage (V)' % inputs[0].label)
                    var_ax.set_ylabel('%s Voltage (V)' % inputs[1].label)

                # Labels
                ax.set_title('%s %s' % (component.label, axlabel))
                ax.set_xlabel('%s Voltage (V)' % inputs[0].label)
                ax.set_ylabel('%s Voltage (V)' % inputs[1].label)

                imgs = []
                for means, colourmap, alpha in zip(component.means, colours, (1, 0.5)): # For forwards and backwards datasets
                    # Smooth
                    Z_GRID = gaussian_filter(means, smoothing_window)
                    Z_GRID = zoom(Z_GRID, zlevel, order = order)
                    
                    # Plot
                    img = ax.imshow(Z_GRID.T, origin = 'lower', extent = x_range + y_range, aspect = 'auto', alpha = alpha)
                    img.set_cmap(colourmap)
                    
                    imgs.append(img)

                # Colourbar
                divider = make_axes_locatable(ax)

                cax = divider.append_axes('right', size="5%", pad = 0.1)
                cbar = plt.colorbar(imgs[0], cax = cax)
                cbar.set_label('Forward sweep')
                cbar.set_ticks([])

                cax = divider.append_axes('right', size="5%", pad = 0.3)
                cbar = plt.colorbar(imgs[1], cax = cax)
                cbar.set_label('Backward sweep')

                ax.set_xlim(x_range)
                ax.set_ylim(y_range)
            
                # Variances
                if variance:
                    variances = (sum(component.vars_p) + sum(component.vars_m)) / 2

                    # Smooth
                    variances = gaussian_filter(variances, smoothing_window)
                    variances = zoom(variances, zlevel, order = order)

                    # Plot
                    img = var_ax.imshow(variances.T, origin = 'lower', extent = x_range + y_range, aspect = 'auto', alpha = 1.0)
                    img.set_cmap(colours[2])

                    # Colourbar
                    divider = make_axes_locatable(var_ax)
                    cax = divider.append_axes("right", size="5%", pad=0.1)
                    cbar = plt.colorbar(img, cax = cax)
                    cbar.set_label('%s Variance %s' % (axlabel.split()[0], axlabel.split()[-1]))
                    
                    var_ax.set_xlim(x_range)
                    var_ax.set_ylim(y_range)

                # For title
                dVdt = '(%.1g, %.1g)' % ((x_range[1] - x_range[0])/float(runtime), (y_range[1] - y_range[0])/float(runtime))
    # Title
    frequency = (1/float(runtime))
    frequency = '%.1f %sHz' % prefix(frequency)
    fig.text(.5, .93,
             'Characterisation Under Stepped (Quasi-DC), %s Bi-Directional\n Voltage Sweeps (mean dV/dt = %s V/s), T=%.1fK, Averaged Over %d repetitions' % (frequency, dVdt, nanosim.T, repetitions),
             horizontalalignment='center', fontsize=17)
    plt.tight_layout(rect = [0, 0, 1, .93])
        
        
    print 'done!'

    
    if show: plt.show()








def dynamic_analysis(runtime, v_probes = [], i_probes = [], q_probes = [], v_diffs = [], repetitions = 1, raw_data = True, show = True, fig = False, axs = False, colours = line_colours):
    '''
    Analyse a circuit over time.
        runtime: time, in seconds that each simulation is to be run for
        v_probes: components to be monitored for voltage-levels
        i_probes: components to be monitored for current
        q_probes: components to be monitored for charge-levels
        v_diffs: pairs (tuples) of components whose relative voltage is to be monitored (the plotted value will be: pair[0].V - pair[1].V)
        repetitions: run the sim this many times and average the results
        raw_data: plot the raw data-points?
        show: call pyplot.show() at the end. Us ethis if you want to modify the graph externally before showing.
        fig, axs: provide fig and axs objects for the plot to work with (if you want external control over the graph). Both must be provided if either is, and axs MUST have one subplot axis for each of voltage, current and charge if they are to be monitored.
        colours: change the default list of colours for plotting.

    At least one of v_probes, i_probes , q_probes or v_diffs should be filled or there will just be an empty plot.
    '''
    
    dont_plot = [] # Components that need to monitored for v_diff, but whose voltage shouldn't be independantly plotted.
    for pair in v_diffs:
        for c in pair:
            if c not in v_probes:
                dont_plot.append(c)
    v_probes += dont_plot # Add the not-for-plotting nodes to the monitoring-list

    dont_plot = Set(dont_plot) # For quick membership tests
    v_probe_indicies = dict(zip(v_probes, range(len(v_probes)))) # For quick-lookup of data when plotting v_diff pairs
    
    if (fig is False) or (axs is False): # If fig and axs aren't provided as args
        fig, axs = plt.subplots(len(nonempty(v_probes, i_probes, q_probes)), figsize=(14, 10)) # Plot size

    # The data for each component will be stored in these lists in the same order as the components are listed in the v/i/q_probes arguments
    V = [[] for p in v_probes]
    I = [[] for p in i_probes]
    Q = [[] for p in q_probes]

    all_probes = v_probes + i_probes + q_probes
    all_data = (V, I, Q)

    print 'Calculating...',

    #  Main computation
    for j in xrange(repetitions):
        nanosim.reset_logs() # Clear logs
        nanosim.simulate(runtime, logging = all_probes)

        # Extract data from component logs
        for i, p in enumerate(v_probes):
            V[i] += p.voltages
        
        for i, p in enumerate(i_probes):
            I[i] += p.current
        
        for i, p in enumerate(q_probes):
            Q[i] += p.charges

        print '%.1f%%' % (100 * float(j + 1)/repetitions)

    # Current is different to voltages because it's made up of delta-functions (which can't be easily interpolated).
    # Pre-smooth the delta functions into square-functions so that standard smoothing will work properly
    for i in range(len(I)):
        t_presmoothed, I_presmoothed = spread_signal_spikes(*zip(*I[i]), x_range = (0, runtime), m = m)
        I_presmoothed /= float(repetitions)
        I[i] = zip(t_presmoothed, I_presmoothed) # Replace the stored data with this smoothed version

    labels = (l for l, x in zip(('Voltage (V)', 'Current (e/s)', 'Charge (e)'), all_data) if bool(x)) # Lables for each graph (used for both title and y-axis)

    # Plot all the data (except voltage-diffs)
    smoothing_window = int(m * smoothing_factor) # Smoothing width in array-indicies
    for ax, data in zip(axs, nonempty(*all_data)): # Loop though V, I and Q
        for component_data, colour, component in zip(data, colours[::2], all_probes): # Loop though monitored components
            if component in dont_plot: continue # Skip voltage-diff only components
            
            component_data.sort(key = lambda x: x[0]) # Sort data by time
            t, y = zip(*component_data)

            # Plot raw data
            if raw_data: ax.scatter(t, y, color = colour, alpha = 0.4)

            # Interpolate data into evenly-spaced samples
            x_smooth = np.linspace(0, runtime, m)
            y_smooth = np.interp(x_smooth, t, y)

            # Gaussian-smooth the interpolated data
            x_smooth = smooth1D(x_smooth, k = smoothing_window) # The edges of the data will be auto-trimmed to the region of convolution-validity, so the x data needs to be smoothed and trimmed to match the y-data
            y_smooth = smooth1D(y_smooth, k = smoothing_window)

            # Plot the smoothed curve
            a = ax.plot(x_smooth, y_smooth, color = colour, alpha = 0.7, lw = 2, label = component.label)

    # Plot voltage-diffs
    ax = axs[0] # V axis
    already_plotted = len(v_probes) - len(dont_plot)
    for colour, pair in zip(colours[2*already_plotted::2], v_diffs):
        i, j = v_probe_indicies[pair[0]], v_probe_indicies[pair[1]]

        data1, data2 = V[i], V[j]

        # Sort data by time
        data1.sort(key = lambda x: x[0])
        data2.sort(key = lambda x: x[0])
        t1, y1 = zip(*data1)
        t2, y2 = zip(*data2)

        # Interpolate data into evenly-spaced samples
        x_smooth = np.linspace(0, runtime, m)
        y1_smooth = np.interp(x_smooth, t1, y1)
        y2_smooth = np.interp(x_smooth, t2, y2)

        # Calcualte the voltage-difference values
        y_smooth = y1_smooth - y2_smooth

        # Gaussian-smooth the interpolated data
        x_smooth = smooth1D(x_smooth, k = smoothing_window) # The edges of the data will be auto-trimmed to the region of convolution-validity, so the x data needs to be smoothed and trimmed to match the y-data
        y_smooth = smooth1D(y_smooth, k = smoothing_window)

        # Plot the smoothed curve
        a = ax.plot(x_smooth, y_smooth, color = colour, alpha = 0.7, lw = 2, label = 'V[%s - %s]' % (pair[0].label, pair[1].label))

    # Finalise the plots with labels, legends and bounds
    for ax, label in zip(axs, labels):
        # Legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

        # Labels
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(label)
        ax.set_title('%s vs Time' % label.split()[0])

        # Bounds
        ax.set_xlim(0, runtime) # Ensure that all the plots align properly
    
    print 'done!'

    # Title
    fig.text(.52, .955, 'Dynamic Circuit Analysis', horizontalalignment='center', fontsize=17)
    
    plt.tight_layout(rect = [0, 0, 1, .94])
    
    if show: plt.show()
