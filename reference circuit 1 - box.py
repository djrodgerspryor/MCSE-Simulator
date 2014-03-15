import nanosim
from analysis import characterise, plot_circuit, nanosim, dynamic_analysis
import numpy as np
from scipy import interpolate

nanosim.initialise(temp = 0.025)

C = 1.0 * nanosim.aF # attofarads
R = 20 * nanosim.Rk
mV = 0.001 # millivolts

Vd = 50 * mV

nanosim.voltage_tolerances(Vd)

# Components:
def set_components():
    global ground, drain, island
    ground = nanosim.Reservoir(V = 0, label = 'Ground')
    drain = nanosim.Reservoir(V = Vd, label = 'Drain')
    island = nanosim.Island(label = 'Box')

set_components()


# Connections:
def set_connections():
    # Tunnel Junctions
    ground.connect_to(island, R, 1*C)

    # Capacitors
    island.connect_to(drain, C = 9*C)
    
set_connections()


# Static analysis
runtime = 10**-6 # Seconds
characterise(runtime, inputs = [drain], input_ranges = [(0, Vd)], steps = 500, v_probes = [island, drain], i_probes = [ground], v_diffs = [(drain, island)], repetitions = 5, mean_data = False)

# Dynamic analysis (for the same voltage range)
runtime = 1 # Seconds
def get_v(t):
    v = min(Vd * float(t), Vd) # V increases from 0 to Vd over 1 sec, then stays constant.
    return v
drain.set_V(get_v)

dynamic_analysis(runtime, v_probes = [island, drain], i_probes = [ground], v_diffs = [(drain, island)], repetitions = 1, raw_data = False)
