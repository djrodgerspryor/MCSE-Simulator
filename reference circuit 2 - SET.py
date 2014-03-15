import nanosim
from analysis import characterise, plot_circuit, nanosim
import numpy as np
from scipy import interpolate
from analysis import dynamic_analysis, characterise

nanosim.initialise(temp = 0.025)

runtime = 1.0 * 10**-8

C = 1.0 * nanosim.aF # attofarads
R = 20 * nanosim.Rk
mV = 0.001 # millivolts

Vs = 2 * mV
Vg = 90 * mV

nanosim.voltage_tolerances(Vs)

# Components:
def set_components():
    global ground, drain, island, gate
    ground = nanosim.Reservoir(V = 0, label = 'Ground')
    drain = nanosim.Reservoir(V = Vs, label = 'Drain')
    gate = nanosim.Reservoir(V = Vg, label = 'Gate')
    island = nanosim.Island(label = 'Island', q0 = 0.3)

set_components()


# Connections:
def set_connections():
    # Tunnel Junctions
    ground.connect_to(island, R, 57.14*C)
    drain.connect_to(island, R, 53.94*C)

    # Capacitors
    island.connect_to(gate, C = 3.2*C)
    
set_connections()

'''
runtime = 1 # Seconds
def get_v(t):
    v = Vd * float(t)/runtime
    return v
drain.set_V(get_v)

#dynamic_analysis(runtime, v_probes = [island, drain], i_probes = [ground], v_diffs = [(drain, island)], repetitions = 10, raw_data = False)
'''

characterise(runtime, inputs = [gate, drain], input_ranges = [(0, Vg), (-Vs, Vs)], steps = 20, v_probes = [island], i_probes = [drain], q_probes = [island], repetitions = 2, mean_data = 0, variance = True) #, v_diffs = [(drain, ground)]
