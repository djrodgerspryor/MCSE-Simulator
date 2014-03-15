# For a description of the algorithm used here, see: Monte Carlo simulation for single electron circuits - M. Kirihara and K. Taniguchi 1997
import numpy as np
from itertools import izip, chain, repeat, takewhile, combinations
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import namedtuple
import numexpr as ne

# TODO: This module abuses global variables, which makes it difficult to run multiple simulations in parallel (as well as being bad practice). This should all be wrapped in a class.


### Physical Constants ###

Rk = 25800 # Quantum resistance = h/e^2 is the base value of tunneling resistance (Ohms)
e = 1.6021 * 10**-19 # Electron Charge (Coulombs)
e_sq = e**2
Kb = 1.38065 * 10**-23 # (J/Kelvin)
aF = 1.0 * 10**-18 # (Farads)

inf = np.inf
ninf = -np.inf

### Simulation Constants ###

max_delta_V_fraction = 0.01 # Maximal fractional change in voltage in a time-step
v_typical = 1.0 # Typical voltages (for working with percentage changes). Since voltages are often 0 at some point, you can't directly use fractions of the current voltage levels.
max_delta_V = v_typical * max_delta_V_fraction # For voltages changing with time, this is the maximum voltage change allowed between simulation steps (Volts)

### Utility Functions ###

def null_connection():
    return Coupling(np.inf, 0.0)

### Classes ###

Coupling = namedtuple('Coupling', 'R C')

class Island(object):
    '''
        Islands are conductors that contain a fixed number of localised electrons. They have unknown potential, but known charge.
        Islands cannot be reliably queried for potential.
    '''
    # Connections to other components. dicts with other-island:values, where values is a tuple of (resistance, capacitance):
    n = 0 # Currrent number of electrons (n positive indicted negative charge)
    n0 = 0.0 # Fixed charge offset
    V = None # Unreliable - may be set by simulation

    def __init__(self, label = '', q0 = 0, charge_limits = (None, None)):
        '''
            Provide a label for descriptive plotting.
            q0 is the fixed island charge in electrons (can be, and usually is, fractional)
            Island will be prevented from exceeding (min, max) charge limits (in electrons)
        '''
        global components_altered
        components_altered = True
        
        self.connections = defaultdict(null_connection)
        conductors.append(self) # Global tracking of condctors

        # Initial, empty logs
        self.voltages = []
        self.charges = [(0, 0)]

        self.n_limits = (charge_limits[0] if charge_limits[0] is not None else ninf, charge_limits[1] if charge_limits[1] is not None else inf)

        self.n0 = q0
        
        self.label = label # For circuit analysis and plotting

    def connect_to(self, other, R=np.inf, C=0.0):
        global components_altered
        components_altered = True
        
        self.connections[other] = Coupling(R, C)
        other.connections[self] = Coupling(R, C) # Reverse connection

    def set_V(self, V):
        'Set (and log) voltage'
        self.voltages.append((time, V))
        self.V = V

    def log(self):
        'Create a log entry of the present state at the present time'
        self.voltages.append((time, self.V))
        self.charges.append((time, self.n))

    def charge(self, m):
        self.n -= m

        # Check charge limits
        if self.n < self.n_limits[0] or self.n > self.n_limits[1]:
            self.n += m # Undo charge
            raise ValueError("Disallowed island charge requested (current: %d, requested change:%d). This island ('%s') is limited to charge-range: %s" % (self.n, m, self.label, str(self.n_limits)))
        self.charges.append((time, self.n))

    @property
    def q(self):
        'Charge in culombs.'
        return (self.n - self.n0) * e

    @property
    def total_capacitance(self):
        return np.sum([c.C for c in self.connections.values()])

    @property
    def minimal_time_constant(self):
        'Rough characteristic timescale for events on this conductor.'
        return np.min([c.C*c.R for c in self.connections.values()])
        

class Reservoir(Island):
    '''
        Reservoirs are conductors that are held externally at a fixed voltage. They have unknown charge, but known potential.
        Reservoirs cannot be reliably queried for charge.
        
        Note: Current is an instantaneous event, not a change of state. The events in the current log should be interpreted as delta functions, scaled by
        the given value.
    '''
    n = None # Unreliable - may be set by simulation
    
    def __init__(self, V = 0, *args, **kwargs):
        super(Reservoir, self).__init__(*args, **kwargs)
        self.current = [(0, 0)] # All current events
        self._V = V  # Voltage can be a constant, or a function of time.
    
    def charge(self, m):
        'Charge the reservoir by having it absorb/release electrons and do work to replenish its voltage level'
        self.current.append((time, m)) # Log current flow, but don't bother updating (since reservoirs always have effectivley infinite charge)

    def log(self):
        "Log voltages only; currents get logged as they happen, and charge isn't defined for reservoirs."
        self.voltages.append((time, self.V(time)))

    def set_V(self, V):
        self._V = V
    
    def V(self, t):
        'Get voltage at a given time'
        try: return self._V(t) # Dynamic volatge as a function of time
        except TypeError: return self._V # Constant voltage


### Simulation Functions ###

def initialise(temp = 0.0):
    'Set the temperature, and initialise the global list of conductors. This must be called before components can be created.'
    global T, conductors, KbT, components_altered
    conductors = []
    T = temp # (Kelvin)
    KbT = Kb*T # For efficiency

    # Mark that the components have not yet been processed
    components_altered = True

def reset_components():
    'Call this if any connections are changed to force the sim to re-check all the connections'
    global components_altered
    components_altered = True
    
def reset_sim():
    'Reset capacitance, matricies, time, logs and charge states.'
    global time
    time = 0
    reset_logs()

    # Reset island states
    for c in conductors:
        if not isinstance(c, Reservoir):
            c.n = 0
            c.V  = 0

def set_capacitances_from_matrix(maxwell_cap, component_order):
    '''
        [numpy array] maxwell_cap must be a Maxwell capacitance matrix (the kind fastcap produces), defined as follows:
            M_ii = total capacitance of component i (ie. sum over (all j =/= i) Cij)
            Mij (i =/= j) = -Cij = - Cji
        [list] component_order is a list of components corresponding to the matrix row/column order.

        This wil overwrite any existing capacitance values, but preserve resistance values.
    '''
    global components_altered
    components_altered = True
    
    for combination in combinations(enumerate(component_order), enumerate(component_order)):
            i, ci = combination[0]
            j, cj = combination[1]
            Cij = (maxwell_cap[i, j] + maxwell_cap[j, i]) * (-0.5) # Average the symmetric values to minimise random error
            
            ci.connect_to(cj, R = ci.connections[cj].R, C = Cij) # Set cpacitance while preserving existing resistance value

def reset_logs():
    'Reset current, voltage and charge logs for all components'
    for c in conductors:
        c.voltages = [] # Reset voltage log
        if not isinstance(c, Reservoir): c.charges = [(0, 0)] # Reset charge log for islands
        else: c.current = [(0, 0)] # Reset current log for reservoirs

def component_access_tools():
    'Define a bunch of useful datastructures for addressing reservoirs and islands seperatley'
    global reservoirs, islands, conductors_to_reservoirs, conductors_to_islands, reservoirs_to_conductors, islands_to_conductors, island_filter, reservoir_filter
    
    reservoirs, islands = [], []
    conductors_to_reservoirs = -1 * np.ones(len(conductors), dtype=np.int) # The ith value of this will be the index of conductors[i] in reservoirs
    conductors_to_islands = -1 * np.ones(len(conductors), dtype=np.int) # The ith value of this will be the index of conductors[i] in islands
    reservoirs_to_conductors = [] # The ith value of this will be the index of reservoirs[i] in conductors
    islands_to_conductors = [] # The ith value of this will be the index of islands[i] in conductors
    for i, c in enumerate(conductors):
        if isinstance(c, Reservoir):
            conductors_to_reservoirs[i] = len(reservoirs)
            reservoirs.append(c)
            reservoirs_to_conductors.append(i)
        else:
            conductors_to_islands[i] = len(islands)
            islands.append(c)
            islands_to_conductors.append(i)
    reservoirs_to_conductors = np.array(reservoirs_to_conductors, dtype=np.int)
    islands_to_conductors = np.array(islands_to_conductors, dtype=np.int)

    # Boolean maps for the indicies of islands and reservoirs
    reservoir_filter = (conductors_to_reservoirs >= 0)
    island_filter = (conductors_to_islands >= 0)

def voltage_tolerances(v = v_typical, step_delta_V_fraction = max_delta_V_fraction):
    'Tell the simulation about the typical voltages of your simulation so that it can adjust its maximal timestep accordingly.'
    global max_delta_V_fraction, v_typical, max_delta_V
    max_delta_V_fraction = step_delta_V_fraction # New allowable fractional change in voltage
    v_typical = v # New typical voltage
    max_delta_V = v_typical * max_delta_V_fraction # Update absolute allowable delta-V
    

def rates(dE, Rt = None):
    '''
        Calculate the tunelling rate from energy change and tunnel-resistance.
        dE and Rt must be numpy 1D arrays.
        
        Formula:
            rate = Gamma = dE/(e^2 * Rt * (1 - exp(-dE/(Kb*T))))

        Parallelised with numexpr.
    '''
    if Rt is None: Rt = Rk * np.ones(len(dE)) # Default to the minimum resistace for electron localisation
    rates = np.zeros(len(dE))
    
    factor = ne.evaluate('dE/(e_sq * Rt)')

    if T == 0: # Step function behaviour for zero temperature
        nonzero_rates = dE > 0
        rates[nonzero_rates] = factor[nonzero_rates]
        return rates
    
    return ne.evaluate('where(dE == 0, KbT/(e_sq * Rt), factor/(1 - exp(-dE/(KbT))))')

def sample_times(rates):
    '''
        Generate monte-carlo samples; stochastically choose the time until the next event based on its rate of occurence.

        Parallelised with numexpr.
    '''
    randomness = np.random.random(len(rates))
    return ne.evaluate('where(rates <= 0, inf, (1.0/rates) * log(1.0/randomness))')

def get_possible_tunnellings(conductors):
    'Get pairs of indicies for components that may tunnel to one another at some point in the simulation. Returns an (n*2) numpy array.'
    indicies = range(len(conductors))
    return np.array(sum(([(i, j) for i in indicies if
                    ((i != j) and # Ignore self-tunnellings
                    (conductors[i].connections[conductors[j]].R < np.inf)) # Resistance much be non-infinite (ie. don't allow tunnelling across purely capacitative junctions).
                ] for j in indicies), []), dtype = np.int)

def capacitance_matricies(reservoirs, islands):
    '''
        Capacitance interconnection sub-matricies.
        See: Monte Carlo Simulation of Single Electronics Based on Orthodox Theory (Ali A. Elabd et al.)
    '''
    
    # Ca
    reservoir_couplings = np.zeros([len(reservoirs)]*2) # Size is len(reservoirs) * len(reservoirs)
    for i, j in combinations(range(len(reservoirs)), 2): # Off-diagonals
        if i != j: reservoir_couplings[i][j] = reservoir_couplings[j][i] = -reservoirs[i].connections[reservoirs[j]].C
    for i, reservoir in enumerate(reservoirs): # Diagonal
        reservoir_couplings[i][i] = reservoir.total_capacitance

    # Cb
    reservoir_island_couplings = np.zeros([len(reservoirs), len(islands)]) # Size is len(reservoirs) * len(islands)
    for i, reservoir in enumerate(reservoirs):
        for j, island in enumerate(islands):
            reservoir_island_couplings[i][j] = -reservoir.connections[island].C

    # Cc
    island_couplings = np.zeros([len(islands)]*2) # Size is len(islands) * len(islands)
    for i, j in combinations(range(len(islands)), 2): # Off-diagonals
        if i != j: island_couplings[i][j] = island_couplings[j][i] = -islands[i].connections[islands[j]].C
    for i, island in enumerate(islands): # Diagonal
        island_couplings[i][i] = island.total_capacitance

    return reservoir_couplings, reservoir_island_couplings, island_couplings

def get_island_charges():
    'Fetch current charge states'
    return np.array([i.q for i in islands])

def get_reservoir_potentials(t):
    'Fetch current potential states'
    return np.array([r.V(t) for r in reservoirs])

def repolarisation_work(reservoir_potentials, reservoir_charges_initial, reservoir_charges_final):
    '''
        Work done by the reservoirs to repolarise the system.

        Parallelised with numexpr.
    '''
    intermediate = ne.evaluate('(reservoir_potentials * (reservoir_charges_final - reservoir_charges_initial))').T
    return ne.evaluate('sum(intermediate, axis = 0)').T

def replacement_work(sources, destinations, extended_reservoir_potentials):
    '''
        Work done by the reservoirs to replace lost electrons.

        Parallelised with numexpr.
    '''
    source_potentials = extended_reservoir_potentials[sources]
    destination_potentials = extended_reservoir_potentials[destinations]
    return ne.evaluate('e * (source_potentials - destination_potentials)')


# Energy calculation:
def energy_calculation_matricies(Ca, Cb, Cc):
    'Turn the submatricies Ca, Cb and Cc into useful matricies for calculating energies.'
    global charge_work_matrix, reservoir_work_matrix, charge_potential_matrix
    
    charge_work_matrix = np.linalg.inv(Cc) # Cc^-1
    
    reservoir_work_matrix = Ca - Cb.dot(charge_work_matrix.dot(np.transpose(Cb))) # Ca - Cb Cc^-1 Cb^T
    
    charge_potential_matrix = np.array(np.bmat([[reservoir_work_matrix, Cb.dot(charge_work_matrix)],
                                       [charge_work_matrix.dot(np.transpose(Cb)), charge_work_matrix]]))

"""    
def electrostatic_energies(reservoir_potentials, final_island_potentials, final_reservoir_charges, final_island_charges):
    '''
        Get the electrostatic energy of the system.
    '''
    E_reservoir = np.sum(reservoir_potentials * final_reservoir_charges, axis = -1)

    E_island = np.sum(final_island_potentials * final_island_charges, axis = -1)

    
    # Energy from isolated electrostatic charges:
    #E_island = island_charges.dot(charge_work_matrix)
    #E_island = np.sum(ne.evaluate('(E_island * island_charges)'), axis = -1) # 'len(E_island.shape) - 1' is equivilent to setting axis to -1
    
    #E_reservoir = reservoir_potentials.dot(reservoir_work_matrix.dot(reservoir_potentials)) # Energy from charges held at an external potential
    
    return 0.5*(E_island + E_reservoir)
"""

def electrostatic_energies(reservoir_potentials, island_charges):
    '''
        Get the electrostatic energy of the system.
    '''
    # Energy from isolated electrostatic charges:
    E_island = island_charges.dot(charge_work_matrix)
    E_island = np.sum(ne.evaluate('(E_island * island_charges)'), axis = -1) # 'len(E_island.shape) - 1' is equivilent to setting axis to -1
    
    E_reservoir = reservoir_potentials.dot(reservoir_work_matrix.dot(reservoir_potentials)) # Energy from charges held at an external potential
    return ne.evaluate('0.5*(E_island + E_reservoir)')


def full_charge_potentials(reservoir_potentials, island_charges):
    '''
        Transform reservoir potentials and island charges into reservoir charges and island potentials.
    '''
    r_V_and_i_Q = np.hstack((reservoir_potentials, island_charges))
    r_Q_and_i_V = -charge_potential_matrix.dot(r_V_and_i_Q.T)
    return r_Q_and_i_V[:reservoir_potentials.shape[-1]], r_Q_and_i_V[-island_charges.shape[-1]:] # reservoir charges, island potentials

def free_energy_change(events, island_charge_limits, reservoir_potentials, initial_reservoir_charges, initial_island_charges, E_electrostatic):
    '''
        dF = dE_electrostatic - dW
    '''

    # Seperate sources and destinations, convert them to island-indices (rather than conductor-indices), and filter the list to ignore reservoirs
    island_sources = events[:, [0, 1]] # array of: event-index, island-index
    island_sources[:, 1] = conductors_to_islands[island_sources[:, 1]]
    island_sources = island_sources[island_sources[:, 1] >= 0]

    island_destinations = events[:, [0, 2]] # array of: event-index, island-index
    island_destinations[:, 1] = conductors_to_islands[island_destinations[:, 1]] 
    island_destinations = island_destinations[island_destinations[:, 1] >= 0]

    # Potentials for all conductors, extended with 0 if the conductor is an island
    extended_reservoir_potentials = np.zeros(len(conductors))
    extended_reservoir_potentials[reservoir_filter] = reservoir_potentials

    dQ = np.zeros((events.shape[0], len(islands))) # Charge changes for each component caused by each event. Shape: events * components
    
    # Address the islands for each event in (event-index, component-index) format, and set charge change to +/-1
    dQ[island_sources[:, 0], island_sources[:, 1]] += e # Source looses an electron - gains a charge of +e
    dQ[island_destinations[:, 0], island_destinations[:, 1]] += -e # Destination gains an electron - gains a charge of -e
    
    final_island_charges = initial_island_charges + dQ # Shape: events * islands
    
    # Find allowable events (by island charge constraints):
    min_island_charge_limits, max_island_charge_limits = island_charge_limits[:, 0], island_charge_limits[:, 1]
    allowable_charges_filter = ne.evaluate('(final_island_charges <= max_island_charge_limits) & (final_island_charges >= min_island_charge_limits)')
    allowable_events_filter =  np.all(allowable_charges_filter, axis = 1)

    # Filter disallowed events
    final_island_charges = final_island_charges[allowable_events_filter, :]
    events = events[allowable_events_filter]

    # Reservoir charges and island potentials for all possible events:
    repeated_reservoir_potentials = reservoir_potentials.reshape((1,) + reservoir_potentials.shape).repeat(len(events), 0) # Shape: events * reservoirs
    final_reservoir_charges, final_island_potentials = full_charge_potentials(repeated_reservoir_potentials, final_island_charges)
    final_reservoir_charges = final_reservoir_charges.T
    final_island_potentials = final_island_potentials.T

    # Change in electrostatic energy from each event (change in the potential energy from concentrated charges on the islands)
    dE_electrostatic = electrostatic_energies(reservoir_potentials, final_island_charges) - E_electrostatic # Shape: events
    
    # dW = (dW_T + dW_P) ~ energy change from reservoirs
    dW_T = repolarisation_work(reservoir_potentials, initial_reservoir_charges, final_reservoir_charges)
    dW_P = replacement_work(events[:, 1], events[:, 2], extended_reservoir_potentials)

    return -ne.evaluate('dE_electrostatic - dW_T - dW_P'), allowable_events_filter, events

def update_state():
        reservoir_potentials = get_reservoir_potentials(time)
        island_charges = get_island_charges()
        reservoir_charges, island_potentials = full_charge_potentials(reservoir_potentials, island_charges)
        E_electrostatic = electrostatic_energies(reservoir_potentials, island_charges)
        
        # Update node attributes:
        for V, i in zip(island_potentials, islands):
            i.set_V(V)
            
        return reservoir_potentials, island_charges, reservoir_charges, island_potentials, E_electrostatic


def simulate(runtime, callback = lambda t, d: None, logging = []):
    '''
        Set-up and run a simulation.
        
        callback is function that will be called after each tunnelling event.
        logging is a list of components to log with high frequency.
    '''
    global time, components_altered, possible_tunnellings, R_tunnelling, island_charge_limits
    
    time = 0.0 # (Seconds)
    # Note: time will generally be updated after each tunnelling event, not in fixed steps. This is, essentially, a perfect adaptive-timestep algorithm.
    # This ceases to be true if reservoir voltages are changing with time - they may only change by a limited amount in a given timestep.
    
    # Process the components into useful formats (capacitance matricies, lists of possible tunelling events etc.)
    if components_altered:
        component_access_tools() # Define a bunch of global datastructures for addressing reservoirs and islands
        
        # Capacitance matricies for energy calculation:
        Ca, Cb, Cc = capacitance_matricies(reservoirs, islands)
        energy_calculation_matricies(Ca, Cb, Cc)

        # All possible events:
        possible_tunnellings = get_possible_tunnellings(conductors)
        possible_tunnellings = np.vstack((np.arange(possible_tunnellings.shape[0]), possible_tunnellings.T)).T # Add an index on the left

        island_charge_limits = (e) * np.array([i.n_limits for i in islands]) # Max and min charges (for atomic islands).
    
        # Resitances for each possible event:
        R_tunnelling = np.array([conductors[i].connections[conductors[j]].R for (i, j) in possible_tunnellings[:, 1:]])
        
        components_altered = False # Mark that the datastructures that we just calculated are fresh

    # Initial state
    reservoir_potentials, island_charges, reservoir_charges, island_potentials, E_electrostatic = update_state()
        
    while True:
        for c in logging: c.log() # Log data on watched components

        # Energetic effects of each event:
        dE, allowable_events_filter, allowed_tunnellings = free_energy_change(possible_tunnellings, island_charge_limits, reservoir_potentials, reservoir_charges, island_charges, E_electrostatic)

        # Filter by events that leave the islands in an allowable charge state
        R_t = R_tunnelling[allowable_events_filter]
        
        # Resistances and rates of each event:
        tunnelling_rates = rates(dE, R_t)

        # Stochastically sample the times until each event would be expected to occur:
        tunnelling_times = sample_times(tunnelling_rates) # Randomness is inserted here

        # Select the earliest event:
        i = np.argmin(tunnelling_times)
        event, delay = allowed_tunnellings[i, 1:], tunnelling_times[i]

        # Check that the next event isn't so distant and the input voltages will have changed significantly in the interim
        if np.max(np.abs(reservoir_potentials - get_reservoir_potentials(time + delay))) > max_delta_V:
            # If the delay is too long, step ahead for a smaller delay without tunnelling
            
            # Make sure that delay is finite so that we can use it to find a sufficiently small delay:
            if not np.isfinite(delay): delay = runtime

            # Binary search for a maximal timestep that doesn't violate voltage change constraint:
            step = delay/2.0
            while np.max(np.abs(reservoir_potentials - get_reservoir_potentials(time + step))) > max_delta_V: step /= 2.0

            # Step forward:
            time += step

            # Update state:
            reservoir_potentials, island_charges, reservoir_charges, island_potentials, E_electrostatic = update_state()

            # Quit if the simulation has reached it's end time:
            if (time) > runtime: return

            continue # Skip tunnelling
        
        # Quit if the next event won't happen before the simulation ends:
        if (time + delay) > runtime: return

        # Skip ahead until the event occurs:
        time += delay

        for c in logging: c.log() # Log the pre-change state on watched components

        # Tunnel:
        conductors[event[0]].charge(-1)
        conductors[event[1]].charge(1)

        # Update state:
        reservoir_potentials, island_charges, reservoir_charges, island_potentials, E_electrostatic = update_state()

        # Run a callback function
        callback(time, delay)


    
