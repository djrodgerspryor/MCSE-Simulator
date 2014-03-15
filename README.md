MCSE-Simulator
==============

A Monte-Carlo Single Electron (MCSE) circuit simulator, with dynamic and static performance analysis.

The core algorithm was developed with reference to the following publications:
* 'Monte Carlo Simulation of Single Electronics Based on Orthodox Theory' - Ali A. Elabd, Abdel-Aziz T. Shalaby, El-Sayed M. El-Rabaie (2012)
* 'Monte Carlo simulation for single electron circuits' - Masaharu Kirihara and Kenji Taniguchi (1997)
* 'Boltzmann machine neural network devices using single-electron tunnelling' - Takashi Yamada, Masamichi Akazawa, Tetsuya Asai and Yoshihito Amemiya (2001)

Dependencies
------------
* Python 2.7
* numexpr
* numpy
* matplotlib



Example Usage
-------------

Import the core code, initialise the sim and specify a temperature in Kelvin (default 0K):

    import nanosim
    nanosim.initialise(temp = 0.025)
  
Define some components:

    ground = nanosim.Reservoir(V = 0, label = 'Ground')
    drain = nanosim.Reservoir(V = Vd, label = 'Drain')
    island = nanosim.Island(label = 'Box')`
  
There are two kinds of components: `Reservoirs`, which have a fixed (external) voltage, and `Islands`, which have well defined charge. Each component can be given a label for nice plot titles (and axis labels).
  
Specify couplings:

    ground.connect_to(island, R, 1*C)
    island.connect_to(drain, C = 9*C)
  
A coupling with no specified resistance is taken to be purely capacitative (meaning that tunnelling is impossible).
  
Simulate:

    nanosim.simulate(runtime)`

You'll probably want to collect data from your simulation and characterise your circuit under different conditions. For that, use the analysis library:

    from analysis import characterise, plot\_circuit, nanosim, dynamic\_analysis

Static voltage-characterisation (of island) under varying input voltage (on drain):

    characterise(runtime, inputs = [drain], input_ranges = [(0, Vd)], v_probes = [island], repetitions = 5)
  
`characterise` can handle up to two input voltage nodes (due to plotting limitations).
`plot\_circuit` can be used to visually check your component connections.


See the reference files for functional example code and demonstrations of dynamic analysis.

Notes
-------------
* nanosim.py currently uses global variables, making multiple, simultaneous simulations impossible.
* Static analysis sweeps voltage both forwards and backwards to detect hysteresis effects. The two datasets are plotted with similar (but distinct) colours on 1D plots, and distinct colours for 2D plots (where there is only one dataset per plot). If you're seeing these two datasets diverge, then you're getting hysteresis effects and need to increase your runtime to allow better thermalisation (ie. to make each step effectively equivalent to DC analysis; independent of the previous step).
* There are interfaces in the analysis functions for passing your own plot objects (axes and figure) and stopping the functions from drawing - this lets you take more control over the output is you want to do something fancy. See `analysis.py` for details.
* Electron tunnelling is instantaneous in the Monte-Carlo model, thus, the scale of currents measured in dynamic analysis (where individual events rather than time averages are recorded) is arbitrary.


Author
-------------
Daniel Rodgers-Pryor (2014)

Developed as part of my master's thesis (trying to build atomic scale Restricted Boltzmann Machines).
