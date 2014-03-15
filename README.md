MCSE-Simulator
==============

A Monte-Carlo Single Electron (MCSE) circuit simulator, with dynamic and static performance analysis.

The core algorithm was developed with reference to the following publications:
* 'Monte Carlo Simulation of Single Electronics Based on Orthodox Theory' - Ali A. Elabd, Abdel-Aziz T. Shalaby, El-Sayed M. El-Rabaie (2012)
* 'Monte Carlo simulation for single electron circuits' - Masaharu Kirihara and Kenji Taniguchi (1997)
* 'Boltzmann machine neural network devices using single-electron tunnelling' - Takashi Yamada, Masamichi Akazawa, Tetsuya Asai and Yoshihito Amemiya (2001)

Dependancies
------------
* Python 2.7
* numexpr
* numpy
* matplotlib



Example Usage
-------------

Imprt the core simulation code:

  `import nanosim`
  
Initialise the sim and specify a temperature in Kelvim (default 0K):

  `nanosim.initialise(temp = 0.025)`
  
Define some components:

  `ground = nanosim.Reservoir(V = 0, label = 'Ground')`
  
  `drain = nanosim.Reservoir(V = Vd, label = 'Drain')`
  
  `island = nanosim.Island(label = 'Box')`
  
There are two kinds of components: Reservoirs, which have a fixed (external) voltage, and Islands, which have well defined charge.
  
Specify couplings:

  `ground.connect_to(island, R, 1*C)`
  
  `island.connect_to(drain, C = 9*C)`
  
A coupling with no specified resistance is taken to be puerly capacitative (meaning that tunnelling is impossible).
  
Simulate:

  `nanosim.simulate(runtime)`

You'll probably want to collect data from your simulation and characterise your circuit under different conditions. For that, use the analysis library:

  `from analysis import characterise, plot\_circuit, nanosim, dynamic\_analysis`

Static voltage-characterisation (of island) under varying input voltage (on drain):

  `characterise(runtime, inputs = [drain], input_ranges = [(0, Vd)], v_probes = [island], repetitions = 5)`
  
characterise can handle up to two input voltage nodes (due to plotting limitations).


See the reference files for functional example code and demonstrations of dynamic analysis.


Author
-------------
Daniel Rodgers-Pryor (2014)

Developed as part of my master's thesis (trying to build atomic scale Restricted Boltzmann Machines).
