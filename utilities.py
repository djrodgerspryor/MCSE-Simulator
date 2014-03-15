import nanosim
import numpy as np
import functools
import itertools

_pascal_cache = {}
def pascal_row(n):
    " Returns the nth row of Pascal's triangle using recursion. Cached to make repeated calls extremely fast."
    if n in _pascal_cache: return _pascal_cache[n]
    else: _pascal_cache[n] = _pascal(n)
    return _pascal_cache[n]

def _pascal(n):
    return [1] if n<=0 else reduce(lambda row, n: row[:-1] + [(row[-1] + n), n], _pascal(n-1), [0])

def spread_signal_spikes(x, y, x_range = None, m = 500):
    "Turn instantaneous signal spikes into square-waves that can be properly integrated and convoluted"
    if not x_range: x_range = (np.min(x), np.max(x))
    x_range_length = float(x_range[1] - x_range[0])
    
    x_grid = np.linspace(x_range[0], x_range[1], m) # Evenly spaced time-samples
    y_grid = np.zeros(m) # set default current at 0
    
    for event in zip(x, y): # for each momentary current
        index = ((event[0] - x_range[0])/x_range_length) * m
        left = int(np.floor(index)) # Time-sampling index before the event

        # Make a square function for current over one sample-period:
        rect_height = event[1]/(x_range_length/m)
        y_grid[left] += rect_height
        if (left + 1) < m: y_grid[left + 1] += rect_height # If statement prevents boundry-error
    return x_grid, y_grid
    

def uneven_smooth1D(x, y, x_range = None, smoothing_factor = 0.01, m = 500):
    "Gaussian smooth unevenly-spaced data. smoothing_factor is a fraction of the x range."
    if not x_range: x_range = (np.min(x), np.max(x))
    smoothing_window = int(m * smoothing_factor) # Smoothing width in array-indicies

    events = sorted(zip(x, y), key = lambda pair: pair[0])
    x, y = zip(*events)
    
    x_smooth = np.linspace(x_range[0], x_range[1], m)
    y_smooth = np.interp(x_smooth, x, y)

    # The edges of the data will be auto-trimmed to the region of convolution-validity, so the x data needs to be smoothed and trimmed to match the y-data
    x_smooth = smooth1D(x_smooth, k = smoothing_window)
    y_smooth = smooth1D(y_smooth, k = smoothing_window)

    return x_smooth, y_smooth

def smooth1D(data, k):
    '''
        Gaussian smooth evenly-spaced data with k cells width.
        Output length will change to avoid convolution edge-effects. It's thus advisable to
        pass both X and Y data though smooth1D with the same parameters.

        >>> fake_noisy_data = [x + (1 if x%3 else -1) for x in xrange(10)]
        >>> fake_noisy_data
        [-1, 2, 3, 2, 5, 6, 5, 8, 9, 8]
        >>> smooth1D(fake_noisy_data, 3)
        array([ 2.  ,  2.75,  3.75,  5.  ,  5.75,  6.75,  8.  ])
    '''
    gaussian = np.array(map(float, pascal_row(k)))
    gaussian /= np.sum(gaussian)
    return np.convolve(data, gaussian, mode = 'valid')

prefixes = ('f', 'p', 'n', 'u', 'm', '', 'k', 'M', 'G', 'T', 'P')
def prefix(x):
    '''
        Get a scaled number and an SI prefix from fempto to Peta (10**(+/-)15).
        
        >>> prefix(999)
        (999.0, '')
        >>> prefix(1000)
        (1.0, 'k')
        >>> prefix(40*10**-13)
        (4.0, 'p')
        >>> prefix(2.95643*10**-75)
        (2.95643e-75, '')
    '''
    for i, p in enumerate(prefixes):
        boundary = 10**(3*(i-5))
        if x >= boundary and x < (boundary * 1000):
            return x/float(boundary), p
    return x, ''

def plot_circuit(conductors, t = 0):
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        G = nx.Graph()
        
        reservoirs = [i for i in conductors if isinstance(i, nanosim.Reservoir)]
        islands = [i for i in conductors if not isinstance(i, nanosim.Reservoir)]
        
        G.add_nodes_from(conductors)

        '''
        for n in G.nodes():
            if isinstance(n, nanosim.Reservoir):
                G[n]['t'] = 'r'
                #G[n]['label'] = '%.2gV' % n.V(t)
            else:
                G[n]['t'] = 'i'
                #G[n]['color'] = 'b'
        '''
        
        edge_labels = {}
        for c in conductors:
            for other, coupling in c.connections.items():
                if not G.has_edge(c, other):
                    edge_labels[(c, other)] = (('%.fKOhms' % (coupling.R/1000.0)) if coupling.R < float('inf') else '') + '  ' + (('%.2faF' % (coupling.C * 10.0**18)) if coupling.C > 0 else '')
                    G.add_edge(c, other, R = coupling.R, C = coupling.C)

        pos = nx.spring_layout(G)

        node_labels = dict([(n, '%s%.2fmV' % ((n.label + '\n' if n.label else ''), (1000 * n.V(t)))) for n in reservoirs] + [(n, n.label) for n in islands])
        colours = ['r' if isinstance(n, nanosim.Reservoir) else 'b' for n in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color = colours, node_size = 6000)
        nx.draw_networkx_labels(G, pos, labels = node_labels, font_size = 16)

        styles, widths = zip(*[('solid', 3) if n1.connections[n2].R < float('inf') else ('dotted', 1) for n1, n2 in G.edges()])
        nx.draw_networkx_edges(G, pos, color = 'k', style = styles, width = widths)
        nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)
        
        #nx.draw_networkx_nodes(G.nodes(t='i'), pos, color = 'b', node_labels = node_labels)
        #nx.draw_networkx_nodes(G.nodes(t='i', pos, color = 'b', node_labels = node_labels)
        plt.draw()
                        

    except Exception as e: print 'ERROR: graph could not be plotted: %s' % str(e)

def nonempty(*args):
    '''
        >>> nonempty([])
        []
        >>> nonempty([1])
        [[1]]
        >>> nonempty(range(5))
        [[0, 1, 2, 3, 4]]
        >>> nonempty(xrange(5))
        [xrange(5)]
    '''
    return [x for x in args if bool(x)]

def grouper(iterable, n):
    '''
        Collect data into fixed-length chunks or blocks. Ignore left over elements

        >>> list(grouper('ABCDEFG', 3))
        [('A', 'B', 'C'), ('D', 'E', 'F')]
    '''
    args = [iter(iterable)] * n
    return itertools.izip(*args)
pairwise = functools.partial(grouper, n=2) # Handy shorthand


### Unit Tests ###
def test_pascal_0():
    assert all(x == y for x, y in zip(pascal_row(0), (1,)))

def test_pascal_3():
    assert all(x == y for x, y in zip(pascal_row(0), (1,3,3,1)))

def test_prefix_one():
    assert prefix(1) == (1, '')

def test_prefix_999():
    assert prefix(999) == (999, '')

def test_prefix_k():
    assert prefix(1000) == (1, 'k')

def test_prefix_f():
    x, p = prefix(40.3 * 10**14)
    assert int(x), p == (403, 'f')

def test_prefix_overflow():
    x, p = prefix(7 * 10**67)
    assert int(x), p == (7**10**67, '')
