"""Quick tests for the Plotter class."""
import random
from plotter import ScatterPlotter
import numpy as np

def test_single_y_labels():
    """Test plotting single (x, y) pairs at a time."""
    x = np.linspace(1, 1000)
    y = random.sample(list(np.linspace(0, 1, num=1000)), 1000)
    plt = ScatterPlotter('Single Line Test Plot', 'X Values', ['Y Values'])
    for (x_val, y_val) in zip(x, y):
        plt.plot(x_val, y_val)
    plt.show()

def test_multiple_y_labels():
    """Test plotting (x, y1), (x, y2), ... at the same time."""
    x = np.linspace(1, 10)
    y1 = random.sample(range(1, 100), 30)
    y2 = random.sample(range(150, 190), 30)
    plt = ScatterPlotter('Multiline Test Plot', 'X Values', ['Y1', 'Y2'])
    for (x_val, y_vals) in zip(x, zip(y1, y2)):
        plt.plot_all(x_val, y_vals)
    plt.show()

def test_multiple_y_labels_with_nonuniform_x():
    """Test plotting (x, y1) where there is no corresponding y2 value for x."""
    x = np.linspace(1, 10)
    y1 = random.sample(range(1, 100), 30)
    y1.extend([None]*20)
    y2 = [None]*20
    y2.extend(random.sample(range(1, 100), 30))
    plt = ScatterPlotter('Nonuniform X Plot', 'X Values', ['Y1', 'Y2'])
    for (x_val, y_vals) in zip(x, zip(y1, y2)):
        plt.plot_all(x_val, y_vals)
    plt.show()

test_single_y_labels()
test_multiple_y_labels()
test_multiple_y_labels_with_nonuniform_x()
