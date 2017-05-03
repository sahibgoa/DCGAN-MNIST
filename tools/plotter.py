"""Centralize plotting code so that all graphs look the same."""
import plotly as ply
import plotly.graph_objs as go

class ScatterPlotter(object):
    """Plots one variable over another.

    Supports plotting of multiple lines on the same chart.
    """
    def __init__(self, title, x_label, y_labels):
        """Caches plot information for later display.

        Args:
            title: The title to show on the plot.
            x_label: The x-axis label.
            y_label: List of y-axis labels (can plot many at once).
        """
        if not len(y_labels):
            raise Exception('Please provide at least one y label!')

        self._title = title
        self._x_label = x_label
        self._y_labels = y_labels
        self._x_vals = []
        self._y_vals = {label:[] for label in y_labels}

        # Cache layout for later.
        self._layout = go.Layout(
            title=self._title,
            xaxis=go.Layout(title=self._x_label)
        )

        # Label y axis if only one y label.
        if len(y_labels) == 1:
            self._layout.yaxis = go.Layout(title=self._y_labels[0])

    def plot(self, x_val, y_val, y_label=None):
        """Plots the single (x, y) pair for the given label. If no label provided,
        plots the point under the first label.

        Args:
            x_val: The x coordinate of the new point.
            y_val: The y coordinate of the new point.
            y_label: The line to plot the point for.
        """
        if y_label and not y_label in self._y_vals.keys():
            raise Exception('Invalid y label %s' % y_label)
        y_label = y_label or self._y_labels[0]
        self._x_vals.append(x_val)
        for (label, vals) in self._y_vals.items():
            vals.append(y_val if label == y_label else None)

    def plot_all(self, x_val, y_vals):
        """Adds the point (x, [y1, y2, ...]) to the plot but does not display the plot.
        Must pass one y value for every y label provided at construction. If there is no
        y value available for a given label, pass None in its place.

        Args:
            x_val: The x coordinate of the new point.
            y_vals: List of values for the x-coordinate, one for each y label provided.
        """
        if len(y_vals) != len(self._y_vals):
            raise Exception(
                'Provided %d y values, expected %d' % (len(y_vals), len(self._y_labels)))
        self._x_vals.append(x_val)
        for (label, val) in zip(self._y_labels, y_vals):
            self._y_vals[label].append(val)

    def show(self):
        """Displays the plot with its current data."""
        ply.offline.plot(dict(data=self._construct_lines(), layout=self._layout))

    def save(self, file_path):
        """Saves the plot to the given file path.

        Args:
            file_path: The path at which to save the file.
        """
        ply.offline.plot(
            dict(data=self._construct_lines(), layout=self._layout),
            filename=file_path
        )

    def _construct_lines(self):
        """Constructs the scatterplot lines for the current data.

        Returns:
            A list of graph_objs.Scatter objects, one for each y label.
        """
        return [
            go.Scatter(
                x=[x_val for (x_val, y_val) in zip(self._x_vals, self._y_vals[y_label]) if y_val],
                y=[y_val for y_val in self._y_vals[y_label] if y_val],
                name=y_label,
                mode='lines+markers')
            for y_label in self._y_labels
        ]
