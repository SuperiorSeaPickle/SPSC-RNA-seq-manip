from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

class ScatterPlotWindow(QMainWindow):
    def __init__(self, coords, colors):
        super().__init__()
        self.setWindowTitle("Cell Plot")

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)

        self.scatter = self.ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=colors,
            s=0.5
        )

        self.legend = None  # track so we can remove/replace it

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.canvas)
        self.setCentralWidget(container)
        self.figure.subplots_adjust(right=0.75)
        
        self.canvas.draw()

    def update_legend(self, color_map):
        """color_map: dict of group_name -> RGBA tuple (or hex string)"""
        if self.legend is not None:
            self.legend.remove()

        handles = [
            mpatches.Patch(color=color, label=str(group))
            for group, color in color_map.items()
        ]

        self.legend = self.ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),  # place outside the plot, right side
            fontsize=7,
            markerscale=0.5,
            frameon=True,
        )

        self.figure.tight_layout()