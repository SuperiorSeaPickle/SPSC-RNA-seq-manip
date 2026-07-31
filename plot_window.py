from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QSplitter,
    QLabel
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt import NavigationToolbar2QT

import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


class ScatterPlotWindow(QMainWindow):
    def __init__(self, coords, colors, polygons):
        super().__init__()

        self.setWindowTitle("Cell Plot")

        # Create independent figures

        self.figure1 = Figure()
        self.canvas1 = FigureCanvasQTAgg(self.figure1)
        self.canvas = self.canvas1
        self.ax = self.figure1.add_subplot(111)

        self.figure2 = Figure()
        self.canvas2 = FigureCanvasQTAgg(self.figure2)
        self.ax2 = self.figure2.add_subplot(111)


        # First plot (UMAP)

        self.scatter = self.ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=colors,
            s=0.5
        )


        # Second plot (polygons)

        patches = [
            Polygon(poly, closed=True)
            for poly in polygons
        ]

        self.poly_collection = PatchCollection(
            patches,
            linewidths=0,
            alpha=1
        )

        self.ax2.add_collection(self.poly_collection)

        self.ax2.autoscale_view()
        self.ax2.set_aspect("equal")
        self.ax2.invert_yaxis()


        # Toolbars

        self.toolbar1 = NavigationToolbar2QT(
            self.canvas1,
            self
        )

        self.toolbar2 = NavigationToolbar2QT(
            self.canvas2,
            self
        )


        # Splitter layout

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        left_layout.addWidget(self.toolbar1)
        left_layout.addWidget(self.canvas1)


        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        right_layout.addWidget(self.toolbar2)
        right_layout.addWidget(self.canvas2)


        self.splitter = QSplitter(
            Qt.Orientation.Horizontal
        )

        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)

        # initial split ratio
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)


        # Main layout

        container = QWidget()
        layout = QVBoxLayout(container)

        layout.addWidget(self.splitter, stretch=1)

        self.legend = QWidget()
        self.legend.setMaximumHeight(100)  # limit legend area

        layout.addWidget(self.legend, stretch=0)

        self.setCentralWidget(container)

    def get_text_color(self, hex_color):
        """
        Returns black or white depending on background brightness.
        """
        import matplotlib.colors as mcolors

        r, g, b = mcolors.to_rgb(hex_color)
        
        # Convert 0-1 floats to 0-255
        r *= 255
        g *= 255
        b *= 255

        brightness = 0.299*r + 0.587*g + 0.114*b

        return "white" if brightness < 140 else "black"

    def update_legend(self, color_map):
        """
        color_map:
            dict of group_name -> color
        """

        import matplotlib.colors as mcolors
        from PyQt6.QtWidgets import QHBoxLayout

        # remove old legend
        if self.legend is not None:
            self.legend.deleteLater()

        self.legend = QWidget()

        legend_layout = QHBoxLayout(self.legend)

        # Prevent items from stretching across the row
        legend_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        for group, color in color_map.items():

            label = QLabel(str(group))

            hex_color = mcolors.to_hex(color)
            text_color = self.get_text_color(hex)

            label.setStyleSheet(
                f"""
                background-color: {hex_color};
                color: black;
                padding: 3px 8px;
                border-radius: 4px;
                """
            )

            # Make label only as large as its text
            label.setSizePolicy(
                label.sizePolicy().horizontalPolicy().Fixed,
                label.sizePolicy().verticalPolicy().Fixed
            )

            legend_layout.addWidget(label)

        legend_layout.setSpacing(5)

        self.centralWidget().layout().addWidget(self.legend) # type: ignore