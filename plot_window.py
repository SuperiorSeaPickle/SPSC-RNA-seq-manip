from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QSplitter,
    QLabel,
    QSlider,
    QHBoxLayout,
    QCheckBox
)
from PyQt6.QtCore import Qt, QTimer

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from superqt import QRangeSlider
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

import numpy as np

# 0 (12, 28048, 46543)
# 1 (12, 14024, 23271)
# 2 (12, 7012, 11635)
# 3 (12, 3506, 5817)
# 4 (12, 1753, 2908)
# 5 (12, 876, 1454)
# 6 (12, 438, 727)
# 7 (12, 219, 363)
class ScatterPlotWindow(QMainWindow):
    def __init__(self, coords, colors, polygons, tf):
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
        self.image_state = {
            "show_dapi": True,
            "dapi_contrast": (2,99),
            "projection": "single",
            "z_plane": 6,
        }

        self.tf = tf
        ome_xml = self.tf.ome_metadata
        print(ome_xml)
        self.series = self.tf.series[0]

        #contrast slider:

        
        # Create the background image once
        self.dapi_background = self.ax2.imshow(
            np.zeros((10, 10), dtype=np.float32),
            cmap="Greys",
            vmin=0,
            vmax=1,
            extent=(0,1,0,1),
            origin="upper",
            interpolation="nearest",
            zorder=0
        )
        self.dapi_background.set_visible(False)

        # Run update when zoom/pan finishes
        self.canvas2.mpl_connect(
            "button_release_event",
            self.update_background
        )

        # Also update after toolbar zoom redraws
        self.canvas2.mpl_connect(
            "draw_event",
            self.update_background
        )

        # First plot (UMAP)

        self.scatter = self.ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=colors,
            s=0.5
        )


        # Second plot (polygons)

        mpp = 0.21249222 #update this to read from metadata!
        patches = [
            Polygon(poly/mpp, closed=True)
            for poly in polygons
        ]

        self.poly_collection = PatchCollection(
            patches,
            linewidths=0,
            alpha=1
        )

        self.ax2.add_collection(self.poly_collection) # type: ignore
        self.ax2.set_xlim(0,self.series.levels[0].shape[2])
        self.ax2.set_ylim(self.series.levels[0].shape[1],0)

        self.ax2.set_aspect("equal")


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
        sett_layout = QVBoxLayout(container)

        sett_layout.addWidget(self.splitter, stretch=1)

        # image controls
        self.image_controls = self.create_image_controls()
        sett_layout.addWidget(
            self.image_controls,
            stretch=0
        )

        self.legend = QWidget()
        self.legend.setMaximumHeight(100)

        sett_layout.addWidget(
            self.legend,
            stretch=0
        )

        self.setCentralWidget(container)

        QTimer.singleShot(
            100,
            self.update_background
        )
        self._last_region = None
    def get_mpp(self, tf):
        # Try OME-XML metadata first (most reliable for WSI formats)
        if getattr(tf, "ome_metadata", None):
            import xml.etree.ElementTree as ET
            root = ET.fromstring(tf.ome_metadata)
            ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
            pixels = root.find(".//ome:Pixels", ns)
            for i, ch in enumerate(root.findall(".//ome:Channel", ns)):
                print(i, ch.get("Name"), ch.get("Fluor"), ch.get("EmissionWavelength"))

        # Fallback: standard TIFF resolution tags
        page = tf.series[0].levels[0].pages[0]
        tags = page.tags
        if "XResolution" in tags and "ResolutionUnit" in tags:
            res = tags["XResolution"].value  # (numerator, denominator)
            unit = tags["ResolutionUnit"].value  # 2=inch, 3=cm

            pixels_per_unit = res[0] / res[1]

            if unit == 3:  # centimeter
                pixels_per_micron = pixels_per_unit / 10000
            else:  # inch
                pixels_per_micron = pixels_per_unit / 25400

            return 1 / pixels_per_micron

        raise ValueError("Could not determine microns-per-pixel from file metadata")
    def compute_scale(self, xmin, xmax):
        viewport_width = xmax - xmin
        dpr = self.canvas2.devicePixelRatioF()
        screen_width = max(self.canvas2.width() * dpr, 1)

        ppsp = viewport_width / screen_width

        full_width = self.series.levels[0].shape[2]

        for i, level in enumerate(self.series.levels):
            level_width = level.shape[2]
            downsample = full_width / level_width
            if downsample >= ppsp:
                return i

        return len(self.series.levels) - 1
    def update_background(self, event=None):
        if getattr(self, "_updating_background", False):
            return

        x_lo, x_hi = self.ax2.get_xlim()
        xmin, xmax = min(x_lo, x_hi), max(x_lo, x_hi)

        y_lo, y_hi = self.ax2.get_ylim()
        ymin, ymax = min(y_lo, y_hi), max(y_lo, y_hi)

        level_index = self.compute_scale(xmin, xmax)
        level = self.series.levels[level_index]

        downsample = self.series.levels[0].shape[2] / level.shape[2]

        x0 = max(0, int(xmin / downsample))
        x1 = min(level.shape[2], int(xmax / downsample))
        y0 = max(0, int(ymin / downsample))
        y1 = min(level.shape[1], int(ymax / downsample))

        if x1 <= x0 or y1 <= y0:
            return

        # Skip entirely if nothing has changed since last time
        region = (level_index, x0, x1, y0, y1)
        if region == self._last_region:
            return

        self._updating_background = True
        try:
            self._last_region = region

            # Load Z stack
            raw = level.asarray()[:, y0:y1, x0:x1]

            # Maximum intensity projection over Z or pixk a z pane
            #tile = np.max(raw, axis=0)
            tile = raw[6]
            self.current_tile = tile.astype(np.float32)
            self.apply_contrast()

            self.current_tile = tile.astype(np.float32)
            self.apply_contrast()

            self.dapi_background.set_extent(
                (x0 * downsample, x1 * downsample, y1 * downsample, y0 * downsample)
            )
        finally:
            self._updating_background = False

        self.canvas2.draw_idle()
    def update_contrast(self):
        self.apply_contrast()
    def apply_contrast(self):

        if not hasattr(self, "current_tile"):
            return

        low_percent, high_percent = (
            self.contrast_slider.value()
        )

        if low_percent >= high_percent:
            return

        low = np.percentile(
            self.current_tile,
            low_percent
        )

        high = np.percentile(
            self.current_tile,
            high_percent
        )

        img = np.clip(
            (self.current_tile - low) / (high-low),
            0,
            1
        )

        self.dapi_background.set_data(img)

        self.contrast_label.setText(
            f"Contrast: {low_percent}% - {high_percent}%"
        )

        self.canvas2.draw_idle()
    def toggle_dapi(self, state):

        visible = (
            state == Qt.CheckState.Checked.value
        )

        self.dapi_background.set_visible(visible)

        self.canvas2.draw_idle()
    def create_image_controls(self):

        panel = QWidget()
        layout = QVBoxLayout(panel)


        # H&E toggle
        self.he_checkbox = QCheckBox(
            "Show DAPI"
        )

        self.he_checkbox.stateChanged.connect(
            self.toggle_dapi
        )

        layout.addWidget(
            self.he_checkbox
        )


        # Contrast controls
        contrast_row = QWidget()
        contrast_layout = QHBoxLayout(contrast_row)

        self.contrast_label = QLabel(
            "Contrast: 2%-99%"
        )

        self.contrast_slider = QRangeSlider(
            Qt.Orientation.Horizontal
        )

        self.contrast_slider.setRange(
            0,100
        )

        self.contrast_slider.setValue(
            (2,99)
        )

        self.contrast_slider.valueChanged.connect(
            self.update_contrast
        )

        contrast_layout.addWidget(
            self.contrast_label
        )

        contrast_layout.addWidget(
            self.contrast_slider
        )

        layout.addWidget(
            contrast_row
        )


        return panel
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
            text_color = self.get_text_color(hex_color)

            label.setStyleSheet(
                f"""
                background-color: {hex_color};
                color: {text_color};
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