from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QLabel,
    QCheckBox,
    QDockWidget
)

from PyQt6 import QtWidgets

from PyQt6.QtCore import Qt, QTimer

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from superqt import QRangeSlider
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

import numpy as np

# Pyramid level -> (num_z_planes, width, height), for reference:
# 0 (12, 28048, 46543)
# 1 (12, 14024, 23271)
# 2 (12, 7012, 11635)
# 3 (12, 3506, 5817)
# 4 (12, 1753, 2908)
# 5 (12, 876, 1454)
# 6 (12, 438, 727)
# 7 (12, 219, 363)

class MyWidgetGroup(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout for the group template
        main_layout = QtWidgets.QHBoxLayout(self)
        child_layout = QtWidgets.QVBoxLayout(self)

        # Add group components
        self.label = QtWidgets.QLabel("Template Title", self)
        self.rslider = QtWidgets.
        
        layout.addWidget(self.label)
        layout.addWidget(self.input_field)
        layout.addWidget(self.button)

class ScatterPlotWindow(QMainWindow):
    """
    Main window with two side-by-side plots:

      - Left  (ax / canvas1):  a UMAP-style scatter plot of cells.
      - Right (ax2 / canvas2): a spatial plot of cell polygons, drawn on
        top of a lazily-loaded DAPI image pyramid tile (loaded from
        `tf`/`series`, a tifffile-style pyramidal image).

    Layout hierarchy:

        QMainWindow
        ├── centralWidget
        │   └── QVBoxLayout
        │       ├── QSplitter (horizontal)
        │       │   ├── left_widget  (toolbar1 + canvas1 = scatter plot)
        │       │   └── right_widget (toolbar2 + canvas2 = spatial plot)
        │       └── legend (QWidget, colored group labels)
        └── control_dock (QDockWidget, right side)
            └── image controls panel (DAPI checkbox + contrast slider)
    """

    def __init__(self, coords, colors, polygons, tf):
        super().__init__()
        self.setWindowTitle("Cell Plot")
        self.showFullScreen()

        # ------------------------------------------------------------
        # State
        # ------------------------------------------------------------
        self.tf = tf
        self.series = self.tf.series[0]
        self.image_type = "he"

        # Misc image-display state (contrast, projection mode, etc.)
        self.image_state = {
            "show_dapi": True,
            "dapi_contrast": (2, 99),
            "projection": "single",
            "z_plane": 6,
        }

        # Tracks the last-rendered background region so we can skip
        # redundant redraws (level_index, x0, x1, y0, y1).
        self._last_region = None

        # ------------------------------------------------------------
        # Build UI, piece by piece
        # ------------------------------------------------------------
        self._create_scatter_plot(coords, colors)
        self._create_spatial_plot(polygons)
        self._create_toolbars()
        self._create_dock_controls()
        self._create_legend()
        self._create_central_layout()

        # Draw the initial background tile once the window has a real size.
        QTimer.singleShot(100, self.update_background)

    # ==================================================================
    # UI CONSTRUCTION
    # ==================================================================

    def _create_scatter_plot(self, coords, colors):
        """Left panel: matplotlib scatter plot (e.g. UMAP embedding)."""
        self.figure1 = Figure()
        self.canvas1 = FigureCanvasQTAgg(self.figure1)
        self.canvas = self.canvas1  # kept for backwards-compatible access
        self.ax = self.figure1.add_subplot(111)

        self.scatter = self.ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=colors,
            s=0.5,
        )

    def _create_spatial_plot(self, polygons):
        """
        Right panel: cell polygons drawn over a DAPI image tile.

        Sets up:
          - the background image (`self.dapi_background`), an imshow
            placeholder that gets its data/extent filled in lazily by
            `update_background()` as the user pans/zooms.
          - the polygon overlay (`self.poly_collection`).
        """
        self.figure2 = Figure()
        self.canvas2 = FigureCanvasQTAgg(self.figure2)
        self.ax2 = self.figure2.add_subplot(111)

        # Placeholder background image; real data is loaded on demand.
        self.dapi_background = self.ax2.imshow(
            np.zeros((10, 10), dtype=np.float32),
            cmap="Greys",
            vmin=0,
            vmax=1,
            extent=(0, 1, 0, 1),
            origin="upper",
            interpolation="nearest",
            zorder=0,
        )
        self.dapi_background.set_visible(False)

        # Reload the background tile whenever the view changes (pan/zoom
        # via mouse release, or any toolbar-triggered redraw).
        self.canvas2.mpl_connect("button_release_event", self.update_background)
        self.canvas2.mpl_connect("draw_event", self.update_background)

        # Cell polygon overlay. `mpp` = microns per pixel, used to convert
        # polygon coordinates (microns) into pixel space.
        mpp = 0.21249222  # TODO: read this from image metadata instead of hardcoding
        patches = [Polygon(poly / mpp, closed=True) for poly in polygons]

        self.poly_collection = PatchCollection(patches, linewidths=0, alpha=1)
        self.ax2.add_collection(self.poly_collection)  # type: ignore

        # Orient axes to match image pixel coordinates (origin top-left).
        self.ax2.set_xlim(0, self.series.levels[0].shape[2])
        self.ax2.set_ylim(self.series.levels[0].shape[1], 0)
        self.ax2.set_aspect("equal")

    def _create_toolbars(self):
        """Matplotlib navigation toolbars (pan/zoom/save) for each plot."""
        self.toolbar1 = NavigationToolbar2QT(self.canvas1, self)
        self.toolbar2 = NavigationToolbar2QT(self.canvas2, self)

    def _create_dock_controls(self):
        """Right-hand dockable panel with display controls (DAPI toggle, contrast)."""
        self.control_dock = QDockWidget("Display", self)
        self.control_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )

        self.image_controls = self.create_image_controls()
        self.control_dock.setWidget(self.image_controls)

        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self.control_dock,
        )

    def _create_legend(self):
        """Placeholder legend widget; populated later via `update_legend()`."""
        self.legend = QWidget()
        self.legend.setMaximumHeight(100)

    def _create_central_layout(self):
        """
        Assembles the central widget:

            QVBoxLayout
            ├── QSplitter (left plot | right plot)
            └── legend
        """
        # Left side: scatter plot + its toolbar.
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(self.toolbar1)
        left_layout.addWidget(self.canvas1)

        # Right side: spatial plot + its toolbar.
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(self.toolbar2)
        right_layout.addWidget(self.canvas2)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.splitter, stretch=1)
        layout.addWidget(self.legend, stretch=0)

        self.setCentralWidget(container)

    def create_image_controls(self):
        """Builds the panel shown in the right dock: DAPI toggle + contrast slider."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # --- DAPI visibility toggle ---
        self.he_checkbox = QCheckBox("Show DAPI")
        self.he_checkbox.stateChanged.connect(self.toggle_dapi)
        layout.addWidget(self.he_checkbox)

        # --- Contrast range slider ---
        contrast_row = QWidget()
        contrast_layout = QHBoxLayout(contrast_row)

        self.contrast_label = QLabel("Contrast: 2%-99%")
        self.contrast_slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(0, 100)
        self.contrast_slider.setValue((2, 99))
        self.contrast_slider.valueChanged.connect(self.update_contrast)

        contrast_layout.addWidget(self.contrast_label)
        contrast_layout.addWidget(self.contrast_slider)
        layout.addWidget(contrast_row)

        return panel

    # ==================================================================
    # BACKGROUND TILE LOADING (right/spatial plot)
    # ==================================================================

    def compute_scale(self, xmin, xmax):
        """
        Picks the most appropriate pyramid level for the current viewport,
        based on how many image pixels map to each screen pixel.
        """
        viewport_width = xmax - xmin
        dpr = self.canvas2.devicePixelRatioF()
        screen_width = max(self.canvas2.width() * dpr, 1)

        pixels_per_screen_pixel = viewport_width / screen_width
        full_width = self.series.levels[0].shape[2]

        for i, level in enumerate(self.series.levels):
            level_width = level.shape[2]
            downsample = full_width / level_width
            if downsample >= pixels_per_screen_pixel:
                return i

        return len(self.series.levels) - 1

    def update_background(self, event=None):
        """
        Loads (or reuses) the DAPI image tile matching the current
        pan/zoom viewport of the spatial plot, and refreshes the display.

        Triggered on pan/zoom ("button_release_event"/"draw_event") and
        once at startup via QTimer.
        """
        # Guard against re-entrancy triggered by our own canvas redraws.
        if getattr(self, "_updating_background", False):
            return

        x_lo, x_hi = self.ax2.get_xlim()
        xmin, xmax = min(x_lo, x_hi), max(x_lo, x_hi)

        y_lo, y_hi = self.ax2.get_ylim()
        ymin, ymax = min(y_lo, y_hi), max(y_lo, y_hi)

        level_index = self.compute_scale(xmin, xmax)
        level = self.series.levels[level_index]
        downsample = self.series.levels[0].shape[2] / level.shape[2]

        # Convert full-resolution viewport bounds into this level's pixel space.
        x0 = max(0, int(xmin / downsample))
        x1 = min(level.shape[2], int(xmax / downsample))
        y0 = max(0, int(ymin / downsample))
        y1 = min(level.shape[1], int(ymax / downsample))

        if x1 <= x0 or y1 <= y0:
            return

        # Skip the (potentially expensive) reload if nothing has changed.
        region = (level_index, x0, x1, y0, y1)
        if region == self._last_region:
            return

        self._updating_background = True
        try:
            self._last_region = region

            # Load just the visible crop of the Z stack for this level.
            raw = level.asarray()[:, y0:y1, x0:x1]

            # Pick a single Z-plane. (Alternative: max-intensity projection
            # via `np.max(raw, axis=0)`.)
            tile = raw[self.image_state["z_plane"]]

            self.current_tile = tile.astype(np.float32)
            self.apply_contrast()

            self.dapi_background.set_extent(
                (x0 * downsample, x1 * downsample, y1 * downsample, y0 * downsample)
            )
        finally:
            self._updating_background = False

        self.canvas2.draw_idle()

    # ==================================================================
    # DISPLAY CONTROLS (DAPI toggle + contrast)
    # ==================================================================

    def toggle_dapi(self, state):
        """Show/hide the DAPI background image."""
        visible = state == Qt.CheckState.Checked.value
        self.dapi_background.set_visible(visible)
        self.canvas2.draw_idle()

    def update_contrast(self):
        """Slot for the contrast slider; re-applies contrast to the current tile."""
        self.apply_contrast()

    def apply_contrast(self):
        """
        Rescales `self.current_tile` intensities to [0, 1] using the
        low/high percentiles selected on the contrast slider, and pushes
        the result into the background image.
        """
        if not hasattr(self, "current_tile"):
            return

        low_percent, high_percent = self.contrast_slider.value()
        if low_percent >= high_percent:
            return

        low = np.percentile(self.current_tile, low_percent)
        high = np.percentile(self.current_tile, high_percent)

        img = np.clip((self.current_tile - low) / (high - low), 0, 1)
        self.dapi_background.set_data(img)

        self.contrast_label.setText(f"Contrast: {low_percent}% - {high_percent}%")
        self.canvas2.draw_idle()

    # ==================================================================
    # LEGEND
    # ==================================================================

    def get_text_color(self, hex_color):
        """Returns 'black' or 'white', whichever is more readable on `hex_color`."""
        import matplotlib.colors as mcolors

        r, g, b = mcolors.to_rgb(hex_color)
        r, g, b = r * 255, g * 255, b * 255

        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        return "white" if brightness < 140 else "black"

    def update_legend(self, color_map):
        """
        Rebuilds the legend row beneath the plots.

        Args:
            color_map: dict of group_name -> color, one colored chip per entry.
        """
        import matplotlib.colors as mcolors

        # Remove the old legend widget and swap in a fresh one.
        if self.legend is not None:
            self.legend.deleteLater()

        self.legend = QWidget()
        legend_layout = QHBoxLayout(self.legend)
        legend_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        legend_layout.setSpacing(5)

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

            # Size the chip to fit its text rather than stretching.
            label.setSizePolicy(
                label.sizePolicy().horizontalPolicy().Fixed,
                label.sizePolicy().verticalPolicy().Fixed,
            )

            legend_layout.addWidget(label)

        self.centralWidget().layout().addWidget(self.legend)  # type: ignore