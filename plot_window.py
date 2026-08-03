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
from functools import partial
from PyQt6 import QtWidgets, QtCore

from PyQt6.QtCore import Qt, QTimer

import cv2

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from superqt import QRangeSlider
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Affine2D
from pathlib import Path
import numpy as np
import dask.array as da
from skimage.transform import resize
import zarr

INTERP = "bicubic"
class LayerContraster(QtWidgets.QWidget):

    def __init__(self, title: str, toggle_conn, contrast_conn, contrast_conn_par, color, initial_checked: bool = False, parent=None):
        super().__init__(parent)

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            """
            LayerContraster {
                border: 1px solid #C0C0C0;
                border-radius: 6px;
                background-color: #FAFAFA;
            }
        """
        )

        # 2. Prevent vertical expansion (Fixed vertical size policy)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        # 1. Main horizontal layout attached to self
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(8)

        # 2. Checkbox on the left
        self.toggle = QtWidgets.QCheckBox(self)
        main_layout.addWidget(
            self.toggle, alignment=QtCore.Qt.AlignmentFlag.AlignVCenter
        )

        # 3. Child vertical layout for text and slider (NO parent passed to constructor)
        child_layout = QtWidgets.QVBoxLayout()
        child_layout.setContentsMargins(0, 0, 0, 0)
        child_layout.setSpacing(2)

        # Title & Contrast status label
        self.label = QtWidgets.QLabel(title, self)
        self.contrast_label = QtWidgets.QLabel("Contrast: 2%-99%", self)

        # Format labels nicely
        self.contrast_label.setStyleSheet("color: gray; font-size: 11px;")

        # Range Slider
        self.rslider = QRangeSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.rslider.setRange(0, 100)
        self.rslider.setValue((2, 99))

        # Hide the default solid bar between handles so the background groove gradient shows
        self.rslider.hideBar()  # or self.rslider.setBarColor("transparent")

        self.rslider.setStyleSheet(
            f"""
            QSlider::groove:horizontal {{
                border: none;
                height: 8px;
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #000000, 
                    stop: 1 {color}
                );
                border-radius: 4px;
            }}

            QSlider::handle:horizontal {{
                background: #FFFFFF;
                border: 1px solid #A0A0A0;
                width: 14px;
                height: 14px;
                margin: -3px 0;
                border-radius: 7px;
            }}
            """
        )

        # Restrain vertical expansion so handles don't blow up in height
        self.rslider.setFixedHeight(20)

        # Add child widgets
        child_layout.addWidget(self.label)
        child_layout.addWidget(self.contrast_label)
        child_layout.addWidget(self.rslider) # type: ignore

        # 4. Correctly nest the vertical layout inside the horizontal main layout
        main_layout.addLayout(child_layout)

        # Signals.
        # NOTE: both callbacks are wired through small wrapper lambdas so
        # that `contrast_conn_par` (the layer key, e.g. "dapi") is always
        # forwarded, regardless of what argument(s) the underlying Qt
        # signal itself emits (the slider's (low, high) tuple, or the
        # checkbox's int state).
        self.rslider.valueChanged.connect(
            lambda _value, key=contrast_conn_par: contrast_conn(key)
        )
        self.toggle.stateChanged.connect(
            lambda state, key=contrast_conn_par: toggle_conn(key, state)
        )
        # Set the initial state without emitting stateChanged: at this
        # point the parent window hasn't appended `self` to
        # image_metadata[key] yet, so a live signal here would make
        # apply_contrast() reach for a widget that isn't registered yet.
        self.toggle.blockSignals(True)
        self.toggle.setChecked(initial_checked)
        self.toggle.blockSignals(False)


class ScatterPlotWindow(QMainWindow):
    """
    Main window with two side-by-side plots:

      - Left  (ax / canvas1):  a UMAP-style scatter plot of cells.
      - Right (ax2 / canvas2): a spatial plot of cell polygons, drawn on
        top of one or more lazily-loaded, single-resolution OME-TIFF
        image layers (H&E, DAPI, boundary, RNA, protein).

    Layout hierarchy:

        QMainWindow
        ├── centralWidget
        │   └── QVBoxLayout
        │       ├── QSplitter (horizontal)
        │       │   ├── left_widget  (toolbar1 + canvas1 = scatter plot)
        │       │   └── right_widget (toolbar2 + canvas2 = spatial plot)
        │       └── legend (QWidget, colored group labels)
        └── control_dock (QDockWidget, right side)
            └── image controls panel (per-layer checkbox + contrast slider)
    """

    def __init__(self, coords, colors, polygons, zf_morph, tf_he, tf_transform):
        super().__init__()
        self.setWindowTitle("Cell Plot")

        # ------------------------------------------------------------
        # State
        # ------------------------------------------------------------
        self.zf_morph = zf_morph
        self.tf_he = tf_he
        self.image_type = "he"

        # Misc image-display state.
        self.image_state = {
            "show_dapi": True,
            "dapi_contrast": (2, 99),
            # If a morphology channel still has a focal Z-stack (shape
            # (Z, H, W) instead of plain (H, W)), this is which plane we
            # display. Matches the original single-image pipeline.
            "z_plane": 6,
        }
        # color, image (AxesImage), widget (LayerContraster)
        self.image_metadata = {
            "he": [None],
            "dapi": ["#0f73e6"],
            "cbm": ["#f300a5"],
            "rna": ["#a4a400"],
            "prot": ["#008a00"]
        }

        # ------------------------------------------------------------
        # Lazy, chunked views onto the on-disk OME-TIFFs. Wrapping them
        # with zarr means indexing/slicing only reads the tiles on disk
        # that overlap the requested region, instead of pulling the
        # whole image into memory the way `tifffile_obj.asarray()` would.
        #
        # tf_he: RGB, and (unlike the morphology channels) still grouped
        # by resolution level, e.g. levels[0] = full res, levels[1] =
        # half res, etc. - same idea as the original pyramid used for
        # the morphology image. We keep the whole level list around and
        # pick the coarsest level that still covers the viewport at
        # >=1 image pixel per screen pixel, so we never have to touch
        # the full-resolution array just to render a zoomed-out view.
        # Assumes each level's array is channel-first, (3, H, W) -
        # adjust `_read_tile` below if yours is (H, W, 3) instead.
        #
        # tf_morphology: a list of four single-channel, single-resolution
        # OME-TIFFs, one per stain -> [dapi_tf, cbm_tf, rna_tf, prot_tf].
        # IMPORTANT: "single-channel" doesn't necessarily mean each array
        # is flat (H, W) - if it still carries a focal Z-stack, its shape
        # is (Z, H, W), and reading `[:, y0:y1, x0:x1]` (all Z) instead of
        # `[z_plane, y0:y1, x0:x1]` (one Z) is exactly the kind of
        # accidental multi-x memory blowup you just hit. `_read_tile`
        # below now only ever pulls a single plane for grayscale sources.
        # ADJUST the order below if your list isn't in this order.
        # ------------------------------------------------------------
        self.he_levels = self._open_zarr_levels(self.tf_he)
        self.morphology_levels = [
            self._open_zarr_levels(channel_path) for channel_path in self.zf_morph
        ]

        # layer key -> (list of resolution levels, kind). Non-pyramidal
        # layers just get a one-element list, so the same level-picking
        # logic in `update_background` works for every layer. `kind` is
        # "rgb" (channel-first, take all channels) or "gray" (take a
        # single Z-plane if 3D, or the array as-is if already 2D).
        self.image_sources = {
            "he":   (self.he_levels, "rgb"),
            "dapi": (self.morphology_levels[0], "gray"),
            "cbm":  (self.morphology_levels[1], "gray"),
            "rna":  (self.morphology_levels[2], "gray"),
            "prot": (self.morphology_levels[3], "gray"),
        }

        self.he_transform_matrix = np.asarray(tf_transform, dtype=np.float64)

        # Per-layer cache of the last-loaded viewport crop (pre-contrast),
        # so the contrast slider can re-apply without re-reading from disk.
        self.current_tiles = {}

        # Tracks the last-rendered region per layer, (x0, x1, y0, y1), so
        # we can skip redundant reloads.
        self._last_regions = {}

        # ------------------------------------------------------------
        # Build UI, piece by piece
        # ------------------------------------------------------------
        self._create_scatter_plot(coords, colors)
        self._create_spatial_plot(polygons)
        self._create_toolbars()
        self._create_dock_controls()
        self._create_legend()
        self._create_central_layout()

        # Draw the initial background tile(s) once the window has a real size.
        self.showMaximized()
        QTimer.singleShot(100, self.update_background)
    @staticmethod
    def _open_zarr_levels(source) -> list:
        """
        Accepts:
        - Path / str to a channel folder containing `level_*.zarr` subfolders
        - List of open Zarr stores / paths
        - Opened zarr.Group or zarr.Array
        - tifffile object
        Returns a list of zarr Arrays sorted from level 0 (highest res) to lowest res.
        """
        # 1. Path/str pointing to a channel directory containing level stores
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_dir():
                # Find all level_*.zarr directories inside this channel folder
                level_paths = sorted(path.glob("level_*.zarr"))
                if level_paths:
                    source = level_paths  # Hand off to list processing below

        # 2. Processing a list of level paths or open Zarr stores
        if isinstance(source, (list, tuple)):
            levels = []
            for item in source:
                node = zarr.open(item, mode="r") if not isinstance(item, (zarr.Array, zarr.Group)) else item
                if isinstance(node, zarr.Group):
                    first_key = next(iter(node.array_keys()))
                    levels.append(node[first_key])
                else:
                    levels.append(node)
                    
            # Sort descending by width (shape[-1]) so Index 0 is always Full Resolution
            levels.sort(key=lambda arr: arr.shape[-1], reverse=True)
            return levels

        # 3. Handling a single tifffile instance
        if hasattr(source, "aszarr"):
            node = zarr.open(source.aszarr(), mode="r")# type: ignore
            if isinstance(node, zarr.Array):
                return [node]
            levels = [node[key] for key in node.array_keys()]
            levels.sort(key=lambda arr: arr.shape[-1], reverse=True)# type: ignore
            return levels

        # 4. Fallback for single open Zarr stores or groups
        node = zarr.open(source, mode="r") if not isinstance(source, (zarr.Array, zarr.Group)) else source
        if isinstance(node, zarr.Array):
            return [node]
        levels = [node[key] for key in node.array_keys()]
        levels.sort(key=lambda arr: arr.shape[-1], reverse=True)# type: ignore
        return levels

    def _select_pyramid_level(self, levels, viewport_width):
        """
        Select the coarsest pyramid level that still has at least
        one pyramid pixel per screen pixel.
        """
        if len(levels) == 1:
            return 0

        dpr = self.canvas2.devicePixelRatioF()
        screen_width = max(self.canvas2.width() * dpr, 1)

        pixels_per_screen_pixel = viewport_width / screen_width

        full_width = levels[0].shape[-1]

        selected = 0

        for i, level in enumerate(levels):
            downsample = full_width / level.shape[-1]

            if downsample <= pixels_per_screen_pixel:
                selected = i
            else:
                break

        return selected
    def _read_tile(self, level, y0, y1, x0, x1, kind):
        """
        Reads the [y0:y1, x0:x1] crop out of a resolution level, only
        touching the on-disk chunks that overlap that window.

        - "rgb" (H&E): assumes channel-first (3, H, W); reads all 3
          channels and moves the channel axis last so matplotlib gets
          (H, W, 3). If your H&E levels are actually channel-last
          (H, W, 3), swap this branch to `level[y0:y1, x0:x1, :]`.
        - "gray" (DAPI/boundary/RNA/protein): if the array is already
          2D, (H, W), reads it directly. If it's 3D, (Z, H, W) - a focal
          stack rather than color channels - reads exactly ONE plane
          (`image_state["z_plane"]`), not the whole stack.
        """
        if level.ndim == 2:
            return level[y0:y1, x0:x1]

        if kind == "rgb":
            return np.moveaxis(level[:, y0:y1, x0:x1], 0, -1)

        # "gray" with a leading Z axis: pick a single focal plane.
        z = min(self.image_state.get("z_plane", 0), level.shape[0] - 1)
        return level[z, y0:y1, x0:x1]

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
        Right panel: cell polygons drawn over the image layers.

        Sets up:
          - the background images (`self.image_metadata[key][1]`), imshow
            placeholders whose data/extent get filled in lazily by
            `update_background()` as the user pans/zooms.
          - the polygon overlay (`self.poly_collection`).
        """
        self.figure2 = Figure()
        self.canvas2 = FigureCanvasQTAgg(self.figure2)
        self.ax2 = self.figure2.add_subplot(111)

        # Placeholder background images; real data is loaded on demand.
        for key in self.image_metadata:
            if key == "he":
                self.image_metadata[key].append(
                    self.ax2.imshow(
                        np.zeros((10, 10, 3), dtype=np.float32),  # Shape: (H, W, 3) with values [0..1] float or [0..255] uint8
                        extent=(0, 1, 0, 1),
                        origin="upper",
                        interpolation=INTERP
                    )
                )
            else:
                self.image_metadata[key].append(
                    self.ax2.imshow(
                        np.zeros((10, 10), dtype=np.float32),
                        cmap=LinearSegmentedColormap.from_list(f"{key}_cmap", ["#000000",self.image_metadata[key][0]]),
                        vmin=0,
                        vmax=1,
                        extent=(0, 1, 0, 1),
                        origin="upper",
                        interpolation=INTERP
                    )
                )
                self.image_metadata[key][1].set_visible(False)

        self.composite_artist = self.ax2.imshow(
            np.zeros((10, 10, 3), dtype=np.float32),  # Shape: (H, W, 3) with values [0..1] float or [0..255] uint8
            extent=(0, 1, 0, 1),
            origin="upper",
            interpolation=INTERP
        )
        self.composite_artist.set_visible(True)
        # Reload the background tile(s) whenever the view changes (pan/zoom
        # via mouse release, or any toolbar-triggered redraw).
        self.canvas2.mpl_connect("button_release_event", self.update_background)

        # Cell polygon overlay. `mpp` = microns per pixel, used to convert
        # polygon coordinates (microns) into pixel space.
        mpp = 0.21249222  # TODO: read this from image metadata instead of hardcoding
        polygons_px = [poly / mpp for poly in polygons]
        patches = [Polygon(poly, closed=True) for poly in polygons_px]

        self.poly_collection = PatchCollection(patches, linewidths=0, alpha=1)
        self.ax2.add_collection(self.poly_collection)  # type: ignore

        # Start the view on the cells themselves, not the whole slide -
        # a full-slide starting viewport is what was forcing every layer
        # to load a huge crop (or the whole image, for non-pyramidal
        # layers) the moment it was toggled on.
        all_pts = np.concatenate(polygons_px, axis=0)
        px_min, py_min = all_pts.min(axis=0)
        px_max, py_max = all_pts.max(axis=0)
        pad_x = (px_max - px_min) * 0.05
        pad_y = (py_max - py_min) * 0.05
        self.ax2.set_xlim(px_min - pad_x, px_max + pad_x)
        # Inverted y-axis to match image pixel coordinates (origin top-left).
        self.ax2.set_ylim(py_max + pad_y, py_min - pad_y)
        self.ax2.set_aspect("equal")
        self.ax2.callbacks.connect("xlim_changed", self.update_background) # type: ignore

    def _create_toolbars(self):
        """Matplotlib navigation toolbars (pan/zoom/save) for each plot."""
        self.toolbar1 = NavigationToolbar2QT(self.canvas1, self)
        self.toolbar2 = NavigationToolbar2QT(self.canvas2, self)

    def _create_dock_controls(self):
        """Right-hand dockable panel with display controls (per-layer toggle + contrast)."""
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
        """Builds the panel shown in the right dock: per-layer toggle + contrast slider."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.heW = LayerContraster("H&E", self.toggle_layer, self.apply_contrast, "he" , "#000000", initial_checked=False)
        self.dapiW = LayerContraster("DAPI", self.toggle_layer, self.apply_contrast, "dapi", "#0f73e6", initial_checked=False)
        self.cbmW = LayerContraster("Boundary", self.toggle_layer, self.apply_contrast, "cbm", "#f300a5", initial_checked=False)
        self.rnaW = LayerContraster("RNA Stain", self.toggle_layer, self.apply_contrast, "rna", "#a4a400", initial_checked=False)
        self.protW = LayerContraster("Protein Stain", self.toggle_layer, self.apply_contrast, "prot", "#008a00", initial_checked=False)

        cbToggler = QHBoxLayout()
        self.toggleCB = QCheckBox()
        self.cbToggler_label = QLabel()
        self.toggleCB.setChecked(True)
        self.cbToggler_label.setText("View Annotations")

        cbToggler.addWidget(self.toggleCB)
        cbToggler.addWidget(self.cbToggler_label)

        self.toggleCB.stateChanged.connect(self.toggle_cells)

        self.image_metadata["he"].append(self.heW)
        self.image_metadata["dapi"].append(self.dapiW)
        self.image_metadata["cbm"].append(self.cbmW)
        self.image_metadata["rna"].append(self.rnaW)
        self.image_metadata["prot"].append(self.protW)

        layout.addWidget(self.heW)
        layout.addWidget(self.dapiW)
        layout.addWidget(self.cbmW)
        layout.addWidget(self.rnaW)
        layout.addWidget(self.protW)

        layout.addLayout(cbToggler)

        layout.addStretch()

        return panel

    # ==================================================================
    # BACKGROUND TILE LOADING (right/spatial plot)
    # ==================================================================
    #
    # `_select_pyramid_level` restores the original level-picking logic
    # for whichever layers actually have a resolution pyramid (H&E).
    # Layers with a single level (the morphology channels) just always
    # get level 0. Either way, only the crop overlapping the current
    # viewport is ever read off disk.
    def _load_tile_for_layer(self, layer_key):
        levels, kind = self.image_sources[layer_key]

        x_lo, x_hi = self.ax2.get_xlim()
        xmin, xmax = min(x_lo, x_hi), max(x_lo, x_hi)
        y_lo, y_hi = self.ax2.get_ylim()
        ymin, ymax = min(y_lo, y_hi), max(y_lo, y_hi)

        if layer_key == "he":
            # xmin/xmax/ymin/ymax above are in DAPI/display space. HE lives
            # in its own pixel space related to display space by
            # he_transform_matrix, so map the viewport corners back through
            # the inverse affine to know which HE pixels to actually read -
            # using display-space coords directly here was the bug: it read
            # an arbitrary small/offset patch of the HE pyramid instead of
            # the patch that actually corresponds to the current view.
            inv = cv2.invertAffineTransform(self.he_transform_matrix.astype(np.float32))
            corners = np.array(
                [[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]],
                dtype=np.float32,
            ).reshape(-1, 1, 2)
            he_corners = cv2.transform(corners, inv).reshape(-1, 2)
            crop_xmin, crop_ymin = he_corners.min(axis=0)
            crop_xmax, crop_ymax = he_corners.max(axis=0)
        else:
            crop_xmin, crop_xmax, crop_ymin, crop_ymax = xmin, xmax, ymin, ymax

        if layer_key == "he":
            inv = cv2.invertAffineTransform(self.he_transform_matrix.astype(np.float32))
            corners = np.array(
                [[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]],
                dtype=np.float32,
            ).reshape(-1, 1, 2)
            he_corners = cv2.transform(corners, inv).reshape(-1, 2)
            crop_xmin, crop_ymin = he_corners.min(axis=0)
            crop_xmax, crop_ymax = he_corners.max(axis=0)
            print("display viewport (xmin,xmax,ymin,ymax):", xmin, xmax, ymin, ymax)
            print("HE crop box (xmin,xmax,ymin,ymax):", crop_xmin, crop_xmax, crop_ymin, crop_ymax)
            print("HE crop size:", crop_xmax - crop_xmin, crop_ymax - crop_ymin)
    
        level_index = self._select_pyramid_level(levels, crop_xmax - crop_xmin)
        level = levels[level_index]
        print(
            f"HE viewport width={crop_xmax-crop_xmin:.1f}, "
            f"selected level={level_index}, "
            f"level shape={level.shape}"
        )

        full_width = levels[0].shape[-1]
        level_height, level_width = level.shape[-2], level.shape[-1]
        downsample = full_width / level_width

        x0 = max(0, int(crop_xmin / downsample))
        x1 = min(level_width, int(np.ceil(crop_xmax / downsample)))
        y0 = max(0, int(crop_ymin / downsample))
        y1 = min(level_height, int(np.ceil(crop_ymax / downsample)))

        if x1 <= x0 or y1 <= y0:
            return None

        region = (level_index, x0, x1, y0, y1)
        if self._last_regions.get(layer_key) == region and layer_key in self.current_tiles:
            return self.current_tiles[layer_key]
        self._last_regions[layer_key] = region

        tile = self._read_tile(level, y0, y1, x0, x1, kind)
        self.current_tiles[layer_key] = np.asarray(tile).astype(np.float32)

        self.image_metadata[layer_key][1].set_extent(
            (x0 * downsample, x1 * downsample, y1 * downsample, y0 * downsample)
        )
        if layer_key == "he":
            self.image_metadata[layer_key][1].set_transform(self._he_display_transform())
        else:
            self.image_metadata[layer_key][1].set_transform(self.ax2.transData)
        self.apply_contrast(layer_key, redraw=False)
        return self.current_tiles[layer_key]
    def update_background(self, event=None):
        """
        Refreshes the tile data for all active layers when panning or zooming.
        """
        if getattr(self, "_updating_background", False):
            return

        self._updating_background = True

        try:
            # Check if H&E is active
            he_visible = self.image_metadata["he"][2].toggle.isChecked()

            if he_visible:
                # Load H&E tile
                self._load_tile_for_layer("he")
                if hasattr(self, "composite_artist"):
                    self.composite_artist.set_visible(False)
            else:
                # Load all active morphology tiles and combine them
                active_layers = [
                    key for key in ["dapi", "cbm", "rna", "prot"]
                    if self.image_metadata[key][2].toggle.isChecked()
                ]
                for key in active_layers:
                    self._load_tile_for_layer(key)
                
                self.combine_images()

        finally:
            self._updating_background = False

        self.canvas2.draw_idle()

    # ==================================================================
    # DISPLAY CONTROLS (per-layer toggle + contrast)
    # ==================================================================
    def toggle_cells(self):
        visible = self.toggleCB.isChecked()
        self.poly_collection.set_visible(visible)

        if self.image_metadata["he"][2].toggle.isChecked():
            self.apply_contrast("he", redraw=False)
        else:
            self.combine_images()

        self.canvas2.draw_idle()

    def toggle_layer(self, layer_key, state):
        """Show/hide background layers with mutual exclusion between H&E and morphology."""
        visible = state == Qt.CheckState.Checked.value

        if layer_key == "he":
            # 1. Update H&E visibility explicitly
            self.image_metadata["he"][1].set_visible(visible)

            # 2. If H&E is checked, uncheck & disable all morphology checkboxes
            if visible:
                for key in ["dapi", "cbm", "rna", "prot"]:
                    widget = self.image_metadata[key][2]
                    widget.toggle.blockSignals(True)
                    widget.toggle.setChecked(False)
                    widget.toggle.blockSignals(False)
                    widget.toggle.setEnabled(False)
                    self.image_metadata[key][1].set_visible(False)
            else:
                # If H&E is unchecked, re-enable morphology controls
                for key in ["dapi", "cbm", "rna", "prot"]:
                    self.image_metadata[key][2].toggle.setEnabled(True)

        else:
            # Morphology channels (DAPI, CBM, RNA, Prot)
            # 1. Disable H&E checkbox if ANY morphology channel is checked
            any_morph_checked = any(
                self.image_metadata[k][2].toggle.isChecked()
                for k in ["dapi", "cbm", "rna", "prot"]
            )
            self.heW.toggle.setEnabled(not any_morph_checked)

        # 3. Handle data loading & redraws for both checking AND unchecking
        if visible:
            self._last_regions.pop(layer_key, None)
            self.update_background()

        # Re-blend or redraw regardless of checking/unchecking
        if layer_key != "he":

            self.combine_images()
        else:
            self.canvas2.draw_idle()

    def apply_contrast(self, layer_key, redraw=True, overide = False):
        tile = self.current_tiles.get(layer_key)

        if tile is None:
            return

        widget = self.image_metadata[layer_key][2]
        low_percent, high_percent = widget.rslider.value()

        if low_percent >= high_percent:
            return

        low = np.percentile(tile, low_percent)
        high = np.percentile(tile, high_percent)

        if high <= low:
            return

        # Convert only the final display image to float32
        img = tile.astype(np.float32)
        img -= low
        img /= (high - low)
        np.clip(img, 0, 1, out=img)

        # Update layer's Matplotlib artist
        if layer_key == "he" and self.toggleCB.isChecked():
            # Convert H&E to grayscale when annotations are visible
            gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
            img = np.stack([gray, gray, gray], axis=-1)

        self.image_metadata[layer_key][1].set_data(img)

        widget.contrast_label.setText(
            f"Contrast: {low_percent}% - {high_percent}%"
        )

        # CRITICAL: If a morphology layer slider is moved, trigger composite re-blend!
        if layer_key != "he" and redraw:
            self.combine_images()
        elif redraw:
            self.canvas2.draw_idle()
        if layer_key == "he":
            self.image_metadata[layer_key][1].set_transform(self._he_display_transform())
            print("HE transform matrix:\n", self.image_metadata[layer_key][1].get_transform().get_matrix())
    def combine_images(self):
        # 1. Gather all active morphology layers
        active_layers = [
            key for key in ["dapi", "cbm", "rna", "prot"]
            if self.image_metadata[key][2].toggle.isChecked()
        ]

        # Hide composite if no morphology layers are checked
        if not active_layers:
            if hasattr(self, "composite_artist"):
                self.composite_artist.set_visible(False)
            self.canvas2.draw_idle()
            return

        # 2. Make sure all active tiles are loaded into memory and contrasted
        for key in active_layers:
            if key not in self.current_tiles or self.current_tiles[key] is None:
                self._load_tile_for_layer(key)

        # 3. Reference shape from first active layer artist
        ref_artist = self.image_metadata[active_layers[0]][1]
        ref_data = ref_artist.get_array()
        if ref_data is None or ref_data.shape == (10, 10):
            # If still placeholder size, force tile load again
            self._load_tile_for_layer(active_layers[0])
            ref_data = ref_artist.get_array()

        h, w = ref_data.shape[:2]
        bg = np.zeros((h, w, 3), dtype=np.float32)

        # 4. Perform Screen Blend mode math
        for key in active_layers:
            artist = self.image_metadata[key][1]
            fg_raw = artist.get_array()
            
            if fg_raw is None:
                continue
                
            fg_float = fg_raw.astype(np.float32)

            if fg_float.ndim == 2:
                cmap = artist.get_cmap()
                fg_rgb = cmap(fg_float)[..., :3]
            else:
                fg_rgb = fg_float

            # Screen Blend
            bg = 1.0 - (1.0 - bg) * (1.0 - fg_rgb)

        bg = np.clip(bg, 0.0, 1.0)
        if self.toggleCB.isChecked():
            gray = np.dot(bg[..., :3], [0.2989, 0.5870, 0.1140])
            bg = np.stack([gray, gray, gray], axis=-1).astype(bg.dtype)
        # 5. Display the composite artist
        if not hasattr(self, "composite_artist") or self.composite_artist is None:
            self.composite_artist = self.ax2.imshow(
                bg,
                extent=ref_artist.get_extent(),
                origin="upper",
                interpolation=INTERP,
                zorder=2
            )
        else:
            self.composite_artist.set_data(bg)
            self.composite_artist.set_extent(ref_artist.get_extent())
            self.composite_artist.set_visible(True)

        self.canvas2.draw_idle()
    def _he_display_transform(self):
        """
        Builds the Matplotlib Affine2D that maps HE pixel space -> display
        (DAPI) space, from the raw cv2 2x3 affine matrix.

        cv2 convention:   x' = a*x + b*y + tx ; y' = c*x + d*y + ty
            M = [[a, b, tx],
                [c, d, ty]]

        Matplotlib's Affine2D.from_values(A, B, C, D, E, F) convention:
            x' = A*x + C*y + E ; y' = B*x + D*y + F

        So the b/c terms swap position when handed to from_values - this
        is the whole reason a straight scale-only decomposition (the old
        decompose_affine_matrix) silently dropped rotation/shear: it threw
        away exactly the b/c terms being preserved here.
        """
        a, b, tx = self.he_transform_matrix[0]
        c, d, ty = self.he_transform_matrix[1]
        return Affine2D.from_values(a, c, b, d, tx, ty) + self.ax2.transData

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