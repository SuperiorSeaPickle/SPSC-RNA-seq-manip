import sys
import pandas as pd

from PyQt6.QtWidgets import (
    QApplication,
    QSlider,
    QLabel,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QInputDialog,
    QFileDialog,
    QMenu,
    QMainWindow
)

from PyQt6.QtGui import QColor, QAction

from PyQt6.QtCore import pyqtSignal, Qt, QTimer, QEvent
import pickle

class DFE_data():
    def __init__(self,df: pd.DataFrame, threshes: dict) -> None:
        self.df = df
        self.col_meta = {}
        self.col_meta['thresholds'] = threshes

    def save(self, file_path):
        print("Saving:")
        print(self.df)

        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    def load(self, file_path):
        with open(file_path, "rb") as file:
            loaded = pickle.load(file)

        print(type(loaded))
        print(loaded)

        self.df = loaded.df
        self.col_meta = loaded.col_meta

class DataFrameEditor(QDialog):
    """
    Simple popup window for editing a pandas DataFrame.
    """
    updateFigs = pyqtSignal(tuple, object, dict, bool)

    def __init__(self, df, valid_genes, extr_info, thresh):
        super().__init__()

        
        # Save a copy of the dataframe.
        # We edit the copy so Cancel leaves the original untouched.

        self.df = df.copy()
        self.valid_genes = valid_genes
        self.extr_info = extr_info
        self.thresh = thresh

        self.column_thresholds = {
            str(col): thresh for col in self.df.columns
        }

        self.active_column = None

        # Window settings

        self.setWindowTitle("DataFrame Editor")
        self.resize(800, 500)

        self.result_df = None

        self.pressed_keys = set()
        # Main vertical layout

        layout = QVBoxLayout(self)

        #slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(-300)
        self.slider.setMaximum(300)
        self.slider.setValue(int(round(self.thresh*100,0)))
        self.slider.setSingleStep(1)

        self.label = QLabel("Expression Threshold: 0.80")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._slider_timer = QTimer(self)
        self._slider_timer.setSingleShot(True)
        self._slider_timer.setInterval(150)  # ms after last movement
        self.slider.valueChanged.connect(self.update_slabel)
        self.slider.valueChanged.connect(lambda v: self._slider_timer.start())
        self._slider_timer.timeout.connect(
            lambda: self.updateFigs.emit(self.extr_info, self, self.column_thresholds, False)
        )
        self.slider.valueChanged.connect(self.slider_changed)
        # Spreadsheet widget

        self.table = QTableWidget()
        layout.addWidget(self.table)

        # Load dataframe into spreadsheet

        self.table.blockSignals(True)
        self.load_dataframe()
        self.table.blockSignals(False)

        # Update colors whenever a user changes a cell
        self.table.itemChanged.connect(self.make_caps)
        self.table.itemChanged.connect(self.cell_changed)
        self.update_colors()
        # Allow double-clicking column headers to rename them
        
        self.table.horizontalHeader().sectionClicked.connect( # type: ignore
            self.select_column
        )
        self.table.horizontalHeader().sectionDoubleClicked.connect( # type: ignore
            self.rename_column
        )

        # Buttons

        button_layout = QHBoxLayout()

        add_row_btn = QPushButton("Add Row")

        add_col_btn = QPushButton("Add Column")

        load_config_btn = QPushButton("Load Config")

        try_assosc_btn = QPushButton("Try Assosiation")

        ok_btn = QPushButton("OK")

        button_layout.addWidget(add_row_btn)

        button_layout.addWidget(add_col_btn)
        button_layout.addWidget(load_config_btn)

        button_layout.addStretch()
        button_layout.addWidget(try_assosc_btn)

        button_layout.addWidget(ok_btn)

        layout.addWidget(self.label)
        layout.addWidget(self.slider)

        layout.addLayout(button_layout)

        # Connect buttons to functions

        add_row_btn.clicked.connect(self.add_row)

        add_col_btn.clicked.connect(self.add_column)
        load_config_btn.clicked.connect(self.load_file)

        try_assosc_btn.clicked.connect(lambda: self.updateFigs.emit(self.extr_info, self, self.column_thresholds, True))

        ok_btn.clicked.connect(self.close_ok)


    # Fill the spreadsheet with the dataframe contents

    def load_dataframe(self):

        self.table.clear()
        self.table.setRowCount(len(self.df))
        self.table.setColumnCount(len(self.df.columns))

        # Create editable column headers

        for c, name in enumerate(self.df.columns):
            self.table.setHorizontalHeaderItem(
                c,
                QTableWidgetItem(str(name))
            )

        # Fill every cell

        for r in range(len(self.df)):
            for c in range(len(self.df.columns)):
                item = QTableWidgetItem(str(self.df.iat[r, c]))
                self.table.setItem(r, c, item)
    #del collumn
    def keyPressEvent(self, a0):
        assert a0 is not None
        if (
            a0.key() == Qt.Key.Key_Delete
            and a0.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.remove_row()

        elif a0.key() == Qt.Key.Key_Delete:
            self.remove_column()

        elif (
            a0.key() == Qt.Key.Key_S
            and a0.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self.save_file()
        elif a0.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            # Move to the cell below
            row = self.table.currentRow()
            col = self.table.currentColumn()
            if row < self.table.rowCount() - 1:
                self.table.setCurrentCell(row + 1, col)
        
        else:
            super().keyPressEvent(a0)

    #slider lable:
    def update_slabel(self, value):

        thresh = value / 100

        if self.active_column:
            self.label.setText(
                f"{self.active_column} threshold: {thresh:.2f}"
            )
        else:
            self.label.setText(
                f"Threshold: {thresh:.2f}"
            )

    def select_column(self, column):

        header = self.table.horizontalHeaderItem(column)

        if header is None:
            return

        self.active_column = header.text()

        # Load that column's current threshold
        value = self.column_thresholds.get(
            self.active_column,
            self.thresh
        )

        self.slider.blockSignals(True)
        self.slider.setValue(int(value * 100)) # type: ignore
        self.slider.blockSignals(False)

        self.label.setText(
            f"{self.active_column} threshold: {value:.2f}"
        )

    def slider_changed(self, value):

        new_thresh = value / 100

        if self.active_column is not None:
            self.column_thresholds[self.active_column] = new_thresh
    #when cell is changed
    def cell_changed(self, item):

        row = item.row()
        col = item.column()

        value = item.text()
        if value in self.valid_genes:
            item.setBackground(QColor("#82FF82"))
        elif value == None:
            item.setBackground(QColor("#FFFFFF"))
        else:
            item.setBackground(QColor("#FF8282"))
    # Add an empty row
    def update_colors(self):

        for r in range(self.table.rowCount()):
            for c in range(self.table.columnCount()):

                item = self.table.item(r, c)

                if item is None:
                    continue

                self.cell_changed(item)

    def add_row(self):

        row = self.table.rowCount()

        self.table.insertRow(row)

        for c in range(self.table.columnCount()):
            self.table.setItem(row, c, QTableWidgetItem(""))

    # Remove selected row

    def remove_row(self):

        row = self.table.currentRow()

        if row >= 0:
            self.table.removeRow(row)

    # Add a new empty column
    def make_caps(self, item):
        # Block signals to prevent infinite loops when calling setText
        self.blockSignals(True)
        item.setText(item.text().upper())
        self.blockSignals(False)

    def add_column(self):

        col = self.table.columnCount()

        self.table.insertColumn(col)
        name = f"Column {col+1}" #eventually change name to user defined name!
        self.table.setHorizontalHeaderItem(
            col,
            QTableWidgetItem(f"Column {col+1}")
        )
        self.column_thresholds[name] = self.thresh
        for r in range(self.table.rowCount()):
            self.table.setItem(r, col, QTableWidgetItem(""))

    # Remove selected column

    def remove_column(self):

        col = self.table.currentColumn()

        if col >= 0:
            self.table.removeColumn(col)
    # Check validiity of table and update if valid
            
    # Rename a column by double-clicking its header

    def rename_column(self, column):

        item = self.table.horizontalHeaderItem(column)

        assert item is not None
        old_name = item.text()

        new_name, ok = QInputDialog.getText(
            self,
            "Rename Column",
            "Column name:",
            text=old_name
        )
        
        if ok and new_name:
            self.column_thresholds[new_name] = self.column_thresholds.pop(old_name)
            item.setText(new_name)

    def close_ok(self):

        self.accept()

    # Convert spreadsheet back into a pandas DataFrame

    def get_dataframe(self):

        rows = self.table.rowCount()
        cols = self.table.columnCount()

        # Read column names

        headers = []

        for c in range(cols):

            item = self.table.horizontalHeaderItem(c)

            if item is None:
                headers.append(f"Column {c+1}")
            else:
                headers.append(item.text())

        # Read every table cell

        data = []

        for r in range(rows):

            row = []

            for c in range(cols):

                item = self.table.item(r, c)

                if item is None:
                    row.append("")
                else:
                    row.append(item.text())

            data.append(row)

        # Return dataframe

        return pd.DataFrame(data, columns=headers)
    def save_file(self):
        config = DFE_data(self.get_dataframe(), self.column_thresholds)
        # Open native cross-platform save dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save File",
            "",
            "Data Files (*.dat);;All Files (*)"
        )
        
        # If user didn't cancel, write the file
        if file_path:
            config.save(file_path)

    def load_file(self):
        loaded_config = DFE_data(pd.DataFrame(), {})
        # Open native cross-platform open dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "",
            "Data Files (*.dat);;All Files (*)"
        )
        
        # If user selected a file, read it
        if file_path:
            self.table.blockSignals(True)
            loaded_config.load(file_path)
            self.df = loaded_config.df
            self.column_thresholds = loaded_config.col_meta['thresholds']

            print(loaded_config.df)
            print(self.df)
            print(self.table.rowCount(), self.table.columnCount())

            self.load_dataframe()
            self.table.viewport().update() # type: ignore
            self.update_colors()

            self.table.blockSignals(False)