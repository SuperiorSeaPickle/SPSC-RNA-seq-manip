import sys
import pandas as pd

from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QInputDialog
)

from PyQt6.QtGui import QColor
from PyQt6.QtCore import QEventLoop,QSignalBlocker

class DataFrameEditor(QDialog):
    """
    Simple popup window for editing a pandas DataFrame.
    """

    def __init__(self, df, valid_genes):
        super().__init__()

        # Save a copy of the dataframe.
        # We edit the copy so Cancel leaves the original untouched.

        self.df = df.copy()
        self.valid_genes = valid_genes.copy()

        # Window settings

        self.setWindowTitle("DataFrame Editor")
        self.resize(800, 500)


        self.waiting = None
        self.finished = False # type: ignore
        self.accepted = False # type: ignore
        self.result_df = None

        # Main vertical layout

        layout = QVBoxLayout(self)

        # Spreadsheet widget

        self.table = QTableWidget()
        layout.addWidget(self.table)

        # Load dataframe into spreadsheet

        self.table.blockSignals(True)
        self.load_dataframe()
        self.table.blockSignals(False)

        # Update colors whenever a user changes a cell
        self.table.itemChanged.connect(self.cell_changed)
        self.update_colors()
        # Allow double-clicking column headers to rename them

        self.table.horizontalHeader().sectionDoubleClicked.connect( # type: ignore
            self.rename_column
        )

        # Buttons

        button_layout = QHBoxLayout()

        add_row_btn = QPushButton("Add Row")
        remove_row_btn = QPushButton("Remove Row")

        add_col_btn = QPushButton("Add Column")
        remove_col_btn = QPushButton("Remove Column")

        try_assosc_btn = QPushButton("Try Assosiation")

        ok_btn = QPushButton("OK")

        button_layout.addWidget(add_row_btn)
        button_layout.addWidget(remove_row_btn)

        button_layout.addWidget(add_col_btn)
        button_layout.addWidget(remove_col_btn)

        button_layout.addStretch()
        button_layout.addWidget(try_assosc_btn)

        button_layout.addWidget(ok_btn)

        layout.addLayout(button_layout)

        # Connect buttons to functions

        add_row_btn.clicked.connect(self.add_row)
        remove_row_btn.clicked.connect(self.remove_row)

        add_col_btn.clicked.connect(self.add_column)
        remove_col_btn.clicked.connect(self.remove_column)

        try_assosc_btn.clicked.connect(self.try_assosiation)

        ok_btn.clicked.connect(self.close_ok)

    # Fill the spreadsheet with the dataframe contents

    def load_dataframe(self):

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
    #wait for input
    def wait_for_user(self):

        self.waiting = QEventLoop()

        self.waiting.exec()

        return self.result_df

    #when cell is changed
    def cell_changed(self, item):

        row = item.row()
        col = item.column()

        value = item.text()
        with QSignalBlocker(self.table):
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

    def add_column(self):

        col = self.table.columnCount()

        self.table.insertColumn(col)

        self.table.setHorizontalHeaderItem(
            col,
            QTableWidgetItem(f"Column {col+1}")
        )

        for r in range(self.table.rowCount()):
            self.table.setItem(r, col, QTableWidgetItem(""))

    # Remove selected column

    def remove_column(self):

        col = self.table.currentColumn()

        if col >= 0:
            self.table.removeColumn(col)
    # Check validiity of table and update if valid

    def try_assosiation(self):

        current_df = self.get_dataframe()

        is_valid = current_df.isin(self.valid_genes).all().all()

        if is_valid:

            self.result_df = current_df

            # wake up wait_for_user()
            if self.waiting is not None:
                self.waiting.quit()

            else:
                print("Invalid association")
            
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
            item.setText(new_name)

    def close_ok(self):

        self.accepted = True # type: ignore
        self.finished = True # type: ignore

        if self.waiting is not None:
            self.waiting.quit()

        self.close()

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