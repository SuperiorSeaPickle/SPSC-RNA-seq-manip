import pandas as pd
import plotly.express as px
import pyarrow.parquet as pq
from pathlib import Path
import matplotlib.path as mpath

#testing change

class cell:
    def __init__(self, id):
        self.id = id
        self.boundry = None

def selected_to_tmp(source_parquet, output_dir, selected_columns):
    # 1. Open the source file metadata stream
    source_file = pq.ParquetFile(source_parquet)

    # 3. Create an incremental batch generator for ONLY those columns
    batch_stream = source_file.iter_batches(
        batch_size=65536, 
        columns=selected_columns
    )

    # 4. Pull the schema of the first slice to initialize the writer
    first_batch = next(batch_stream)
    writer = pq.ParquetWriter(output_dir, schema=first_batch.schema)

    # 5. Stream chunks from input disk to output disk sequentially
    writer.write_batch(first_batch)
    for batch in batch_stream:
        writer.write_batch(batch)

    # 6. Finalize and close the file
    writer.close()
def delete_file(fp):
    file_path = Path(fp)

    try:
        # missing_ok=True prevents FileNotFoundError if the file is already gone
        file_path.unlink(missing_ok=True)
        print("File deleted successfully.")
    except PermissionError:
        print("Permission denied: Cannot delete this file.")
    except Exception as e:
        print(f"An error occurred: {e}")
def df_to_cells(boundries_df):
    cells = []
    last_cell_id = None
    current_cell_id = None
    current_cell = None
    tmp_coords = []

    for row in boundries_df:
        current_cell_id = boundries_df.iat[row,0]
        if (current_cell_id != last_cell_id):
            last_cell_id = current_cell_id
            if current_cell != None:
                current_cell.boundry = mpath.Path(tmp_coords)
                tmp_coords.clear()
                cells.append(current_cell)

            current_cell = cell(current_cell_id)
        tmp_coords.append((boundries_df.iat[row,1],boundries_df.iat[row,2]))
    
    return cells


def assign_gene_to_cell(transcripts_dir, cellBoundres_dir):
    import polars as pl
    import shutil
    import psutil

    #enviorment info
    PARENT_PATH = Path(transcripts_dir).parent
    trns_seleced_path = PARENT_PATH + r"\tmp\transcripts_selected.parquet"
    WORKING_SPACE = shutil.disk_usage(Path(transcripts_dir).root) #tupple: total,used,free
    WORKING_MEMORY = (psutil.virtual_memory().total, psutil.virtual_memory().available)
    
    #create tmp if it doesnt exist allready
    if(Path(PARENT_PATH + r"\tmp").is_dir() == False):
        (PARENT_PATH + r"\tmp").mkdir(parents=True, exist_ok = False)
    
    selected_to_tmp(transcripts_dir, trns_seleced_path, ['transcript_id', 'cell_id', 'x_location', 'y_location'])
    
    lazy_trns = pl.scan_parquet(trns_seleced_path)
    sorted_lazy_trns = lazy_trns.sort("x_location")
    sorted_lazy_trns.sink_parquet(Path(trns_seleced_path).parent + r"\sorted_selected_transcripts.parquet", statistics=True)
    delete_file(trns_seleced_path)
    trns_seleced_path = Path(trns_seleced_path).parent + r"\sorted_selected_transcripts.parquet"

    #load constituant data into memory
    cell_boundries = pd.read_parquet(cellBoundres_dir) #about 15 mb ram
    cell_boundries = df_to_cells(cell_boundries)
    

    transcripts_rdr = pq.ParquetFile(trns_seleced_path)
    trns_nrows = pq.read_metadata(trns_seleced_path).num_rows
    trns_bsize = Path(trns_seleced_path).stat().st_size


    for batch in transcripts_rdr.iter_batches(batch_size= min(((WORKING_MEMORY - 10**9)/trns_bsize), 1)*trns_nrows): #base batch size on available memory
        # 'batch' is a pyarrow.RecordBatch object
        print(f"Loaded batch with {len(batch)} rows.")
        
        # Optional: Convert the specific batch into a standard Pandas DataFrame
        df = batch.to_pandas()

        for c in cell_boundries:
            for row in batch:
                is_inside = cell_boundries[c].boundry.contains_point((df.iat[row, 2],df.iat[row, 3])) #if it contains the point
                if is_inside:
                    df.iat[row,1] = c.id
        
        df.to_parquet(
            Path(trns_seleced_path).parent + r"\trns_with_cellID.parquet", 
            engine='fastparquet', 
            append=True
        )

    delete_file(trns_seleced_path)
    trns_seleced_path = Path(trns_seleced_path).parent + r"\trns_with_cellID.parquet"
    print("cell attributation saved as:    " + trns_seleced_path)
    return trns_seleced_path
    



def total_count_scatter():

    dirpath = r"F:\SPSC-RNA-Seq\WTA_Preview_FFPE_Breast_Cancer_outs\cells.parquet"
    df = pd.read_parquet(dirpath)
    print(df.columns)

    fig = px.scatter(

        df,
        x = "x_centroid",
        y = "y_centroid",
        color  = "total_counts",
        color_continuous_scale= "Viridis"
    )

    fig.update_layout(
        plot_bgcolor="black",   # Inner plot area background
        paper_bgcolor="white"     # Outer chart area background
    )
    fig.update_traces(marker_size=3) 
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)


    fig.show(config={'scrollZoom': True})