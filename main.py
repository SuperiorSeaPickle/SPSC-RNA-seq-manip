import pandas as pd
import plotly.express as px
import pyarrow.parquet as pq
from pathlib import Path
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.strtree import STRtree
import pickle
import time

DATA_DIR = r"D:\SPSC-RNA-Seq\WTA_Preview_FFPE_Breast_Cancer_outs"
class cell:
    def __init__(self, id, coords):
        self.id = id
        self.boundry = Polygon(coords)

def selected_to_tmp(source_parquet, output_dir, selected_columns):
    import gc
    # 1. Open the source file metadata stream
    source_file = pq.ParquetFile(source_parquet)

    # 3. Create an incremental batch generator for ONLY those columns
    batch_stream = source_file.iter_batches(
        batch_size=8192, 
        columns=selected_columns
    )

    # 4. Pull the schema of the first slice to initialize the writer
    first_batch = next(batch_stream)
    with pq.ParquetWriter(output_dir, first_batch.schema) as writer:
        writer.write_batch(first_batch)
        del first_batch

    # 5. Stream chunks from input disk to output disk sequentially
        for batch in batch_stream:
            writer.write_batch(batch)
            del batch

    # 6. Finalize and close the file
    writer.close()
    gc.collect()
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
    print("loading cells as objects ...")
    cells = []
    if (Path(DATA_DIR) / 'tmp'/ 'cell_objects_loaded.pkl').is_file():
        with open(Path(DATA_DIR) / 'tmp'/ 'cell_objects_loaded.pkl', "rb") as file:
            cells = pickle.load(file)
    else:
        TOTAL_CELLS = boundries_df['cell_id'].nunique()
        last_cell_id = None
        tmp_coords = []

        for row in boundries_df.itertuples(index=False):

            if row.cell_id != last_cell_id:
        
                if last_cell_id is not None:
                    cells.append(cell(last_cell_id, tmp_coords))
                    if len(cells) % 1000 == 0: 
                        print(f"loaded {round((len(cells)/TOTAL_CELLS)*100,2)}% cell objects")
                tmp_coords = []
                last_cell_id = row.cell_id

            tmp_coords.append((row.vertex_x, row.vertex_y))

        # Don't forget the final cell
        if last_cell_id is not None:
            cells.append(cell(last_cell_id, tmp_coords))
        with open(Path(DATA_DIR) / 'tmp'/ 'cell_objects_loaded.pkl', "wb") as file:
            pickle.dump(cells, file)
    print(f"Succesfully loaded {len(cells)} cells")
    return cells
def build_spatial_index(cells):
    print("Building spatial index ...")
    polygons = [c.boundry for c in cells]

    tree = STRtree(polygons)
    polygon_lookup = {
        id(poly): cell
        for poly, cell in zip(polygons,cells)
    }

    return tree, polygon_lookup
def find_cell(point, tree, polygon_lookup):

    candidates = tree.query(point)

    for poly in candidates:

        if poly.contains(point):

            return polygon_lookup[id(poly)].id

    return None
def assign_gene_to_cell(transcripts_dir, cellBoundres_dir):
    import shutil
    import psutil

    #enviorment info
    PARENT_PATH = Path(transcripts_dir).parent
    trns_seleced_path = PARENT_PATH / "tmp" / "transcripts_selected.parquet"
    WORKING_SPACE = shutil.disk_usage(PARENT_PATH.root) #tupple: total,used,free
    WORKING_MEMORY = (psutil.virtual_memory().total, psutil.virtual_memory().available)
    
    #create tmp if it doesnt exist allready
    if((PARENT_PATH / "tmp").is_dir() == False):
        (PARENT_PATH / "tmp").mkdir(parents=True, exist_ok = False)

    #load constituant data into memory
    cell_boundries = pd.read_parquet(cellBoundres_dir) #about 15 mb ram
    cells = df_to_cells(cell_boundries)
    tree, lookup = build_spatial_index(cells)

    transcripts = pq.ParquetFile(transcripts_dir)
    trns_nrows = pq.read_metadata(transcripts_dir).num_rows
    trns_bsize = Path(transcripts_dir).stat().st_size

    bsize = min(((WORKING_MEMORY[1]*0.1)/trns_bsize), 1)*trns_nrows
    time_tracker = time.perf_counter()
    projected_time = time.perf_counter()
    for batch in transcripts.iter_batches(batch_size=bsize,columns=['transcript_id', 'cell_id', 'x_location', 'y_location']):

        df = batch.to_pandas()

        for row in range(len(df)):

            point = Point(
                df.iat[row,2],
                df.iat[row,3]
            )
            df.iat[row,1] = find_cell(
                point,
                tree,
                lookup
            )

            if row % 20 == 0:
                projected_time = ((time.perf_counter()- time_tracker)/20)*trns_nrows
                print(f"process will complete in {round(projected_time/3600,0)} hours, {round(((projected_time/3600) % 1)*60,0)} min, and {round(((((projected_time/3600) % 1)*60) % 1)*60,0)} sec")
                time_tracker = time.perf_counter()
        
        df.to_parquet(
            trns_seleced_path / "trns_with_cellID.parquet", 
            engine='fastparquet', 
            append=True
        )

    trns_seleced_path = trns_seleced_path / "trns_with_cellID.parquet"
    print("cell attributation saved as:    " + trns_seleced_path)
    return trns_seleced_path

#def create_UMAP_profile(tc_association_dir, )

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

assign_gene_to_cell(r"D:\SPSC-RNA-Seq\WTA_Preview_FFPE_Breast_Cancer_outs\transcripts.parquet", r"D:\SPSC-RNA-Seq\WTA_Preview_FFPE_Breast_Cancer_outs\cell_boundaries.parquet")