import pandas as pd
import numpy as np
import plotly.express as px
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from shapely.geometry import Polygon
from shapely.strtree import STRtree
import pickle
import shapely
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from cell import cell
import duckdb
from collections import Counter

DATA_DIR = Path(r"D:\SPSC-RNA-Seq\WTA_Preview_FFPE_Breast_Cancer_outs")

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
def init_worker(polygons, ids):
    global TREE, CELL_IDS

    TREE = STRtree(polygons)
    CELL_IDS = ids
def process_batch(df):
    global TREE, CELL_IDS

    # Make sure CELL_IDS supports numpy indexing
    cell_ids = np.asarray(CELL_IDS)

    # Create point geometries
    points = shapely.points(
        df["x_location"].to_numpy(),
        df["y_location"].to_numpy()
    )

    # Query STRtree
    point_ids, poly_ids = TREE.query(
        points,
        predicate="within"
    )

    # No points matched any polygon
    if len(point_ids) == 0:
        return df.iloc[0:0].copy().assign(
            cell_id=np.array([], dtype=np.int32)
        )

    # Assign cell IDs
    cell_id = np.full(len(df), -1, dtype= object)

    cell_id[point_ids] = cell_ids[poly_ids]

    # Remove unmatched points
    keep = cell_id != -1

    df = df.loc[keep].copy()
    df["cell_id"] = cell_id[keep]

    return df

def assign_gene_to_cell(transcripts_dir, cellBoundres_dir, keep_unasigned = False):
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

    transcripts = pq.ParquetFile(transcripts_dir)
    trns_nrows = pq.read_metadata(transcripts_dir).num_rows
    trns_bsize = Path(transcripts_dir).stat().st_size

    bsize = 100000#min(((WORKING_MEMORY[1]*0.02*0.1)/trns_bsize), 1)*trns_nrows
    rows_comp = 0
    
    if (DATA_DIR / "tmp" / "trns_with_cellID.parquet").is_file() == False:
        schema_dict = {
        'transcript_id': pd.Series(dtype='uint64'),
        'feature_name': pd.Series(dtype = 'str'),
        'cell_id': pd.Series(dtype='str'),
        'x_location': pd.Series(dtype='float'),
        'y_location': pd.Series(dtype='float'),
        'is_gene': pd.Series(dtype='bool')
        }

        df= pd.DataFrame(schema_dict)
        df.to_parquet(DATA_DIR / "tmp" / "trns_with_cellID.parquet", index=False)

    MAX_IN_FLIGHT = 16  # About 2 × max_workers is a good starting point
    tmp_file = DATA_DIR / "tmp" / "trns_with_cellID.incomplete.parquet"
    final_file = DATA_DIR / "tmp" / "trns_with_cellID.parquet"
    writer = None
    with ProcessPoolExecutor(
        max_workers=8,
        initializer=init_worker,
        initargs=([c.boundry for c in cells], [c.id for c in cells]),
    ) as executor:

        batch_iter = transcripts.iter_batches(
            batch_size=bsize,
            columns=[
                "transcript_id",
                "feature_name",
                "cell_id",
                "x_location",
                "y_location",
                "is_gene"
            ],
        )

        futures = {}

        # Fill the pipeline
        for _ in range(MAX_IN_FLIGHT):
            try:
                batch = next(batch_iter)
            except StopIteration:
                break

            df = batch.to_pandas()
            future = executor.submit(process_batch, df)
            futures[future] = None

        try:
                while futures:

                    # Wait for one completed batch
                    future = next(as_completed(futures))
                    futures.pop(future)

                    df = future.result()

                    rows_comp += len(df)

                    print(
                        f"{100 * rows_comp / trns_nrows:.2f}% complete "
                        f"({rows_comp:,}/{trns_nrows:,})",
                        flush=True
                    )

                    # Convert pandas dataframe to arrow table
                    table = pa.Table.from_pandas(df)

                    # Create parquet writer once

                    if writer is None:
                        writer = pq.ParquetWriter(
                            tmp_file,
                            table.schema
                        )

                    # Write this batch
                    writer.write_table(table)


                    # Submit one more batch
                    try:
                        batch = next(batch_iter)

                        df = batch.to_pandas()

                        future = executor.submit(
                            process_batch,
                            df
                        )

                        futures[future] = None

                    except StopIteration:
                        pass


        finally:
            # Ensure footer is written even if something fails
            if writer is not None:
                    writer.close()


        # Only rename after successful completion
        if tmp_file.exists():
            tmp_file.replace(final_file)

        print(f"Saved: {final_file}")
    return trns_seleced_path


def create_UMAP_profile(tc_associations):
    if((DATA_DIR / "tmp" / "trns_with_cellID_regroup.parquet").is_file() == False):
        con = duckdb.connect()
        con.execute(f"""
            COPY (
                SELECT *
                FROM '{tc_associations}'
                ORDER BY cell_id
            )
            TO '{DATA_DIR / "tmp" / "trns_with_cellID_regroup.parquet"}'
            (FORMAT PARQUET);
        """) #takes 5-20 min

    selected_file = DATA_DIR / "tmp" / "trns_with_cellID_regroup.parquet"

    tc_parquet = pq.ParquetFile(selected_file)
    num_row_groups = tc_parquet.num_row_groups

    print(f"Total row groups to process: {num_row_groups}")

    # Connect to an in-memory database and run a distinct query
    unique_values = duckdb.query(f"""
        SELECT DISTINCT feature_name 
        FROM '{selected_file}'
    """).df()


    features_uo = unique_values.sort_values('feature_name')

    schema_dict = {
        'feature_name': pd.Series(dtype='str'),
        }

    df= pd.DataFrame(schema_dict)
    df.to_parquet(DATA_DIR / "tmp" / "cell_matrix.parquet", index=False)

    # 2. Convert it into a DataFrame
    df = pd.DataFrame(features_uo)

    # 3. Write to a Parquet file
    df.to_parquet(DATA_DIR / "tmp" / "cell_matrix.parquet", engine="pyarrow")
    del df

    current_cell_id = None

    for row in stream_rows_from_parquet():

        # Have we reached a new cell?
        if row["cell_id"] != current_cell_id:

            # Finish processing the previous cell
            if current_cell_id is not None:
                # Insert code here
                pass

            # Start the new cell
            current_cell_id = row["cell_id"]

            # Insert code here
            pass

        # Process a row belonging to the current cell
        # Insert code here
        pass

    # Finish processing the last cell
    if current_cell_id is not None:
        # Insert code here
        pass
    for i in range(num_row_groups):
        arrow_table = tc_parquet.read_row_group(i)
        df_group = arrow_table.to_pandas()

        for r in df_group:
            gfeat.append(r['feature_name'])

        frequencies = Counter((gfeat))
        freq_org = {item: frequencies[item] for item in features_uo}
        
        
        
        
        gfeat.clear()
        

        





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
if __name__ == "__main__":
    #assign_gene_to_cell(DATA_DIR / "transcripts.parquet", DATA_DIR / "cell_boundaries.parquet")
    create_UMAP_profile(r"D:\SPSC-RNA-Seq\WTA_Preview_FFPE_Breast_Cancer_outs\tmp\trns_with_cellID.parquet")