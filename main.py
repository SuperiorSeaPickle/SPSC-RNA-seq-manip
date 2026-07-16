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

DATA_DIR = Path(r"D:\SPSC-RNA-Seq\WTA_Preview_FFPE_Breast_Cancer_outs")

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
def init_worker(polygons, ids):
    global TREE, CELL_IDS

    TREE = STRtree(polygons)
    CELL_IDS = ids
def process_batch(df):
    global TREE, CELL_IDS

    points = shapely.points(
        df["x_location"].to_numpy(),
        df["y_location"].to_numpy()
    )

    point_ids, poly_ids = TREE.query(
        points,
        predicate="within"
    )

    result = np.full(len(df), None, dtype=object)

    for p, poly in zip(point_ids, poly_ids):
        result[p] = CELL_IDS[poly]

    df["cell_id"] = result


    return df

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

    print(pq.read_schema(transcripts_dir))
    bsize = 100000#min(((WORKING_MEMORY[1]*0.02*0.1)/trns_bsize), 1)*trns_nrows
    time_tracker = time.perf_counter()
    projected_time = time.perf_counter()
    rows_comp = 0
    
    if (DATA_DIR / "tmp" / "trns_with_cellID.parquet").is_file() == False:
        schema_dict = {
        'transcript_id': pd.Series(dtype='uint64'),
        'cell_id': pd.Series(dtype='str'),
        'x_location': pd.Series(dtype='float'),
        'y_location': pd.Series(dtype='float')
        }

        df= pd.DataFrame(schema_dict)
        df.to_parquet(DATA_DIR / "tmp" / "trns_with_cellID.parquet", index=False)

    MAX_IN_FLIGHT = 16  # About 2 × max_workers is a good starting point
    tmp_file = DATA_DIR / "tmp" / "trns_with_cellID.incomplete.parquet"
    final_file = DATA_DIR / "tmp" / "trns_with_cellID.parquet"

    with ProcessPoolExecutor(
        max_workers=8,
        initializer=init_worker,
        initargs=([c.boundry for c in cells], [c.id for c in cells]),
    ) as executor:

        batch_iter = transcripts.iter_batches(
            batch_size=bsize,
            columns=[
                "transcript_id",
                "cell_id",
                "x_location",
                "y_location",
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

def purge_rows(input_file, output_file = DATA_DIR / "tmp" / "transcripts_purged.parquet"):
    target_column = '"status"'
    value_to_remove = None # Use string or bytes depending on your data schema

    # Establish a connection
    conn = duckdb.connect()

    # Stream out rows that DO NOT match your unwanted value
    query = f"""
        COPY (
            SELECT * 
            FROM read_parquet('{input_file}') 
            WHERE {target_column} != '{value_to_remove}'
        ) 
        TO '{output_file}' (FORMAT 'PARQUET');
    """

    conn.execute(query)

def create_UMAP_profile(tc_association_dir):
    purge_rows(tc_association_dir, DATA_DIR / "tmp" / "transcripts_purged.parquet")
    

    import pyarrow.dataset as ds
    
    # 1. Point to the file (this only scans metadata, 0% data loaded to RAM)
    dataset = ds.dataset(tc_association_dir, format="parquet")

    # 2. Apply a filter and materialize only the matching rows
    # Replace 'status' and 'active' with your column name and target value
    matching_table = dataset.to_table(filter=ds.field("cell_id") == None)

    # 3. Convert the filtered results to a Pandas DataFrame if needed
    df = matching_table.to_pandas()


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
    # path = Path(r"D:\SPSC-RNA-Seq\WTA_Preview_FFPE_Breast_Cancer_outs\tmp\trns_with_cellID.parquet")

    
    # pf = ParquetFile(path)

    # print("Row groups:", len(pf.row_groups))

    # for i, batch in enumerate(pf.iter_row_groups()):
    #     print(i, batch.shape)

    #     batch.to_parquet(
    #         f"recovered_part_{i}.parquet",
    #         engine="pyarrow"
    #     )

    # input_dir = Path(r"C:\Users\bend2\Documents\GitHub\SPSC-RNA-seq-manip\broken parquet")
    # output_file = Path(r"D:\SPSC-RNA-Seq\WTA_Preview_FFPE_Breast_Cancer_outs\tmp\recovered.parquet")

    # files = sorted(input_dir.glob("recovered_part_*.parquet"))

    # writer = None

    # for file in files:
    #     print(f"Adding {file.name}")

    #     table = pq.read_table(file)

    #     if writer is None:
    #         writer = pq.ParquetWriter(
    #             output_file,
    #             table.schema
    #         )

    #     writer.write_table(table)

    # if writer:
    #     writer.close()

    # print("Finished")
    purge_rows(r"D:\SPSC-RNA-Seq\WTA_Preview_FFPE_Breast_Cancer_outs\tmp\recovered.parquet")