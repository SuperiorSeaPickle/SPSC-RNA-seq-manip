import pandas as pd
import numpy as np
import plotly.express as px
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from shapely.strtree import STRtree
import pickle
import shapely
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from cell import cell
import duckdb
from collections import Counter
import h5py
import scanpy as sc
import anndata as ad
from scipy.sparse import csc_matrix
from sklearn.cluster import KMeans

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
    WORKING_MEMORY = (psutil.virtual_memory().total, psutil.virtual_memory().available) #implement later (was such low lever management required?)
    
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


def format_h5(tc_associations):
    if not (DATA_DIR / "tmp" / "trns_with_cellID_regroup.parquet").is_file():
        con = duckdb.connect()
        con.execute("SET enable_progress_bar = true;")
        con.execute("SET enable_progress_bar_print = true;")
        con.execute(f"""
            COPY (
                SELECT *
                FROM '{tc_associations}'
                WHERE is_gene != false
                ORDER BY cell_id
            )
            TO '{DATA_DIR / "tmp" / "trns_with_cellID_regroup.parquet"}'
            (FORMAT PARQUET);
        """)

    selected_file = DATA_DIR / "tmp" / "trns_with_cellID_regroup.parquet"

    tc_parquet = pq.ParquetFile(selected_file)
    num_row_groups = tc_parquet.num_row_groups

    unique_values = duckdb.query(f"""
        SELECT DISTINCT feature_name
        FROM '{selected_file}'
        ORDER BY feature_name
    """).df()

    feature_names = unique_values["feature_name"].tolist()

    feature_index = {
        gene: i
        for i, gene in enumerate(feature_names)
    }

    TOTAL_CELLS = pq.read_metadata(DATA_DIR / "cells.parquet").num_rows

    h5_file = DATA_DIR / "tmp" / "cell_matrix.h5"


    # Build CSC arrays


    data = []
    indices = []
    indptr = [0]
    column_names = []

    last_cell = None
    genes_in_cell = []

    processed = 0

    def finish_cell(cell_id):

        nonlocal processed

        counts = Counter(genes_in_cell)

        # store only nonzero entries
        for gene, count in sorted(
            counts.items(),
            key=lambda x: feature_index[x[0]]
        ):
            indices.append(feature_index[gene])
            data.append(count)

        indptr.append(len(data))
        column_names.append(str(cell_id))

        processed += 1

        if processed % 1000 == 0:
            print(
                f"{processed:,}/{TOTAL_CELLS:,} cells "
                f"({processed/TOTAL_CELLS:.1%})"
            )


    # Iterate through parquet


    for rg in range(num_row_groups):

        df = tc_parquet.read_row_group(rg).to_pandas()

        for row in df.itertuples(index=False):

            if last_cell is None:
                last_cell = row.cell_id

            if row.cell_id != last_cell:

                finish_cell(last_cell)

                genes_in_cell.clear()
                last_cell = row.cell_id

            genes_in_cell.append(row.feature_name)

    # finish final cell

    if last_cell is not None:
        finish_cell(last_cell)

    # Save CSC matrix

    dt = h5py.string_dtype("utf-8")

    with h5py.File(h5_file, "w") as f:

        counts = f.create_group("counts")

        counts.create_dataset(
            "data",
            data=np.asarray(data, dtype=np.uint16),
            compression="gzip"
        )

        counts.create_dataset(
            "indices",
            data=np.asarray(indices, dtype=np.uint32),
            compression="gzip"
        )

        counts.create_dataset(
            "indptr",
            data=np.asarray(indptr, dtype=np.uint64),
            compression="gzip"
        )

        counts.create_dataset(
            "shape",
            data=np.array(
                [len(feature_names), len(column_names)],
                dtype=np.uint64
            )
        )

        f.create_dataset(
            "row_names",
            data=np.asarray(feature_names, dtype=object),
            dtype=dt
        )

        f.create_dataset(
            "column_names",
            data=np.asarray(column_names, dtype=object),
            dtype=dt
        )

    print("Finished writing sparse CSC matrix.")

def create_UMAP(cell_matrix_h5, from_file = False):
    if not from_file:   
        print("loading matrix...")
        with h5py.File(cell_matrix_h5, 'r') as f:
            counts = f['counts']
            gene_names  = f['row_names'][:].astype(str)
            cell_names  = f['column_names'][:].astype(str)
            X = csc_matrix(
            (
                counts["data"][:],
                counts["indices"][:],
                counts["indptr"][:]
            ),
            shape=tuple(counts["shape"][:])
            )

            adata = ad.AnnData(X.T)
            adata.var_names = gene_names
            adata.obs_names = cell_names

        # Keep genes expressed in at least 3 cells and cells with at least 200 genes
        print("cleaning data...")
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)

        print("normalizing data...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        print("finding highly variable genes...")
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata = adata[:, adata.var.highly_variable].copy()
        print("rescaling...")
        sc.pp.scale(adata, max_value=10, zero_center= False)
        print("running PCA...")
        sc.tl.pca(adata,n_comps=50, random_state=0)

        print("generating UMAP...")
        sc.pp.neighbors(adata,n_neighbors=15,n_pcs=50)
        sc.tl.umap(adata)
        
        print("creating KNN groups...")
        sc.tl.leiden(adata, resolution=0.75, flavor="igraph", n_iterations=2)
        print("saving progress...")
        adata.write_h5ad(str(DATA_DIR / "tmp" / "adata_tmp.h5ad"))
    else:
        adata = sc.read_h5ad(str(DATA_DIR / "tmp" / "adata_tmp.h5ad"))
        sc.pl.umap(adata, color= "leiden")

    while True:
        print("type how many clusters you see:")
        try:
            nclust = int(input())
            break
        except:
            print("invalid integer input")

    kmeans = KMeans(n_clusters=nclust, random_state=0).fit(adata.obsm['X_pca'])
    adata.obs['kmeans_5'] = kmeans.labels_.astype(str)
    sc.pl.umap(adata,color='kmeans_5')

def diff_analysis():
    print("loading matrix...")
    adata = sc.read_h5ad(str(DATA_DIR / "tmp" / "adata_tmp.h5ad"))
    print(adata)

    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon", key_added="rank_markers", )
    sc.pl.rank_genes_groups_heatmap(adata,n_genes=3, key="rank_markers")
    sc.pl.rank_genes_groups_stacked_violin(adata, n_genes=3, groupby='leiden')

    print("saving progress...")
    adata.write_h5ad(str(DATA_DIR / "tmp" / "adata_tmp.h5ad"))
    
    marker_df = pd.DataFrame(adata.uns["rank_markers"]["names"]).head(10)
    print(marker_df)

def annotate_cells(auto=False):
    import decoupler as dc
    

    print("loading matrix...")
    adata = sc.read_h5ad(str(DATA_DIR / "tmp" / "adata_tmp.h5ad"))

    diff_output = pd.DataFrame(adata.uns["rank_markers"]["names"]).head(10)
    cell_annots = {}
    if auto:
        # markers = dc.op.resource("PanglaoDB", organism="human")
        # markers = markers[
        # markers["human"].astype(bool)
        # & markers["canonical_marker"].astype(bool)
        # & (markers["human_sensitivity"].astype(float) > 0.5)
        # ]

        # markers = markers.rename(
        #     columns={
        #         "cell_type": "source",
        #         "genesymbol": "target"
        #     }
        # )[["source", "target"]]
        # dc.mt.ora(
        # data=adata,
        # net=markers.drop_duplicates(subset=['source', 'target']),
        # tmin=3,
        # )

        # # 1. Extract the scores from your AnnData object (replace 'ora_estimate' with your exact key)
        acts = dc.pp.get_obsm(
            adata,
            key="score_ora"
        )

        # print("saving progress...")
        # adata.write_h5ad(str(DATA_DIR / "tmp" / "adata_tmp.h5ad"))

        # 2. Use Scanpy's built-in heatmap function
        sc.pl.heatmap(
            acts,
            var_names=acts.var_names,
            groupby="leiden",
            cmap="viridis"
        )

        scores = pd.DataFrame(
            acts.X,
            columns=acts.var_names
        )

        scores["leiden"] = acts.obs["leiden"].values

        cluster_scores = (
            scores
            .groupby("leiden")
            .mean()
        )

        sc.pl.matrixplot(
            acts,
            var_names=acts.var_names,
            groupby="leiden",
            standard_scale="var",
            cmap="viridis"
        )

        annotation = cluster_scores.idxmax(axis=1)

        print(annotation)
    else:
        print("Manual Mode: Name the given cell based on its marker genes listed below. Press [enter] to move on to the next group")

        for i in range(diff_output.shape[1]):
            print(diff_output.iloc[:,i])
            print("annotate cell as: ",end="")
            cell_annots[i] = input()
        
        adata.obs['cell_type'] = adata.obs['leiden'].map(cell_annots).astype('category')
        
        print("saving progress...")
        adata.write_h5ad(str(DATA_DIR / "tmp" / "adata_tmp.h5ad"))

        print(adata)

    



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
    #format_h5(r"D:\SPSC-RNA-Seq\WTA_Preview_FFPE_Breast_Cancer_outs\tmp\trns_with_cellID.parquet")
    create_UMAP(r"D:\SPSC-RNA-Seq\WTA_Preview_FFPE_Breast_Cancer_outs\tmp\cell_matrix.h5")
    diff_analysis()