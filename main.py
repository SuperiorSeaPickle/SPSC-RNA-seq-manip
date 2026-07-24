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
import seaborn as sns
import matplotlib.pyplot as plt
import spatialdata as sd
import spatialdata_io as sd_io
from spatialdata.transformations import Identity
from spatialdata.models import PointsModel, Image2DModel,ShapesModel,TableModel
import dask.dataframe as dd
import dask.array as da
import tifffile
import geopandas as gpd

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

def create_UMAP(cell_matrix_h5, from_file = False, view_plots=True):
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

        # Keep genes expressed in at least 3 cells and cells with at least 200 genes\
        print("cleaning data...")
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)

        adata.layers["counts"] = adata.X.copy()

        print("normalizing data...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        print("finding highly variable genes...")
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata_hvg = adata[:, adata.var.highly_variable].copy()

        print("rescaling...")
        sc.pp.scale(adata_hvg, max_value=10, zero_center= False)
        print("running PCA...")
        sc.tl.pca(adata_hvg,n_comps=50, random_state=0)

        print("generating UMAP...")
        sc.pp.neighbors(adata_hvg,n_neighbors=15,n_pcs=50)
        sc.tl.umap(adata_hvg)
        
        print("creating KNN groups...")
        sc.tl.leiden(adata_hvg, resolution=0.75, flavor="igraph", n_iterations=2)

        # Embeddings
        adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
        adata.obsm["X_umap"] = adata_hvg.obsm["X_umap"]

        # Neighbor graph (optional, but useful)
        adata.obsp["connectivities"] = adata_hvg.obsp["connectivities"]
        adata.obsp["distances"] = adata_hvg.obsp["distances"]

        # Metadata
        adata.obs["leiden"] = adata_hvg.obs["leiden"]

        # Copy relevant unstructured data
        for key in ["pca", "neighbors", "umap", "leiden"]:
            adata.uns[key] = adata_hvg.uns[key]
            
        print("saving progress...")
        adata.write_h5ad(
            str(DATA_DIR / "tmp" / "adata_tmp.h5ad"),
            compression="lzf"
        )
    else:
        adata = sc.read_h5ad(str(DATA_DIR / "tmp" / "adata_tmp.h5ad"), backed="r+")
        if view_plots:
            sc.pl.umap(adata, color= "leiden")

    while True:
        print("type how many clusters you see:")
        try:
            nclust = int(input())
            break
        except:
            print("invalid integer input")
    if view_plots:
        kmeans = KMeans(n_clusters=nclust, random_state=0).fit(adata.obsm['X_pca'])
        adata.obs['kmeans_5'] = kmeans.labels_.astype(str)
        sc.pl.umap(adata,color='kmeans_5')

def diff_analysis(view_plots = True):
    print("loading matrix...")
    adata = sc.read_h5ad(str(DATA_DIR / "tmp" / "adata_tmp.h5ad"))
    print("ranking gene groups...")
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon",reference="rest", use_raw=False )

    if view_plots:
        sc.pl.rank_genes_groups(adata, n_genes=20,sharey=False)
        sc.pl.rank_genes_groups_heatmap(adata, n_genes=10, groupby="leiden", show_gene_labels=True,cmap="viridis")
        sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, standard_scale="var")

    print("saving progress...")
    adata.write_h5ad(
        str(DATA_DIR / "tmp" / "adata_tmp.h5ad"),
        compression="lzf"
    )
    
    marker_df = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(20)
    print(marker_df)

def validate_user_input(comm_dict, gene_dict):
    
    while True:
        user_input = input()
        if user_input in comm_dict:
            if user_input == "[help]":
                print(comm_dict)
            else:    
                break
        else:
            if user_input in gene_dict:
                break
            else:
                print("invalid command or gene. Type [help] to see a list of valid commands.")
    return user_input


def annotate_cells(auto=True, view_figures= True):
    import celltypist
    

    print("loading matrix...")
    adata = sc.read_h5ad(str(DATA_DIR / "tmp" / "adata_tmp.h5ad"), backed="r+")
    print(adata)
    diff_output = pd.DataFrame(adata.uns['rank_genes_groups']["names"]).head(10)
    cell_annots = {}
    tmp_cpd = {
        "unassinged": "#727171",  # Light Orange / Orange
        'selected'  : "#56B4E9",  # Sky Blue  # Reddish Purple
        'assigned'  : "#000000"   # Black
    }

    if auto:
        celltypist.models.download_if_required()
        celltypist.models.models_description()
        predictions = celltypist.annotate(
            adata,
            model="Cells_Adult_Breast.pkl",
            majority_voting=True
        )
        adata = predictions.to_adata()
        print("saving progress...")
        adata.write_h5ad(
            str(DATA_DIR / "tmp" / "adata_tmp.h5ad"),
            compression="lzf"
        )
        sc.pl.umap(
            adata,
            color="majority_voting",
            size=2
        )
        ct_table = pd.crosstab(
            adata.obs['leiden'], 
            adata.obs['majority_voting'],
            normalize="index"
        )*100

        # Display raw cell counts
        print(ct_table.round(1))

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            ct_table, 
            annot=True,        # Show percentages inside cells
            fmt=".1f",         # 1 decimal place
            cmap="Blues", 
            cbar_kws={'label': 'Percentage of Cluster (%)'}
        )
        plt.title("Leiden Clusters vs. CellTypist Annotations")
        plt.xlabel("CellTypist Majority Voting")
        plt.ylabel("Leiden Cluster")
        plt.tight_layout()
        plt.show()
    else:
        commands = {
            "[ec]annotation_name": "ends the creation of current cluster mask",
            "[end]" : "complete cluster annotation and close",
            "[set_thresh]": "change the default threshold (between 0 and 10)",
            "[rcc]": "restart current cluster",
            "[RAC]": "restart all clusters",
            "[help]": "print instructions and command list"
        }

        valid_genes = pd.unique((pd.DataFrame(adata.uns['rank_genes_groups']["names"]).head(50)).values.ravel()).tolist()

        print("Manual Mode: Use the figures that will be shown to create Expression filters.\n" \
        "enter the genes by typing the name, then pressing enter to add another gene. type [ec] to finish the filter. for example:\n" \
        "KIT \n" \
        "CTSG \n" \
        "MS4A2\n" \
        "CPA3\n" \
        "IL1RL1\n" \
        "[ec]cell type 1\n" \
        "an updating viewer will show you what's been selected\n" \
        "Press [Enter] to continue...")

        com = None
        tmp_markers = []
        markers = pd.DataFrame()
        thresh = 2.0
        cname = None

        coords = adata.obsm["X_umap"]
        adata.obs["annotation"] = "U"
        fig, ax = plt.subplots()
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=tmp_cpd["unassinged"],
            s=0.5
        )
        plt.show(block=False)

        while True:
            com = validate_user_input(commands, valid_genes)
            if com.startswith("[ec]"):
                cname = com[4:]
                markers[cname] = tmp_markers
                tmp_markers.clear()


            elif com == "[end]":
                break
            elif com == "[set_thresh]":
                while True:
                    try:
                        thresh = float(input())
                        if not (0 <= thresh <= 10):
                            raise ValueError(f"threshold must be between 0 and 1. Got: {thresh}")
                    except:
                        print("invalid input: must be float in interval [0,10]")
            elif com == "[rcc]":
                tmp_markers.clear()
            elif com == "[RAC]":
                markers = pd.DataFrame()
            else:
                tmp_markers.append(com)

            expr = adata[:, tmp_markers].X
            mask = np.ones(adata.n_obs, dtype=bool)
            for i in range(len(tmp_markers)):
                gene_expr = expr[:, i]

                if hasattr(gene_expr, "toarray"):
                    gene_expr = gene_expr.toarray().ravel()

                mask &= gene_expr > thresh
            adata.obs.loc[mask, "annotation"] = cname
            groups = adata.obs["annotation"]
            colors = np.full(
                adata.n_obs,
                tmp_cpd["assigned"],
                dtype=object
            )
            colors[groups == "U"] = tmp_cpd["unassinged"]
            colors[groups == cname] = tmp_cpd["selected"]
            scatter.set_facecolors(colors)   # categorical RGB colors
            fig.canvas.draw_idle()
            plt.pause(0.01)
        



def view_spatial(adata):

    # ddf = dd.read_parquet(str(DATA_DIR / "tmp" / "trns_with_cellID_regroup.parquet"),
    # columns=[
    #     "x_location",
    #     "y_location",
    #     "feature_name",
    # ])
    # points_element = PointsModel.parse(
    #     ddf,
    #     coordinates={'x': "x_location", 'y': "y_location"},
    #     feature_key= "feature_name"
    # )

    ome_dask = da.from_zarr(tifffile.imread(str(DATA_DIR/"morphology.ome", aszarr=True)))
    image_element = Image2DModel.parse(
        data=ome_dask,
        dims=('c','y','x'),
        chunks=(1,1024,1024),
        transformations={"global": Identity()}
    )

    gdf = gpd.read_parquet(str(DATA_DIR/"cell_boundaries.parquet"))
    shape_trnsfr = {"global": Identity()}
    shapes_element = ShapesModel.parse(gdf, transformations=shape_trnsfr)

    adata.obs["region"] = "cell_boundries"
    adata.obs["instance_id"] = adata.obs_names.astype(int)
    table_element = TableModel.parse(
        adata,
        region="cell_boundries",
        region_key="region",
        instance_key="instance_id"
    )
    sdata = sd.SpatialData(
    images={"tissue_image": image_element},
    #points={"transcripts": points_element},
    shapes={"cell_boundaries": shapes_element},
    tables={"expression_table": table_element}
)
    interactive = interactive(sdata)
    interactive.run()


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
    #create_UMAP(r"D:\SPSC-RNA-Seq\WTA_Preview_FFPE_Breast_Cancer_outs\tmp\cell_matrix.h5",view_plots=False)
    #diff_analysis(view_plots=False)
    #annotate_cells(auto=False)
    print("loading matrix...")
    adata = sc.read_h5ad(str(DATA_DIR / "tmp" / "adata_tmp.h5ad"), backed="r+")
    view_spatial(adata)