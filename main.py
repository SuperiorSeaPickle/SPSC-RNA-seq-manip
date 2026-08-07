import pandas as pd
import geopandas as gpd
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
from pyrmid_generator import create_zarr_pyramid
from DataFrameEditor import DataFrameEditor
from plot_window import ScatterPlotWindow
import duckdb
from collections import Counter
import h5py
import scanpy as sc
import anndata as ad
from scipy.sparse import csc_matrix
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from typing import Literal
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QEventLoop
import sys
import spatialdata as sd
import tifffile
import zarr
import cv2

mpl.use("qtagg")
App = QApplication.instance() or QApplication(sys.argv)
editor = None

DATA_DIR = Path(r"C:\Users\bend2\Documents\PROJECTS\aterads test")


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
    feature_ids = feature_names  # placeholder until a real ID column exists

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
    barcodes = []

    last_cell = None
    genes_in_cell = []

    processed = 0

    def finish_cell(cell_id):

        nonlocal processed

        counts = Counter(genes_in_cell)

        # store only nonzero entries, sorted by feature row index
        # (10x expects indices within each column to be sorted ascending)
        for gene, count in sorted(
            counts.items(),
            key=lambda x: feature_index[x[0]]
        ):
            indices.append(feature_index[gene])
            data.append(count)

        indptr.append(len(data))
        barcodes.append(str(cell_id))

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

    # Save CSC matrix in 10x Genomics H5 layout

    dt = h5py.string_dtype("utf-8")

    with h5py.File(h5_file, "w") as f:

        matrix = f.create_group("matrix")

        matrix.create_dataset(
            "data",
            data=np.asarray(data, dtype=np.int32),
            compression="gzip"
        )

        matrix.create_dataset(
            "indices",
            data=np.asarray(indices, dtype=np.int32),
            compression="gzip"
        )

        matrix.create_dataset(
            "indptr",
            data=np.asarray(indptr, dtype=np.int32),
            compression="gzip"
        )

        matrix.create_dataset(
            "shape",
            data=np.array(
                [len(feature_names), len(barcodes)],
                dtype=np.int32
            )
        )

        matrix.create_dataset(
            "barcodes",
            data=np.asarray(barcodes, dtype=object),
            dtype=dt
        )

        features = matrix.create_group("features")

        features.create_dataset(
            "id",
            data=np.asarray(feature_ids, dtype=object),
            dtype=dt
        )

        features.create_dataset(
            "name",
            data=np.asarray(feature_names, dtype=object),
            dtype=dt
        )

        features.create_dataset(
            "feature_type",
            data=np.asarray(
                ["Gene Expression"] * len(feature_names), dtype=object
            ),
            dtype=dt
        )

    print("Finished writing sparse CSC matrix.")


def create_UMAP(cell_matrix_h5, from_file = False, view_plots=True, view_kmeans= False):
    if not from_file:   
        print("loading matrix...")
        with h5py.File(cell_matrix_h5, 'r') as f:
            counts = f['counts']
            gene_names  = f['row_names'][:].astype(str) # type: ignore
            cell_names  = f['column_names'][:].astype(str) # type: ignore
            X = csc_matrix(
            (
                counts["data"][:], # type: ignore
                counts["indices"][:], # type: ignore
                counts["indptr"][:] # type: ignore
            ),
            shape=tuple(counts["shape"][:]) # type: ignore
            )

            adata = ad.AnnData(X.T)
            adata.var_names = gene_names # type: ignore
            adata.obs_names = cell_names # type: ignore

        # Keep genes expressed in at least 3 cells and cells with at least 200 genes\
        print("cleaning data...")
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)

        adata.layers["counts"] = adata.X.copy() # type: ignore

        print("normalizing data...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        print("finding highly variable genes...")
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata_hvg = adata[:, adata.var.highly_variable].copy() # type: ignore

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
        adata.obs["leiden"] = adata_hvg.obs["leiden"] # type: ignore

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
    if view_kmeans and view_plots:
        while True:
            print("type how many clusters you see:")
            try:
                nclust = int(input())
                break
            except:
                print("invalid integer input")

        kmeans = KMeans(n_clusters=nclust, random_state=0).fit(adata.obsm['X_pca']) # type: ignore
        adata.obs['kmeans_5'] = kmeans.labels_.astype(str)
        sc.pl.umap(adata,color='kmeans_5')


def diff_analysis(view_plots = True, save_plots=False):
    print("loading matrix...")
    adata = sc.read_h5ad(str(DATA_DIR / "tmp" / "adata_tmp.h5ad"))
    # print("ranking gene groups...")
    # sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon",reference="rest", use_raw=False )

    if view_plots:
        rgn = sc.pl.rank_genes_groups(adata, n_genes=20,sharey=False, save=save_plots)
        rgd = sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, standard_scale="var",save=save_plots)


    print("saving progress...")
    adata.write_h5ad(
        str(DATA_DIR / "tmp" / "adata_tmp.h5ad"),
        compression="lzf"
    )
    
    marker_df = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(20)
    print(marker_df)

def pyrimidize_morphology():

    for file in (DATA_DIR / "morphology_focus").iterdir():
        if file.is_file():
            create_zarr_pyramid(
                file,
                DATA_DIR / "morphology_focus" / "pyrimidized" /("pyrimidized_" + str(file.name)),
                downsample_factor=2,
                n_levels=6,
                tile_size=1024,
            )
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
def compute_mask(adata, genes, threshold, is_new=True):
    if is_new:
        genes = genes[genes != ""]
        sc.tl.score_genes(adata, gene_list=genes, score_name='tmp_cell_score')
    mask = adata.obs["tmp_cell_score"] > threshold

    return mask
def compute_score(adata, genes):
    sc.tl.score_genes(adata, gene_list=genes, score_name='tmp_cell_score')
    return adata.obs["tmp_cell_score"].to_numpy().copy()  # copy! column gets overwritten next call
def _recolor(info):
    info[0].obs["annotation"] = pd.Categorical(info[0].obs["annotation"])
    groups = info[0].obs["annotation"]
    assigned_groups = [g for g in groups.cat.categories if g != "Unassigned"]

    cmap1 = plt.get_cmap("tab20") # type: ignore
    cmap2 = plt.get_cmap("tab20b")# type: ignore
    cmap3 = plt.get_cmap("tab20c")# type: ignore
    colors_list = (
        [cmap1(i) for i in range(cmap1.N)] +
        [cmap2(i) for i in range(cmap2.N)] +
        [cmap3(i) for i in range(cmap3.N)]
    )
    color_map = {g: colors_list[i] for i, g in enumerate(assigned_groups)}
    color_map["Unassigned"] = (0.0, 0.0, 0.0, 1.0)

    info[0].uns["annotation_colors"] = [
        mcolors.to_hex(color_map[c]) for c in groups.cat.categories
    ]

    codes, uniques = pd.factorize(groups)
    lut = np.array([color_map[u] for u in uniques])
    colors = lut[codes]

    info[2].set_facecolors(colors)
    info[3].set_facecolors(colors)
    info[1].update_legend(color_map)   # <-- new
    info[1].canvas.draw_idle()
def update_figs(info, app, thresh, recomp=True):
    current_df = app.get_dataframe()
    is_valid = current_df.isin(app.valid_genes).all().all()

    if not is_valid:
        print("Invalid association")
        return

    app.result_df = current_df
    markers = current_df.to_dict(orient='list')

    if not hasattr(app, "_score_cache"):
        app._score_cache = {}

    # Force plain object dtype so arbitrary group_name strings can always be assigned
    info[0].obs["annotation"] = pd.Series(
        ["Unassigned"] * info[0].n_obs, index=info[0].obs.index, dtype=object
    )

    for group_name, genes in markers.items():
        if len(genes) == 0:
            continue

        gene_key = frozenset(genes)
        cached = app._score_cache.get(group_name)

        if cached is None:
            score = compute_score(info[0], genes)
            app._score_cache[group_name] = (gene_key, score)
        elif recomp and cached[0] != gene_key:
            score = compute_score(info[0], genes)
            app._score_cache[group_name] = (gene_key, score)
        else:
            score = cached[1]

        mask = score > thresh[group_name]
        info[0].obs.loc[mask, "annotation"] = group_name  # object dtype -> no category restriction

    _recolor(info)
    
def annotation_complete():

    global editor

    assert editor is not None
    df = editor.get_dataframe()

    print("Annotation Complete")

def extract_hematoxylin(he_image):
    # Ensure channels are last: (C, H, W) -> (H, W, C)
    if he_image.ndim == 3 and he_image.shape[0] in (3, 4):
        he_image = np.moveaxis(he_image, 0, -1)

    # If 4-channel (RGBA), slice to 3 channels (RGB)
    if he_image.ndim == 3 and he_image.shape[2] == 4:
        he_image = he_image[:, :, :3]

    gray = cv2.cvtColor(he_image, cv2.COLOR_BGR2GRAY)
    inverted_gray = cv2.bitwise_not(gray)
    return inverted_gray
def prep_for_sift(img):
    # 1. Remove trailing/leading 1-dimensions
    img = np.squeeze(img)[6]
    
    # 2. Check if empty
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or failed to load!")
        
    # 3. Handle non-finite numbers (NaNs/Infs) if present
    if not np.isfinite(img).all():
        img = np.nan_to_num(img)

    # 4. Convert to 8-bit uint8
    if img.dtype != np.uint8:
        # Scale range [min, max] -> [0, 255]
        img_norm = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) # type: ignore
        img_8u = img_norm.astype(np.uint8)
    else:
        img_8u = img

    return img_8u
def level_shape(path, level, series=0):
    with tifffile.TiffFile(path) as tf:
        return tf.series[series].levels[level].shape
def align_images(he_path, dapi_path, lvl=4):
    matr_loc = DATA_DIR/"tmp"/"he_affine_transform.csv"
    if (matr_loc).is_file():
        affine_matrix = np.loadtxt(str(matr_loc), delimiter=',')
        return affine_matrix
    # 1. Load the images
    # he_image: Moving image (RGB)
    # dapi_image: Fixed reference image (Grayscale/Single-channel)
    he_image = tifffile.imread(he_path, series=0, level=lvl)
    dapi_image = tifffile.imread(dapi_path, series=0, level=lvl)

    dapi_image = prep_for_sift(dapi_image)

    he_shape_lvl  = level_shape(he_path, lvl)
    he_shape_0    = level_shape(he_path, 0)
    dapi_shape_lvl = level_shape(dapi_path, lvl)
    dapi_shape_0   = level_shape(dapi_path, 0)

    he_ds_x = he_shape_0[-1] / he_shape_lvl[-1]
    he_ds_y = he_shape_0[-2] / he_shape_lvl[-2]
    dapi_ds_x = dapi_shape_0[-1] / dapi_shape_lvl[-1]
    dapi_ds_y = dapi_shape_0[-2] / dapi_shape_lvl[-2]

    print("HE level0 shape:", he_shape_0, " level%d shape:" % lvl, he_shape_lvl)
    print("DAPI level0 shape:", dapi_shape_0, " level%d shape:" % lvl, dapi_shape_lvl)
    print("HE downsample x/y:", he_ds_x, he_ds_y)
    print("DAPI downsample x/y:", dapi_ds_x, dapi_ds_y)
    
    # 2. Preprocess H&E to match DAPI's modality (bright nuclei on dark background)
    he_processed = extract_hematoxylin(he_image)
    
    # 3. Initialize SIFT detector
    sift = cv2.SIFT_create() # type: ignore
    
    # 4. Find keypoints and descriptors
    kp_he, des_he = sift.detectAndCompute(he_processed, None)
    kp_dapi, des_dapi = sift.detectAndCompute(dapi_image, None)
    
    # 5. Match features using FLANN Matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params) # type: ignore
    
    matches = flann.knnMatch(des_he, des_dapi, k=2)
    
    # 6. Filter matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
            
    if len(good_matches) < 3:
        raise ValueError("Not enough matching keypoints found between images.")
        
    # 7. Extract coordinates of matched keypoints
    src_pts = np.float32([kp_he[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) # type: ignore
    dst_pts = np.float32([kp_dapi[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) # type: ignore
    
    # 8. Estimate Affine Transformation Matrix (accounts for rotation, scale, translation)
    # RANSAC cleans out false geometric matches
    affine_matrix, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC)

    print("good_matches:", len(good_matches))
    print("inliers:", int(inliers.sum()) if inliers is not None else None, "/", len(inliers) if inliers is not None else 0)
    print("raw affine_matrix:\n", affine_matrix)

    a, b = affine_matrix[0, 0], affine_matrix[0, 1]
    c, d = affine_matrix[1, 0], affine_matrix[1, 1]
    col1_norm = np.hypot(a, c)  # effective x-scale
    col2_norm = np.hypot(b, d)  # effective y-scale
    print("column norms (x-scale, y-scale):", col1_norm, col2_norm)
    print("anisotropy ratio:", col2_norm / col1_norm)

    scale_to_level0 = 2**lvl  # 2^4 = 16
    affine_matrix[0, 2] *= scale_to_level0
    affine_matrix[1, 2] *= scale_to_level0


    np.savetxt(str(DATA_DIR/"tmp"/"he_affine_transform.csv"), affine_matrix, delimiter=',')
    
    return affine_matrix

    
def annotate_cells(mode: Literal["cmd", "int"], auto=True, view_figures= True):
    import celltypist
    global editor, plot_window

    print("loading matrix...")
    adata = sc.read_h5ad(str(DATA_DIR / "tmp" / "adata_tmp.h5ad"))
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
            adata.obs['leiden'],  # type: ignore
            adata.obs['majority_voting'],# type: ignore
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

    if mode == "cmd":

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

    commands = {
                    "[ec]annotation_name": "ends the creation of current cluster mask",
                    "[end]" : "complete cluster annotation and close",
                    "[set_thresh]": "change the default threshold (between 0 and 10)",
                    "[rcc]": "restart current cluster",
                    "[RAC]": "restart all clusters",
                    "[help]": "print instructions and command list"
                }

    valid_genes = adata.var_names_make_unique()
    valid_genes = adata.var_names.tolist()
    valid_genes.append("")

    com = None
    tmp_markers = []
    markers = {}
    thresh = 0.8
    cname = "tmp"
    needs_update = False

    coords = adata.obsm["X_umap"]
    adata.obs["annotation"] = "Unassigned"
    with open(DATA_DIR / "tmp" / "cell_objects_loaded.pkl", "rb") as f:
        cells = pickle.load(f)

    
    ordered_map = {name: i for i, name in enumerate(adata.obs_names)}
    filtered_cells = [x for x in cells if x.id in ordered_map]
    filtered_cells.sort(key=lambda x: ordered_map[x.id])
    cell_boundries = [c.boundry for c in filtered_cells]
    cell_vertecies = [np.array(b.exterior.coords) for b in cell_boundries]
    del cell_boundries,ordered_map, filtered_cells,cells

    tfs = [
        f.absolute()
        for f in (DATA_DIR / "morphology_focus" / "pyrimidized").iterdir()
    ]

    tfs = sorted(tfs, key=lambda x: x.name)
    hetfs = tifffile.TiffFile(str(DATA_DIR / "WTA_Preview_FFPE_Breast_Cancer_he_image.ome.tif")) #..FILENAME
    trans_matrix = align_images(str(DATA_DIR / "WTA_Preview_FFPE_Breast_Cancer_he_image.ome.tif"), str(DATA_DIR / "morphology.ome.tif"))
    # type: ignore
    plot_window = ScatterPlotWindow(
        coords,
        tmp_cpd["unassinged"],
        cell_vertecies,
        tfs,
        hetfs,
        trans_matrix


    )

    plot_window.show()

    ext_info = (
        adata,
        plot_window,
        plot_window.scatter,
        plot_window.poly_collection
    )
    if mode == "int":

        # 2. Initialize the DataFrame with empty (None/NaN) values
        editor = DataFrameEditor(
            diff_output,
            valid_genes,
            ext_info,
            thresh
        )

        editor.updateFigs.connect(update_figs)
        editor.accepted.connect(annotation_complete)

        editor.show()  # non-modal, no input restrictions on other windows

        loop = QEventLoop()
        editor.finished.connect(loop.quit)
        loop.exec()    # blocks annotate_cells here until editor closes

        if editor.result_df is not None:
            adata.uns["marker_genes"] = editor.result_df.to_dict(orient="list")
        print("saving annotations...")
        ext_info[0].write_h5ad(
            str(DATA_DIR / "tmp" / "adata_tmp.h5ad"),
            compression="lzf"
        )

def create_spatial_zarr(DATA_DIR, adata: ad.AnnData):
    from napari_spatialdata import Interactive
    from spatialdata.models import (
        Image2DModel,
        ShapesModel,
        TableModel
    )
    import spatialdata.models
    from spatialdata.transformations import Identity,Scale
    import spatialdata_io as sdio
    import dask.array as da
    import inspect

    print(sd.__version__)
    print(sdio.__version__)
    print(inspect.signature(Image2DModel.parse))
    print([x for x in dir(spatialdata.models) if "Image" in x or "Scale" in x])

    # -------------------------
    # Image (lazy)
    # -------------------------
        
    tif = tifffile.TiffFile(DATA_DIR / "morphology.ome.tif")
    root = zarr.open(tif.series[0].aszarr(), mode="r")

    pixel_size = 0.2125
    # highest resolution
    image = da.from_zarr(root["0"]) # type: ignore
    print(image.chunksize)  # sanity check
    image = image.rechunk((1, 1024, 1024))

    image_element = Image2DModel.parse(
        image,
        dims=("c", "y", "x"),
        transformations={"global": Scale([pixel_size, pixel_size], axes=("x", "y"))},
        scale_factors=[2, 2, 2, 2, 2, 2, 2],
        chunks=(1, 1024, 1024),
    )


    # -------------------------
    # Cell boundaries
    # -------------------------

    with open(DATA_DIR / "tmp" / "cell_objects_loaded.pkl", "rb") as f:
        cells = pickle.load(f)

    gdf = gpd.GeoDataFrame(
        {"cell_id": [c.id for c in cells], "geometry": [c.boundry for c in cells]},
        geometry="geometry",
    )# type: ignore
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.5) # type: ignore
    gdf.index = np.arange(len(gdf))
    gdf.index.name = None

    shapes_element = ShapesModel.parse(gdf, transformations={"global": Identity()})

    # table: map adata rows onto the SAME integers, via matching barcode strings
    id_to_int = {cid: i for i, cid in enumerate(gdf["cell_id"])} # type: ignore

    # 1. Drop heavy pairwise graphs and embeddings
    adata.obsp.clear()
    adata.obsm.clear()

    # 2. Clear unneeded analysis metadata from uns (keep spatialdata_attrs!)
    keys_to_remove = ['neighbors', 'pca', 'umap', 'rank_genes_groups', 'dendrogram_leiden', 'hvg', 'log1p']
    for key in keys_to_remove:
        adata.uns.pop(key, None)
        
    adata.obs["instance_id"] = adata.obs_names.map(id_to_int)

    n_before = adata.n_obs
    adata = adata[adata.obs["instance_id"].notna()].copy()
    print(f"dropped {n_before - adata.n_obs} cells with no matching shape")
    adata.obs["instance_id"] = adata.obs["instance_id"].astype(int) # type: ignore
    adata.obs["region"] = "cell_boundaries"

    table_element = TableModel.parse(
        adata, region="cell_boundaries", region_key="region", instance_key="instance_id",
    )


    # -------------------------
    # SpatialData object
    # -------------------------

    sdata = sd.SpatialData(
        images={
            "tissue_image": image_element
        },
        shapes={
            "cell_boundaries": shapes_element
        },
        tables={
            "expression_table": table_element
        }
    )
    shapes_idx = sdata.shapes["cell_boundaries"].index
    table_ids = sdata.tables["expression_table"].obs["instance_id"]

    print("shapes index:", shapes_idx.dtype, shapes_idx[:5].tolist())
    print("table instance_id:", table_ids.dtype, table_ids.values[:5])
    print("overlap:", len(set(shapes_idx) & set(table_ids)), "/", len(table_ids))
    viewer = Interactive(sdata)
    viewer.run()
    #sdata.write(str(DATA_DIR / "tmp" / "sdata.zarr"), overwrite=True)

def load_interactive():
    from napari_spatialdata import Interactive

    sdata = sd.read_zarr(str(DATA_DIR / "tmp" / "sdata.zarr"))
    viewer = Interactive(sdata)
    viewer.run()


def neighbor_analysis(adata):
    import squidpy as sq

    print(adata)

    with open(DATA_DIR / "tmp" / "cell_objects_loaded.pkl", "rb") as f:
        cells = pickle.load(f)

    # Map cell ID -> centroid
    centroid_lookup = {
        str(cell.id): (
            cell.boundry.centroid.x,
            cell.boundry.centroid.y
        )
        for cell in cells
    }

    # Build centroid array in the same order as adata.obs_names
    missing = []
    centroids = []

    for cell_id in adata.obs_names.astype(str):
        if cell_id in centroid_lookup:
            centroids.append(centroid_lookup[cell_id])
        else:
            missing.append(cell_id)

    if missing:
        print(f"Warning: {len(missing)} cells were not found in the pickle file.")

    adata.obsm["spatial"] = np.asarray(centroids, dtype=float)
    adata_tmp = ad.AnnData(
        X=adata.layers["counts"],
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )

    adata.raw = adata_tmp


    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key="leiden")
    sq.gr.ligrec(adata, cluster_key="leiden", n_perms=1000, use_raw=True)

    sc.pl.embedding(adata, basis="spatial", color="leiden")
    sq.pl.ligrec(adata, n_perms=100, cluster_key="leiden")

    print("saving progress...")
    adata.write_h5ad(
        str(DATA_DIR / "tmp" / "adata_tmp.h5ad"),
        compression="lzf"
    )

    sq.pl.spatial_scatter(adata, color="leiden", library_id = None) # type: ignore
    sq.pl.ligrec(adata, n_perms=100, cluster_key="leiden")
                 

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
    #format_h5(DATA_DIR / "tmp" / "trns_with_cellID_regroup.parquet")
    #create_UMAP(r"F:\SPSC-RNA-Seq\WTA_Preview_FFPE_Breast_Cancer_outs\tmp\cell_matrix.h5",view_plots=False)
    #diff_analysis(view_plots=True, save_plots=True)


    #pyrimidize_morphology()
    annotate_cells(mode="int",auto=False)
    # print("loading matrix...")
    # adata = sc.read_h5ad(str(DATA_DIR / "tmp" / "adata_tmp.h5ad"))
    # print(adata)
    # neighbor_analysis(adata)
    # Using rc_context to set black facecolors for axes and figure
    # with plt.rc_context(
    #     {
    #         "axes.facecolor": "black",
    #         "figure.facecolor": "black",
    #         "axes.labelcolor": "white",
    #         "xtick.color": "white",
    #         "ytick.color": "white",
    #         "text.color": "white",
    #         "axes.edgecolor": "white",
    #     }
    # ):
    #     sc.pl.umap(adata, color="majority_voting", frameon=True)

    # create_spatial_zarr(DATA_DIR, adata)

    #load_interactive()
