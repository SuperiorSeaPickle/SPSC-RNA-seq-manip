import pandas as pd
import plotly.express as px


def assign_gene_to_cell(transcripts_dir, cellBoundres_dir):
    import matplotlib.path as mpath
    import polars as pl

    gc_assosiations = []
    
    #sort transcripts by x then by y save
    # load cells boundry area. read untill new cell id reached
    
    cell = []
    cell_boundries = pl.scan_parquet(cellBoundres_dir)

    unique_ids = (
        cell_boundries
        .select("cell_id")
        .unique()
        .collect()
    )

    for cell_id in unique_ids["cell_id"]:
        cell_data = (
            cell_boundries
            .filter(pl.col("cell_id") == cell_id)
            .collect()
        )
        # process this cell
        print(cell_id, cell_data.shape)

        
    # Define the vector area coordinates
    polygon_coords = [(0, 0), (4, 0), (4, 4), (0, 4)]
    path = mpath.Path(polygon_coords)

    # Check if the point (2, 2) is inside
    is_inside = path.contains_point((2, 2))
    
    if(is_inside):
        gc_assosiations.append()


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