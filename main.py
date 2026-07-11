import spatialdata
#import spatialdata_io
from napari_spatialdata import Interactive

sdata_path = "D:\\SPSC-RNA-Seq\\WTA_Preview_FFPE_Breast_Cancer_outs\\transcripts.zarr"
sdata = spatialdata.read_zarr(sdata_path)

interactive = Interactive(sdata)
interactive.run()