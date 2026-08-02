from pathlib import Path

import numpy as np
import tifffile
import zarr

from skimage.transform import resize


def create_zarr_pyramid(
    input_tiff,
    output_dir,
    downsample_factor=2,
    n_levels=6,
    tile_size=1024,
):
    """
    Convert a single-channel OME-TIFF into a multiresolution Zarr pyramid.

    IMPORTANT:
        This function intentionally ignores the OME multidimensional
        series metadata and treats TIFF page 0 as ONE 2D image.

        This is useful when each morphology_focus OME-TIFF is physically
        a single-channel image, but the shared OME metadata incorrectly
        declares multiple channels.

    The source TIFF is read in small spatial tiles, so the entire image
    is never loaded into RAM.

    Expected physical TIFF page shape:
        (Y, X)

    Example:
        (28048, 46543)

    Output pyramid shapes:
        level_0: (1, 28048, 46543)
        level_1: (1, 14024, 23271)
        level_2: (1, 7012, 11635)
        ...
    """

    input_tiff = Path(input_tiff)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(input_tiff) as tif:

        # ------------------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT use tif.series[0] here.
        #
        # The OME metadata may incorrectly describe this file as
        # having 4 channels.
        #
        # Instead, use the first physical TIFF page directly.
        # ------------------------------------------------------------

        page = tif.pages[0]

        print("Source TIFF:")
        print("  file:", input_tiff)
        print("  pages:", len(tif.pages))
        print("  page 0 shape:", page.shape)
        print("  page 0 dtype:", page.dtype)
        print("  page 0 axes:", page.axes)

        # The actual image should be a 2D page.
        if page.ndim != 2:
            raise ValueError(
                f"Expected TIFF page 0 to be 2D (Y, X), "
                f"but got shape {page.shape}"
            )

        full_height, full_width = page.shape
        dtype = page.dtype

        print()
        print("Using physical TIFF page 0 as:")
        print("  shape:", (full_height, full_width))
        print("  dtype:", dtype)
        print("  OME channel metadata is ignored.")

        # ------------------------------------------------------------
        # Open the physical TIFF page lazily through Zarr.
        #
        # tifffile.aszarr() can expose the TIFF pages without reading
        # the entire image into RAM.
        #
        # Since we want page 0 specifically, we use the page-level
        # Zarr store rather than the OME series.
        # ------------------------------------------------------------

        source = zarr.open(
            page.aszarr(),
            mode="r",
        )

        print("  Zarr source shape:", source.shape) # type: ignore
        print("  Zarr source chunks:", source.chunks)# type: ignore

        if source.ndim != 2:# type: ignore
            raise ValueError(
                f"Expected lazy source to be 2D, got {source.shape}"# type: ignore
            )

        # ------------------------------------------------------------
        # Generate each pyramid level.
        # ------------------------------------------------------------

        for level_index in range(n_levels):

            scale = downsample_factor ** level_index

            out_height = max(
                1,
                full_height // scale,
            )

            out_width = max(
                1,
                full_width // scale,
            )

            level_path = (
                output_dir / f"level_{level_index}.zarr"
            )

            print()
            print("=" * 60)
            print(f"Level {level_index}")
            print(f"  scale: 1/{scale}")
            print(
                f"  shape: (1, {out_height}, {out_width})"
            )
            print(f"  output: {level_path}")

            # Don't overwrite an existing level.
            if level_path.exists():
                print("  Already exists -- skipping.")
                continue

            # --------------------------------------------------------
            # Output is explicitly:
            #
            #     (1, Y, X)
            #
            # The leading 1 means one real channel.
            # --------------------------------------------------------

            output = zarr.open(
                level_path,
                mode="w",
                shape=(
                    1,
                    out_height,
                    out_width,
                ),
                chunks=(
                    1,
                    tile_size,
                    tile_size,
                ),
                dtype=dtype,
            )

            # --------------------------------------------------------
            # Process the image tile-by-tile.
            # --------------------------------------------------------

            for y_out in range(
                0,
                out_height,
                tile_size,
            ):

                y_out_end = min(
                    y_out + tile_size,
                    out_height,
                )

                # Corresponding source region.
                y_src = y_out * scale

                y_src_end = min(
                    y_out_end * scale,
                    full_height,
                )

                for x_out in range(
                    0,
                    out_width,
                    tile_size,
                ):

                    x_out_end = min(
                        x_out + tile_size,
                        out_width,
                    )

                    x_src = x_out * scale

                    x_src_end = min(
                        x_out_end * scale,
                        full_width,
                    )

                    # ------------------------------------------------
                    # Read ONLY this small source region.
                    # ------------------------------------------------

                    tile = np.asarray(
                        source[
                            y_src:y_src_end,
                            x_src:x_src_end,
                        ] # type: ignore
                    )

                    target_h = y_out_end - y_out
                    target_w = x_out_end - x_out

                    # ------------------------------------------------
                    # Level 0 doesn't need resizing.
                    # ------------------------------------------------

                    if scale == 1:

                        resized = tile

                    else:

                        resized = resize(
                            tile,
                            (target_h, target_w),
                            order=1,
                            preserve_range=True,
                            anti_aliasing=True,
                        )

                        resized = resized.astype(
                            dtype,
                            copy=False,
                        )

                    # ------------------------------------------------
                    # Write into channel 0.
                    # ------------------------------------------------

                    output[
                        0,
                        y_out:y_out_end,
                        x_out:x_out_end,
                    ] = resized # type: ignore

                # Progress.
                print(
                    f"    row "
                    f"{min(y_out + tile_size, out_height)}"
                    f"/{out_height}"
                )

            print(
                f"  Finished level {level_index}"
            )

    print("\nPyramid complete.")
