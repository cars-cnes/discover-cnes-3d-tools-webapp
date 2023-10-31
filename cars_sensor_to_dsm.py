import os
import numpy as np
import rasterio as rio

from cars import __version__
from cars import import_plugins
from cars.pipelines.sensor_to_dense_dsm import sensors_inputs
from cars.applications.application import Application
from cars.orchestrator import orchestrator
from cars.applications.grid_generation import grid_correction
from cars.applications.sparse_matching import sparse_matching_tools
from cars.applications.dem_generation import dem_generation_tools
from cars.core import preprocessing
import cars.pipelines.sensor_to_dense_dsm.sensor_dense_dsm_constants as sens_cst

def save_data(cars_ds, file_name, tag, dtype="float32", nodata=-9999):
    """
    Save CarsDataset
    """

    # create descriptor
    desc = None

    # Save tiles
    for row in range(cars_ds.shape[0]):
        for col in range(cars_ds.shape[1]):
            if cars_ds[row, col] is not None:
                if desc is None:
                    desc = cars_ds.generate_descriptor(
                        cars_ds[row, col],
                        file_name,
                        tag=tag,
                        dtype=dtype,
                        nodata=nodata,
                    )
                cars_ds.run_save(
                    cars_ds[row, col],
                    file_name,
                    tag=tag,
                    descriptor=desc,
                )

    # close descriptor
    desc.close()


def run(image1, image2, geomodel1, geomodel2,
        output_dir, outdata, my_bar):

    # Import external plugins
    import_plugins()

    inputs_conf = {
        "sensors": {
            "left": {
                "image": image1,
                "geomodel": geomodel1
            },
            "right": {
                "image": image2,
                "geomodel": geomodel2
            }
        },
        "pairing": [["left", "right"]]
    }

    inputs = sensors_inputs.sensors_check_inputs(inputs_conf)

    # Get geometry plugin
    (_, _, geom_plugin_without_dem_and_geoid, geom_plugin_with_dem_and_geoid) = \
        sensors_inputs.check_geometry_plugin(inputs, None)

    epipolar_grid_generation_application = Application("grid_generation")
    resampling_application = Application("resampling")
    sparse_matching_application = Application("sparse_matching")
    dem_generation_application = Application("dem_generation")
    dense_matching_application = Application("dense_matching")
    triangulation_application = Application("triangulation")
    pc_fusion_application = Application("point_cloud_fusion")
    rasterization_application = Application("point_cloud_rasterization")

    # Use sequential mode in notebook
    orchestrator_conf = {"mode": "sequential"}
    cars_orchestrator = orchestrator.Orchestrator(orchestrator_conf=orchestrator_conf,
                                                  out_dir=output_dir)
    _, sensor_image_left, sensor_image_right = \
        sensors_inputs.generate_inputs(inputs,
                                       geom_plugin_without_dem_and_geoid)[0]
    geom_plugin = geom_plugin_with_dem_and_geoid
    if inputs["initial_elevation"] is None:
        geom_plugin = geom_plugin_without_dem_and_geoid


    my_bar.progress(5, text="Sparse pipeline: resampling")
    grid_left, grid_right = epipolar_grid_generation_application.run(
        sensor_image_left,
        sensor_image_right,
        geom_plugin,
        orchestrator=cars_orchestrator
    )

    epipolar_image_left, epipolar_image_right = resampling_application.run(
        sensor_image_left,
        sensor_image_right,
        grid_left,
        grid_right,
        orchestrator=cars_orchestrator,
        margins=sparse_matching_application.get_margins()
    )

    my_bar.progress(10, text="Sparse pipeline: matching")
    epipolar_matches_left, _ = sparse_matching_application.run(
        epipolar_image_left,
        epipolar_image_right,
        grid_left.attributes["disp_to_alt_ratio"],
        orchestrator=cars_orchestrator
    )

    matches_array = sparse_matching_application.filter_matches(epipolar_matches_left,
                                                               orchestrator=cars_orchestrator)

    grid_correction_coef, corrected_matches_array, _, _, _ = \
        grid_correction.estimate_right_grid_correction(matches_array, grid_right)

    corrected_grid_right = grid_correction.correct_grid(grid_right,
                                                        grid_correction_coef,
                                                        False,
                                                        output_dir)

    my_bar.progress(15, text="Sparse pipeline: triangulation")
    triangulated_matches = dem_generation_tools.triangulate_sparse_matches(
        sensor_image_left,
        sensor_image_right,
        grid_left,
        corrected_grid_right,
        corrected_matches_array,
        geometry_plugin=geom_plugin,
    )

    dmin, dmax = sparse_matching_tools.compute_disp_min_disp_max(
        triangulated_matches,
        cars_orchestrator,
        disp_margin=(
            sparse_matching_application.get_disparity_margin()
        ),
        disp_to_alt_ratio=grid_left.attributes["disp_to_alt_ratio"]
    )

    my_bar.progress(20, text="Dense pipeline: resampling")
    dense_matching_margins, disp_min, disp_max = dense_matching_application.get_margins(
        grid_left, disp_min=dmin, disp_max=dmax)

    new_epipolar_image_left, new_epipolar_image_right = resampling_application.run(
        sensor_image_left,
        sensor_image_right,
        grid_left,
        corrected_grid_right,
        orchestrator=cars_orchestrator,
        margins=dense_matching_margins,
        optimum_tile_size=(
            dense_matching_application.get_optimal_tile_size(
                disp_min,
                disp_max,
                cars_orchestrator.cluster.checked_conf_cluster[
                    "max_ram_per_worker"
                ],
            )
        ),
        add_color=True,
    )

    save_data(new_epipolar_image_left,
              outdata["resampling"]["left"], "im",
              nodata=0)
    save_data(new_epipolar_image_right,
              outdata["resampling"]["right"], "im",
              nodata=0)

    my_bar.progress(30, text="Dense pipeline: matching")
    epipolar_disparity_map = dense_matching_application.run(
        new_epipolar_image_left,
        new_epipolar_image_right,
        orchestrator=cars_orchestrator,
        disp_min=disp_min,
        disp_max=disp_max,
    )


    save_data(epipolar_disparity_map,
              outdata["matching"]["disp"], "disp",
              nodata=0)

    epsg = preprocessing.compute_epsg(
        sensor_image_left,
        sensor_image_right,
        grid_left,
        corrected_grid_right,
        geom_plugin_with_dem_and_geoid,
        orchestrator=cars_orchestrator,
        disp_min=disp_min,
        disp_max=disp_max
    )


    my_bar.progress(80, text="Dense pipeline: triangulation")
    epipolar_points_cloud = triangulation_application.run(
        sensor_image_left,
        sensor_image_right,
        new_epipolar_image_left,
        grid_left,
        corrected_grid_right,
        epipolar_disparity_map,
        epsg,
        geom_plugin_without_dem_and_geoid,
        orchestrator=cars_orchestrator,
        uncorrected_grid_right=grid_right,
        geoid_path=inputs[sens_cst.GEOID],
        disp_min=disp_min,
        disp_max=disp_max,
    )

    for dim in ["x", "y", "z"]:
        save_data(epipolar_points_cloud,
                  outdata["triangulation"][dim], dim,
                  nodata=np.nan)

    current_terrain_roi_bbox = preprocessing.compute_terrain_bbox(
        sensor_image_left,
        sensor_image_right,
        new_epipolar_image_left,
        grid_left,
        corrected_grid_right,
        epsg,
        geom_plugin_with_dem_and_geoid,
        resolution=rasterization_application.get_resolution(),
        disp_min=disp_min,
        disp_max=disp_max,
        orchestrator=cars_orchestrator
    )
    terrain_bounds, optimal_terrain_tile_width = preprocessing.compute_terrain_bounds(
        [current_terrain_roi_bbox],
        resolution=rasterization_application.get_resolution()
    )

    merged_points_clouds = pc_fusion_application.run(
        [epipolar_points_cloud],
        terrain_bounds,
        epsg,
        orchestrator=cars_orchestrator,
        margins=rasterization_application.get_margins(),
        optimal_terrain_tile_width=optimal_terrain_tile_width
    )

    my_bar.progress(95, text="Dense pipeline: rasterization")
    dsm = rasterization_application.run(
        merged_points_clouds,
        epsg,
        orchestrator=cars_orchestrator
    )

    dsm_utm_path = os.path.join(output_dir, "dsm_utm.tif")
    save_data(dsm, dsm_utm_path, "hgt", nodata=-32768)
    dsm_path = outdata["rasterization"]

    dst_crs = 'EPSG:4326'

    with rio.open(dsm_utm_path) as src:
        transform, width, height = rio.warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rio.open(dsm_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                rio.warp.reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=rio.warp.Resampling.bilinear)

    os.remove(dsm_utm_path)
