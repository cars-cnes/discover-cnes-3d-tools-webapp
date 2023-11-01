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

from shareloc.geomodels.rpc import RPC
from shareloc.image import Image as sImage
from shareloc.geofunctions import localization

import tempfile
import shutil
import streamlit as st

import requests
import zipfile

# download srtm
def get_srtm_tif_name(lat, lon):
    """Download srtm tiles"""
    # longitude: [1, 72] == [-180, +180]
    tlon = (1+np.floor((lon+180)/5)) % 72
    tlon = 72 if tlon == 0 else tlon

    # latitude: [1, 24] == [60, -60]
    tlat = 1+np.floor((60-lat)/5)
    tlat = 24 if tlat == 25 else tlat

    srtm = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_%02d_%02d.zip" % (tlon, tlat)
    return srtm

def get_temp_data(data):
    if isinstance(data, str) is False:
        data_suffix = os.path.splitext(data.name)[-1]
        __, temp_data = tempfile.mkstemp(suffix=data_suffix)
        with open(temp_data, "wb") as f:
            f.write(data.getbuffer())
    else:
        temp_data = data
    return temp_data

def remove_temp_data(data, temp_data):
    if isinstance(data, str) is False:
        os.remove(temp_data)

def run(image1, image2, geomodel1, geomodel2):
    temp_image1 = get_temp_data(image1)
    temp_image2 = get_temp_data(image2)
    temp_geomodel1 = get_temp_data(geomodel1)
    temp_geomodel2 = get_temp_data(geomodel2)

    outdata = {}

    my_bar = st.progress(0, text="Download SRTM")
    tempdir = tempfile.mkdtemp()
    mycwd = os.getcwd()
    os.chdir(tempdir)

    # get srtm tile
    shareloc_img = sImage(temp_image1)
    shareloc_mdl = RPC.from_any(temp_geomodel1)
    loc = localization.Localization(
        shareloc_mdl,
        image=shareloc_img)

    center = loc.direct(shareloc_img.nb_rows/2,
                        shareloc_img.nb_columns/2,
                        using_geotransform=True)[0]

    srtm = get_srtm_tif_name(center[1], center[0])
    r = requests.get(srtm)
    srtm_bn = os.path.basename(srtm)
    srtm_tif = os.path.splitext(srtm_bn)[0]+".tif"
    open(srtm_bn, "wb").write(r.content)
    with zipfile.ZipFile(srtm_bn, "r") as zf:
        zf.extract(srtm_tif)

    # Import external plugins
    import_plugins()

    inputs_conf = {
        "sensors": {
            "left": {
                "image": temp_image1,
                "geomodel": temp_geomodel1
            },
            "right": {
                "image": temp_image2,
                "geomodel": temp_geomodel2
            }
        },
        "pairing": [["left", "right"]],
    }

    if os.path.exists(srtm_tif):
        inputs_conf["initial_elevation"] = srtm_tif

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
                                                  out_dir=tempdir)
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
                                                        tempdir)

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

    outdata["resampling"] = {"left": {"data": new_epipolar_image_left, "tag": "im", "nodata": 0}, "right": {"data": new_epipolar_image_right, "tag": "im", "nodata": 0}}

    my_bar.progress(30, text="Dense pipeline: matching")
    epipolar_disparity_map = dense_matching_application.run(
        new_epipolar_image_left,
        new_epipolar_image_right,
        orchestrator=cars_orchestrator,
        disp_min=disp_min,
        disp_max=disp_max,
    )

    outdata["matching"] = {"disp": {"data": epipolar_disparity_map, "tag": "disp", "nodata": 0}}

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

    outdata["triangulation"] = {"x": {"data": epipolar_points_cloud, "tag": "x", "nodata": np.nan}, "y": {"data": epipolar_points_cloud, "tag": "y", "nodata": np.nan}, "z": {"data": epipolar_points_cloud, "tag": "z", "nodata": np.nan}}

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

    outdata["rasterization"] = dsm

    shutil.rmtree(tempdir, ignore_errors=True)
    my_bar.empty()
    os.chdir(mycwd)

    remove_temp_data(image1, temp_image1)
    remove_temp_data(image2, temp_image2)
    remove_temp_data(geomodel1, temp_geomodel1)
    remove_temp_data(geomodel2, temp_geomodel2)

    return outdata
