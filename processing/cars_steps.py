import os
import json
import numpy as np
import pandas as pd
import rasterio as rio
from glob import glob

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

from cars.pipelines.pipeline import Pipeline

from shareloc.geomodels.rpc import RPC
from shareloc.image import Image as sImage
from shareloc.geofunctions import localization

import tempfile
import shutil
import streamlit as st

import requests
import zipfile

import logging
logging.getLogger().setLevel(logging.INFO)

# download srtm
def get_srtm_tif_name(lat, lon):
    """Download srtm tiles"""
    # longitude: [1, 72] == [-180, +180]
    tlon = (1+np.floor((lon+180)/5)) % 72
    tlon = 72 if tlon == 0 else tlon

    # latitude: [1, 24] == [60, -60]
    tlat = 1+np.floor((60-lat)/5)
    tlat = 24 if tlat == 25 else tlat

    srtm = "https://download.esa.int/step/auxdata/dem/SRTM90/tiff/srtm_%02d_%02d.zip" % (tlon, tlat)
    # srtm = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_%02d_%02d.zip" % (tlon, tlat)
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

    sparse_outdata = {}
    dense_outdata = {}

    my_bar = st.progress(0, text="Download SRTM")
    tempdir = tempfile.mkdtemp()

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
    srtm_arc = os.path.join(tempdir, os.path.basename(srtm))
    srtm_tif = os.path.splitext(srtm_arc)[0]+".tif"

    try:
        r = requests.get(srtm)
        open(srtm_arc, "wb").write(r.content)
        with zipfile.ZipFile(srtm_arc, "r") as zf:
            zf.extractall(path=tempdir)
    except:
        st.warning("Error while downloading the srtm tile...")

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

    config = {}
    config["inputs"] = inputs_conf
    config["output"] = {"out_dir": tempdir}

    config["orchestrator"] = {"mode": "mp", "nb_workers": 1}
    config["applications"] = {"resampling": {"save_epipolar_image": True},
                              "sparse_matching": {"save_matches": True},
                              "dense_matching": {"save_disparity_map": True},
                              "point_cloud_fusion": {"save_points_cloud_as_csv": True},
                              "triangulation": {"save_points_cloud": True}}

    my_bar.progress(10, text="Sparse pipeline: resampling, sparse matching, triangulation, rasterization...")
    sparse_pipeline = Pipeline("sensors_to_sparse_dsm", config, os.getcwd())
    sparse_pipeline.run()

    def fill_outdata(outdata):
        for step in outdata.keys():
            keys_list = outdata[step].keys()
            for key in keys_list:
                outdata_list = list()
                filenames = glob(outdata[step][key])
                for filename in filenames:
                    basename = os.path.basename(filename)
                    ext = os.path.splitext(filename)[-1]
                    if ext == ".tif":
                        with rio.open(filename) as dt:
                            outdata_list.append({"array": dt.read(),
                                            "profile": dt.profile})
                    elif ext == ".npy":
                        outdata_list.append(np.load(filename))
                    elif ext == ".csv":
                        outdata_list.append(pd.read_csv(filename, delimiter=','))

                if len(outdata_list) == 1:
                    outdata[step][key] = outdata_list[0]
                else:
                    outdata[step][key] = pd.concat(outdata_list)

    pairdir = os.path.join(tempdir, "_".join(config["inputs"]["pairing"][0]))

    sparse_outdata = {"resampling": {"left": os.path.join(pairdir, "epi_img_left.tif"),
                                     "right": os.path.join(pairdir, "epi_img_right.tif")},
                      "matching": {"disp": os.path.join(pairdir, "filtered_matches.npy")},
                      "triangulation": {"pc": os.path.join(pairdir, "epi_pc", "*.csv")},
                      "rasterization": {"dsm": os.path.join(tempdir, "dsm.tif")}}

    with open(os.path.join(tempdir, "refined_config_dense_dsm.json"), "r") as json_reader:
        config_new = json.load(json_reader)

    fill_outdata(sparse_outdata)

    config_new["applications"] = config["applications"]

    dense_outdata = {"resampling": {"left": os.path.join(pairdir, "epi_img_left.tif"),
                                    "right": os.path.join(pairdir, "epi_img_right.tif")},
                     "matching": {"disp": os.path.join(pairdir, "epi_disp.tif")},
                     "triangulation":{"x": os.path.join(pairdir, "epi_pc_X.tif"),
                                      "y": os.path.join(pairdir, "epi_pc_X.tif"),
                                      "z": os.path.join(pairdir, "epi_pc_Z.tif"),
                                      "pc": os.path.join(tempdir, "points_cloud/*.csv")},
                     "rasterization": {"dsm": os.path.join(tempdir, "dsm.tif"),
                                       "clr": os.path.join(tempdir, "clr.tif")}}

    my_bar.progress(30, text="Dense pipeline: resampling, dense matching, triangulation, rasterization...")
    dense_pipeline = Pipeline("sensors_to_dense_dsm", config_new, os.getcwd())
    dense_pipeline.run()
    my_bar.progress(100, text="Pipelines completed")

    fill_outdata(dense_outdata)

    remove_temp_data(image1, temp_image1)
    remove_temp_data(image2, temp_image2)
    remove_temp_data(geomodel1, temp_geomodel1)
    remove_temp_data(geomodel2, temp_geomodel2)

    return sparse_outdata, dense_outdata
