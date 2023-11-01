import os
import streamlit as st
from PIL import Image, ImageMath, ImageStat
from io import StringIO
import numpy as np
import rasterio as rio
import time

from shareloc.geomodels.rpc import RPC
from shareloc.image import Image as sImage
from shareloc.geofunctions import localization

import folium
from streamlit_folium import st_folium
import tarfile
import requests
import tempfile

import cars_sensor_to_dsm
# from localtileserver import get_folium_tile_layer, TileClient

MINI_LOGO ="https://raw.githubusercontent.com/CNES/cars/master/docs/source/images/picto_transparent_mini.png"
st.set_page_config(page_title="cars-webapp",
                   page_icon=MINI_LOGO)

# title
__, center, __ = st.columns((1, 4, 1))
with center:
    st.markdown("![Logo]("+MINI_LOGO+")")

st.markdown(("# CARS, a satellite multi view stereo framework"))

data_select = st.radio("Select the dataset",
                       ["Use demo", "Upload your own data"])

if data_select == "Upload your own data":
    uploaded_files = st.file_uploader("or upload your own data", accept_multiple_files=True)
    uploaded_dict = {None: None}
    choices = []
    for uploaded in uploaded_files:
        choices.append(uploaded.name)
        uploaded_dict[uploaded.name] = uploaded

    image1 = uploaded_dict[st.selectbox("Image 1", choices)]
    image2 = uploaded_dict[st.selectbox("Image 2", choices)]
    geomodel1 = uploaded_dict[st.selectbox("Geomodel 1", choices)]
    geomodel2 = uploaded_dict[st.selectbox("Geomodel 2", choices)]

else:
    image1 = os.path.abspath("demo/img1.tif")
    image2 = os.path.abspath("demo/img2.tif")
    geomodel1 = os.path.abspath("demo/img1.geom")
    geomodel2 = os.path.abspath("demo/img2.geom")


def get_envelope_and_center(image, geomodel):
    if isinstance(image, str) is False:
        image_suffix = os.path.splitext(image.name)[-1]
        __, temp_image = tempfile.mkstemp(suffix=image_suffix)
        with open(temp_image, "wb") as f:
            f.write(image.getbuffer())
    else:
        temp_image = image

    if isinstance(geomodel, str) is False:
        geomodel_suffix = os.path.splitext(geomodel.name)[-1]
        __, temp_geomodel = tempfile.mkstemp(suffix=geomodel_suffix)
        with open(temp_geomodel, "wb") as f:
            f.write(geomodel.getbuffer())
    else:
        temp_geomodel = geomodel

    # read image
    try:
        shareloc_img = sImage(temp_image)
    except rio.errors.RasterioIOError:
        st.warning(image.name + " is not a correct image")
        if isinstance(image, str) is False:
            os.remove(temp_image)
        return None

    try:
        shareloc_mdl = RPC.from_any(temp_geomodel)
    except ValueError:
        st.warning(geomodel.name + " is not a correct geomodel")
        if isinstance(geomodel, str) is False:
            os.remove(temp_geomodel)
        return None

    loc = localization.Localization(
        shareloc_mdl,
        image=shareloc_img)

    envelope = np.array([[0, 0], [0, shareloc_img.nb_columns],
                         [shareloc_img.nb_rows, shareloc_img.nb_columns],
                         [shareloc_img.nb_rows, 0]])

    envelope = loc.direct(envelope[:, 0],
                          envelope[:, 1],
                          using_geotransform=True)

    center = loc.direct(shareloc_img.nb_rows/2,
                        shareloc_img.nb_columns/2,
                        using_geotransform=True)[0]

    if isinstance(image, str) is False:
        os.remove(temp_image)
    if isinstance(geomodel, str) is False:
        os.remove(temp_geomodel)

    return envelope, center


# run cars
if st.button("Run CARS"):
    if None not in [image1, image2, geomodel1, geomodel2]:
        envelope_and_center1 = get_envelope_and_center(image1, geomodel1)
        envelope_and_center2 = get_envelope_and_center(image2, geomodel2)
        if None not in [envelope_and_center1, envelope_and_center2]:
            try:
                st.session_state["outdata"] = cars_sensor_to_dsm.run(image1, image2, geomodel1, geomodel2)
            except Exception as e:
                st.error("CARS encountered a problem during execution")
                st.error(e)
                time.sleep(10)

    else:
        st.warning("Select the dataset first", icon="⚠️")

# map
def create_map_drawing_envelopes(show):
    def draw_envelope(name, image, geomodel, color, m=None):
        envelope_and_center = get_envelope_and_center(image, geomodel)
        if envelope_and_center is not None:
            envelope, center = envelope_and_center
            if m is None:
                m = folium.Map(location=(center[1], center[0]),
                               zoom_start=16)

            fg = folium.FeatureGroup(name=name, show=show)
            m.add_child(fg)
            folium.Polygon(envelope[:, :2][:, [1, 0]],
                           color=color,
                           fill_color=color,
                           opacity=0.1,
                           tooltip=name).add_to(fg)
        return m

    m = draw_envelope("image1", image1, geomodel1, "blue")
    m = draw_envelope("image2", image2, geomodel2, "red", m)

    return m

def show_images():
    m = create_map_drawing_envelopes(show=True)

    if m is not None:
        folium.LayerControl().add_to(m)
        st_data = st_folium(m, height=500, width=500)

def enhance(array, nodata):
    array[array == nodata] = np.nan
    med = np.nanmedian(array)
    std = np.nanstd(array)
    maxi = med + 3*std
    mini = med - 3*std
    array = 255*(array-mini) / (maxi-mini)
    array[array > 255] = 255
    array[array < 0] = 0
    array = np.nan_to_num(array, nan=255)
    return array.astype(np.uint8)

def save_data(cars_ds,
              file_name,
              tag,
              dtype="float32",
              nodata=-9999):

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

def show_epipolar_images(step):
    outdata = st.session_state["outdata"]
    option = st.selectbox(
        'Choose image', outdata[step].keys())

    __, temp = tempfile.mkstemp(suffix=".tif")
    carsdata = outdata[step][option]
    save_data(carsdata["data"], temp,
              tag=carsdata["tag"],
              nodata=carsdata["nodata"])

    with rio.open(temp) as src:
        array = enhance(src.read(1), src.nodata)

    os.remove(temp)
    im = Image.fromarray(array)
    st.image(im.convert("RGB"))


def get_wgs84_dsm_file_from_carsdata(carsdata):
    __, temp1 = tempfile.mkstemp(suffix=".tif")
    __, temp2 = tempfile.mkstemp(suffix=".tif")
    save_data(carsdata,
              temp1,
              tag="hgt",
              nodata=-32768)

    dst_crs = 'EPSG:4326'
    with rio.open(temp1) as src:
        transform, width, height = rio.warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rio.open(temp2, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                rio.warp.reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=rio.warp.Resampling.bilinear)
    os.remove(temp1)
    return temp2


def show_rasterization():
    dsm = st.session_state["outdata"]["rasterization"]
    m = create_map_drawing_envelopes(show=False)

    if m is not None:
        temp = get_wgs84_dsm_file_from_carsdata(dsm)

        with rio.open(temp) as src:
            array = np.moveaxis(src.read(), 0, -1)
            nodata = 255*(array!=src.nodata)
            array = enhance(array, src.nodata)
            array = np.dstack((array, array, array, nodata))
            bounds = src.bounds
            bbox = [(bounds.bottom, bounds.left), (bounds.top, bounds.right)]
            folium.raster_layers.ImageOverlay(
                name="dsm",
                image=array,
                bounds=bbox,
                opacity=1,
                interactive=True,
                cross_origin=False,
                zindex=1,
            ).add_to(m)
        os.remove(temp)
        folium.LayerControl().add_to(m)
        st_data = st_folium(m, height=500, width=500)

    # see https://discuss.streamlit.io/t/streamlit-cloud-port-proxying-on-streamlit-io/24748/4
    # if os.path.exists(dsm):
    #     client = TileClient(temp)
    #     t = get_folium_tile_layer(client)
    #     m = folium.Map(location=client.center(), zoom_start=client.default_zoom)
    #     t.add_to(m)
    #     st_data = st_folium(m, height=500, width=500)

    else:
        st.warning("Clic on \"Run CARS\" before", icon="⚠️")

from st_clickable_images import clickable_images

url_images = "https://raw.githubusercontent.com/CNES/cars/master/docs/source/images/"
steps = ["images", "resampling", "matching", "triangulation", "rasterization"]
url_steps = [url_images + ".".join(["dense", step, "drawio.png"]) for step in steps]

col1, col2 = st.columns([1, 3])

with col1:
    clicked = clickable_images(
        paths=url_steps,
        titles=steps,
        img_style={"width": "220%"}
    )

with col2:
    if clicked in [-1, 0]:
        if None not in [image1, image2, geomodel1, geomodel2]:
            show_images()
        else:
            st.warning("Select the dataset first", icon="⚠️")

    elif "outdata" in st.session_state:
        if clicked == 4:
            show_rasterization()
        else:
            show_epipolar_images(steps[clicked])
    else:
        st.warning("Clic on \"Run CARS\" before", icon="⚠️")
