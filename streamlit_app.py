import os
import shutil
import streamlit as st
from PIL import Image, ImageMath, ImageStat
from io import StringIO
import numpy as np
import rasterio as rio
from shareloc.geomodels.rpc import RPC
from shareloc.image import Image as sImage
from shareloc.geofunctions import localization
import folium
from streamlit_folium import st_folium
import tarfile
import zipfile
import requests
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

# input images + geomodel
image1 = "data/image1.tif"
image2 = "data/image2.tif"
geomodel1 = "data/image1.geom"
geomodel2 = "data/image2.geom"
srtm_path = "data/srtm.tif"
output_dir = "out"
outdata = {}

left_path = os.path.join(output_dir, "left.tif")
right_path = os.path.join(output_dir, "right.tif")
outdata["resampling"] = {"left": left_path,
                         "right": right_path}
disp_path = os.path.join(output_dir, "disp.tif")
outdata["matching"] = {"disp": disp_path}

x_path = os.path.join(output_dir, "x.tif")
y_path = os.path.join(output_dir, "y.tif")
z_path = os.path.join(output_dir, "z.tif")

outdata["triangulation"] = {"x": x_path,
                            "y": y_path,
                            "z": z_path}

dsm_path = os.path.join(output_dir, "dsm.tif")
outdata["rasterization"] = dsm_path

def upload_and_save(name, filename):
    uploaded = st.file_uploader(name)
    if uploaded is not None:
        with open(uploaded.name, "wb") as f:
            f.write(uploaded.getbuffer())
            os.rename(uploaded.name, filename)

# download demo
if st.button("Download demo"):
    with st.spinner('Please wait...'):
        r = requests.get("https://github.com/CNES/cars/raw/master/tutorials/data_gizeh_small.tar.bz2")
        arch_and_dest = {"data_gizeh_small/img1.tif": image1,
                         "data_gizeh_small/img1.geom": geomodel1,
                         "data_gizeh_small/img2.tif": image2,
                         "data_gizeh_small/img2.geom": geomodel2}

        open("data_gizeh_small.tar.bz2", "wb").write(r.content)
        with tarfile.open("data_gizeh_small.tar.bz2", "r") as tf:
            for archive, destination in arch_and_dest.items():
                tf.extract(archive)
                os.rename(archive, destination)

        os.rmdir("data_gizeh_small")
        os.remove("data_gizeh_small.tar.bz2")


st.markdown("or upload your own data")
left, right = st.columns((1, 1))
with left:
    upload_and_save("Image 1", image1)
    upload_and_save("Geomodel 1", geomodel1)
with right:
    upload_and_save("Image 2", image2)
    upload_and_save("Geomodel 2", geomodel2)

def get_envelope(image, geomodel):
    # read image
    shareloc_img = sImage(image)
    shareloc_mdl = RPC.from_any(geomodel)
    loc = localization.Localization(
        shareloc_mdl,
        image=shareloc_img)

    envelope = np.array([[0, 0], [0, shareloc_img.nb_columns],
                         [shareloc_img.nb_rows, shareloc_img.nb_columns],
                         [shareloc_img.nb_rows, 0]])

    envelope = loc.direct(envelope[:, 0],
                          envelope[:, 1],
                          using_geotransform=True)
    return envelope

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

# run cars
if st.button("Run CARS"):
    if os.path.exists(image1) \
       and os.path.exists(image2) \
       and os.path.exists(geomodel1) \
       and os.path.exists(geomodel2):
        my_bar = st.progress(0, text="Download SRTM")
        env1 = get_envelope(image1, geomodel1)
        upleft = env1[0, :2][::-1]
        srtm = get_srtm_tif_name(*upleft)
        r = requests.get(srtm)
        srtm_bn = os.path.basename(srtm)
        srtm_tif = os.path.splitext(srtm_bn)[0]+".tif"
        open(srtm_bn, "wb").write(r.content)
        with zipfile.ZipFile(srtm_bn, "r") as zf:
            zf.extract(srtm_tif)
            os.rename(srtm_tif, srtm_path)
        os.remove(srtm_bn)

        cars_sensor_to_dsm.run(image1, image2,
                               geomodel1, geomodel2,
                               srtm_path,
                               output_dir,
                               outdata,
                               my_bar)
        my_bar.empty()
    else:
        st.warning("Clic on \"Download demo\" or upload your own data before", icon="⚠️")

# map
def create_map_drawing_envelopes(show):
    def draw_envelope(image, geomodel, color, m=None):
        if os.path.exists(image) and os.path.exists(geomodel):
            envelope = get_envelope(image, geomodel)

            if m is None:
                upleft = envelope[0, :2][::-1]
                m = folium.Map(location=upleft, zoom_start=16)

            fg = folium.FeatureGroup(name=image, show=show)
            m.add_child(fg)
            folium.Polygon(envelope[:, :2][:, [1, 0]],
                           color=color,
                           fill_color=color,
                           opacity=0.1,
                           tooltip=image).add_to(fg)
        return m

    m = draw_envelope(image1, geomodel1, "blue")
    m = draw_envelope(image2, geomodel2, "red", m)

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

def show_epipolar_images(step):
    option = st.selectbox(
        'Choose image', outdata[step].keys())

    if os.path.exists(outdata[step][option]):
        with rio.open(outdata[step][option]) as src:
            array = enhance(src.read(1), src.nodata)

        im = Image.fromarray(array)
        st.image(im.convert("RGB"))

    else:
        st.warning("Clic on \"Run CARS\" before", icon="⚠️")

def show_rasterization():
    dsm = outdata["rasterization"]
    m = create_map_drawing_envelopes(show=False)

    if m is not None and os.path.exists(dsm):
        with rio.open(dsm) as src:
            array = np.moveaxis(src.read(), 0, -1)
            nodata = 255*(array!=src.nodata)
            array = enhance(array, src.nodata)
            array = np.dstack((array, array, array, nodata))
            bounds = src.bounds
            bbox = [(bounds.bottom, bounds.left), (bounds.top, bounds.right)]
            folium.raster_layers.ImageOverlay(
                name=dsm,
                image=array,
                bounds=bbox,
                opacity=1,
                interactive=True,
                cross_origin=False,
                zindex=1,
            ).add_to(m)
        folium.LayerControl().add_to(m)
        st_data = st_folium(m, height=500, width=500)

    # see https://discuss.streamlit.io/t/streamlit-cloud-port-proxying-on-streamlit-io/24748/4
    # if os.path.exists(dsm):
    #     client = TileClient(dsm)
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

if os.path.exists(image1) \
   and os.path.exists(image2) \
   and os.path.exists(geomodel1) \
   and os.path.exists(geomodel2):
    col1, col2 = st.columns([1, 3])

    with col1:
        clicked = clickable_images(
            paths=url_steps,
            titles=steps,
            img_style={"width": "220%"}
        )

    with col2:
        if clicked in [-1, 0]:
            show_images()
        elif clicked == 4:
            show_rasterization()
        else:
            show_epipolar_images(steps[clicked])

    if st.button("Clean data and output directory"):
        for filename in [image1, geomodel1,
                         image2, geomodel2]:
            if os.path.exists(filename):
                os.remove(filename)
        shutil.rmtree(output_dir, ignore_errors=True)
