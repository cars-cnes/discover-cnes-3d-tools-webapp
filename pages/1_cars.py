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

import processing.cars_steps as cars_steps
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
# from localtileserver import get_folium_tile_layer, TileClient

import laspy.file
import laspy.header

FAVICON ="https://cnes.fr/sites/all/themes/web3/favicon.ico"
st.set_page_config(page_title="CNES 3D | CARS, a satellite multi view stereo framework", page_icon=FAVICON)

PICTO_CARS = "https://raw.githubusercontent.com/CNES/cars/master/docs/source/images/picto_transparent_mini.png"
left, right = st.columns((1, 4))
with left:
    cars = '<div style="text-align: center;"><img src="'+PICTO_CARS+'"; height=140; alt="cars"></div>'
    st.markdown(cars, unsafe_allow_html=True)

with right:
    st.markdown("<h1 style='text-align: center;'>CARS, a satellite multi view stereo framework</h1>", unsafe_allow_html=True)

st.header("1. Select the dataset")
data_select = st.radio("Select the dataset",
                       ["Pyramids",
                        "Turkey pre-event",
                        "Turkey post-event",
                        "Upload your own data"],
                       label_visibility="collapsed")

if data_select == "Upload your own data":
    uploaded_files = st.file_uploader("or upload your own data",
                                      accept_multiple_files=True,
                                      label_visibility="collapsed")
    uploaded_dict = {None: None}
    choices = []
    for uploaded in uploaded_files:
        choices.append(uploaded.name)
        uploaded_dict[uploaded.name] = uploaded

    geomodel1 = uploaded_dict[st.selectbox("Geomodel 1", choices)]
    image1 = uploaded_dict[st.selectbox("Image 1", choices)]
    geomodel2 = uploaded_dict[st.selectbox("Geomodel 2", choices)]
    image2 = uploaded_dict[st.selectbox("Image 2", choices)]

    dsm_out = "dsm_with_your_own_data.tif"

elif data_select == "Pyramids":
    geomodel1 = os.path.abspath("demo/img1.geom")
    image1 = os.path.abspath("demo/img1.tif")
    geomodel2 = os.path.abspath("demo/img2.geom")
    image2 = os.path.abspath("demo/img2.tif")
    dsm_out = "dsm_pyramids.tif"

elif data_select == "Turkey pre-event":
    geomodel1 = os.path.abspath("demo/pre_event_img1.geom")
    image1 = os.path.abspath("demo/pre_event_img1.tif")
    geomodel2 = os.path.abspath("demo/pre_event_img2.geom")
    image2 = os.path.abspath("demo/pre_event_img2.tif")
    dsm_out = "dsm_turkey_pre_event.tif"

elif data_select == "Turkey post-event":
    geomodel1 = os.path.abspath("demo/post_event_img1.geom")
    image1 = os.path.abspath("demo/post_event_img1.tif")
    geomodel2 = os.path.abspath("demo/post_event_img2.geom")
    image2 = os.path.abspath("demo/post_event_img2.tif")
    dsm_out = "dsm_turkey_post_event.tif"

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
        st.warning(image.name + " is not a correct image", icon="⚠️")
        if isinstance(image, str) is False:
            os.remove(temp_image)
        return None

    try:
        shareloc_mdl = RPC.from_any(temp_geomodel)
    except ValueError:
        st.warning(geomodel.name + " is not a correct geomodel", icon="⚠️")
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
        st_folium(m, height=500, width=500)

if None not in [image1, image2, geomodel1, geomodel2]:
    show_images()

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

st.header("2. Launch CARS")
# run cars
if st.button("Run CARS"):
    if None not in [image1, image2, geomodel1, geomodel2]:
        envelope_and_center1 = get_envelope_and_center(image1, geomodel1)
        envelope_and_center2 = get_envelope_and_center(image2, geomodel2)
        if None not in [envelope_and_center1, envelope_and_center2]:
            try:
                st.session_state["sparse"], st.session_state["dense"] = cars_steps.run(image1, image2, geomodel1, geomodel2)
            except Exception as e:
                st.error("CARS encountered a problem during execution")
                st.error(e)
                time.sleep(10)
    else:
        st.warning("Select the dataset first", icon="⚠️")

if "dense" in st.session_state:
    __, temp1 = tempfile.mkstemp(suffix=".tif")
    dsm_data = st.session_state["dense"]["rasterization"]
    save_data(dsm_data, temp1, tag="hgt", nodata=-32768)
    with open(temp1, "rb") as dsm_file:
        st.download_button("Download DSM",
                           data=dsm_file,
                           file_name=dsm_out)
    os.remove(temp1)

    __, temp2 = tempfile.mkstemp(suffix=".laz")
    header = laspy.LasHeader(point_format=2)
    scale_factor = 0.01
    header.scales = [scale_factor, scale_factor, scale_factor]
    laz = laspy.LasData(header)

    pc = st.session_state["dense"]["triangulation.pc"]

    # positions
    laz.x = pc.x
    laz.y = pc.y
    laz.z = pc.z

    # colors
    color = pc.color_Gray.values
    color_mean = np.mean(color)
    color_std = np.std(color)
    color_max = color_mean + 3*color_std
    color_min = color_mean - 3*color_std
    color = 65535*(color-color_min) / (color_max-color_min)
    color[color > 65535] = 65535
    color[color < 0] = 0
    color = color.astype(int)

    laz.red = color
    laz.green = color
    laz.blue = color

    laz.write(temp2)

    pc_out = os.path.splitext(dsm_out)[0]+".laz"
    pc_out = "pc"+ pc_out.split("dsm")[-1]
    with open(temp2, "rb") as pc_file:
        st.download_button("Download Point Cloud",
                           data=pc_file,
                           file_name=pc_out)
    os.remove(temp2)


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


def show_epipolar_images(step, pipeline):
    outdata = st.session_state[pipeline]

    option = st.selectbox(
        'Choose image', outdata[step].keys(), key=pipeline+" selectbox")

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


def show_matches(group_size=50):
    outdata = st.session_state["sparse"]
    arrays = {}
    for key in outdata["resampling"].keys():
        __, temp = tempfile.mkstemp(suffix=".tif")
        carsdata = outdata["resampling"][key]
        save_data(carsdata["data"], temp,
                  tag=carsdata["tag"],
                  nodata=carsdata["nodata"])

        with rio.open(temp) as src:
            arrays[key] = np.moveaxis(src.read(), 0, -1)
            arrays[key] = enhance(arrays[key], src.nodata)
            arrays[key] = np.dstack((arrays[key],
                                     arrays[key],
                                     arrays[key]))

        os.remove(temp)

    matches_array = outdata["matching"]

    # Create figure
    fig = go.Figure()
    left_right = np.hstack((arrays["left"], arrays["right"]))
    fig = px.imshow(left_right, color_continuous_scale='gray')

    # matches_array = np.load("matches.npy")
    nb_matches = len(matches_array[:, 0])
    text = list(map(lambda x: "idx: "+str(x), range(nb_matches)))

    fig.add_trace(
        go.Scatter(x=matches_array[:, 0],
                   y=matches_array[:, 1],
                   text=text,
                   mode='markers',
                   name='left keypoints')
        )

    fig.add_trace(
        go.Scatter(x=matches_array[:, 2] + arrays["left"].shape[1],
                   y=matches_array[:, 3],
                   text=text,
                   mode='markers',
                   name='right keypoints')
        )

    nb_groups = int(np.ceil(nb_matches / group_size))

    for idx in range(nb_matches):
        if idx < nb_groups:
            showlegend = True
        else:
            showlegend = False

        fig.add_trace(
            go.Scatter(
                visible=False,
                x=[matches_array[idx, 0],
                   matches_array[idx, 2] + arrays["left"].shape[1]],
                y=[matches_array[idx, 1],
                   matches_array[idx, 3]],
                legendgroup='it\'s a match',
                name='it\'s a match',
                showlegend=showlegend,
                line_color='yellow')
        )

    for i in range(nb_matches)[0::nb_groups]:
        fig.data[3+i].visible = True

    # Create and add slider
    steps = []
    for i in range(nb_groups):
        step = dict(
            label=str(i),
            method="update",
            args=[{"visible": 3*[True] + (len(fig.data)-1) * [False]},
                  {"title": "Match group: " + str(i)}],  # layout attribute
        )
        for j in range(nb_matches)[i::nb_groups]:
            step["args"][0]["visible"][3+j] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Group: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1),
        sliders=sliders
    )

    st.plotly_chart(fig, use_container_width=True)

def get_wgs84_dsm_file_from_carsdata(carsdata, pipeline):
    __, temp1 = tempfile.mkstemp(suffix=".tif")
    __, temp2 = tempfile.mkstemp(suffix=".tif")

    if pipeline == "dense":
        save_data(carsdata,
                  temp1,
                  tag="hgt",
                  nodata=-32768)
    else:
        with open(temp1, "wb") as demfile:
            demfile.write(carsdata)

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

def show_points_cloud():
    pc = st.session_state["sparse"]["triangulation"]

    pc_valid = pc[(pc["z"] < pc["z"].quantile(.99)) & \
                  (pc["z"] > pc["z"].quantile(.01))]

    marker = dict(size=3,
                  color=pc_valid["z"],
                  colorscale='Viridis',
                  opacity=0.8)

    scene = dict(aspectmode='data')

    fig = go.Figure(data=[go.Scatter3d(x=pc_valid["x"],
                                       y=pc_valid["y"],
                                       z=pc_valid["z"],
                                       mode='markers',
                                       marker=marker)],
                    layout=go.Layout(scene=scene))

    st.plotly_chart(fig, use_container_width=True)


def show_rasterization(pipeline):
    dsm = st.session_state[pipeline]["rasterization"]
    m = create_map_drawing_envelopes(show=False)

    if m is not None:
        temp = get_wgs84_dsm_file_from_carsdata(dsm, pipeline)

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
        st.warning("Click on \"Run CARS\" before", icon="⚠️")

steps = ["resampling", "matching", "triangulation", "rasterization"]

st.header("3. See intermediate results")

st.subheader("Select the sparse pipeline step 👇")

sparse_step = st.radio(
    "Select the sparse pipeline step 👇",
    steps, horizontal=True,
    label_visibility="collapsed"
)

if "sparse" in st.session_state:
    if sparse_step == "resampling":
        show_epipolar_images("resampling", "sparse")
    elif sparse_step == "matching":
        show_matches()
    elif sparse_step == "triangulation":
        show_points_cloud()
    else:
        show_rasterization("sparse")
else:
    st.warning("Click on \"Run CARS\" before", icon="⚠️")

st.subheader("Select the dense pipeline step 👇")

dense_step = st.radio(
    "Select the dense pipeline step 👇",
    steps, horizontal=True,
    label_visibility="collapsed"
)

if "dense" in st.session_state:
    if dense_step in ["resampling", "triangulation", "matching"]:
        show_epipolar_images(dense_step, "dense")
    else:
        show_rasterization("dense")
else:
    st.warning("Click on \"Run CARS\" before", icon="⚠️")
