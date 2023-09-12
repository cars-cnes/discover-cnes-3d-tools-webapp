import os
import streamlit as st
from io import StringIO
import numpy as np
import rasterio as rio
from shareloc.geomodels.rpc import RPC
from shareloc.image import Image
from shareloc.geofunctions import localization
import folium
from streamlit_folium import st_folium
import tarfile
import requests

MINI_LOGO ="https://raw.githubusercontent.com/CNES/cars/master/docs/source/images/picto_transparent_mini.png"
st.set_page_config(page_title="cars-webapp",
                   page_icon=MINI_LOGO)

# title
__, center, __ = st.columns((1, 4, 1))
with center:
    st.markdown("![Logo]("+MINI_LOGO+")")

st.markdown(("# CARS, a satellite multi view stereo framework"))

# input images + geomodel
st.markdown(("## inputs"))

st.markdown(("### sensors"))

image1 = "data/image1.tif"
image2 = "data/image2.tif"
geomodel1 = "data/image1.geom"
geomodel2 = "data/image2.geom"

def upload_and_save(name, filename):
    uploaded = st.file_uploader(name)
    if uploaded is not None:
        with open(uploaded.name, "wb") as f:
            f.write(uploaded.getbuffer())
            os.rename(uploaded.name, filename)

left, right = st.columns((1, 1))
with left:
    upload_and_save("Image 1", image1)
    upload_and_save("Geomodel 1", geomodel1)
with right:
    upload_and_save("Image 2", image2)
    upload_and_save("Geomodel 2", geomodel2)

# download demo
if st.button("Download demo"):
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

if st.button("Clean data directory"):
    for filename in [image1, geomodel1,
                     image2, geomodel2]:
        if os.path.exists(filename):
            os.remove(filename)

# map
def draw_envelope(image, geomodel, color, m=None):
    if os.path.exists(image) and os.path.exists(geomodel):
        # read image
        shareloc_img = Image(image)
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

        if m is None:
            m = folium.Map(location=envelope[0, :2][::-1], zoom_start=16)

        fg = folium.FeatureGroup(name=image, show=True)
        m.add_child(fg)
        folium.Polygon(envelope[:, :2][:, [1, 0]],
                       color=color,
                       fill_color=color,
                       tooltip=image).add_to(fg)

    return m

m = draw_envelope(image1, geomodel1, "blue")
m = draw_envelope(image2, geomodel2, "red", m)
if m is not None:
    folium.LayerControl().add_to(m)
    st_data = st_folium(m, width=725)
