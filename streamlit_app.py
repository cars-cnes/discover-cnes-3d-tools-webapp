import streamlit as st
from io import StringIO
import numpy as np
import rasterio as rio
from shareloc.geomodels.rpc import RPC
from shareloc.image import Image
from shareloc.geofunctions import localization
import folium
from streamlit_folium import st_folium

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
left, right = st.columns((1, 1))
with left:
    image1 = st.file_uploader("Image 1")
    geomodel1 = st.file_uploader("Geomodel 1")
with right:
    image2 = st.file_uploader("Image 2")
    geomodel2 = st.file_uploader("Geomodel 2")

# map
def draw_envelope(image, geomodel, color, m=None):
    if image is not None and geomodel is not None:
        # read / write geomodel
        with open(image.name, "wb") as f:
            f.write(image.getbuffer())
        with open(geomodel.name, "wb") as f:
            f.write(geomodel.getbuffer())

        # read image
        shareloc_img = Image(image.name)
        shareloc_mdl = RPC.from_any(geomodel.name)

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

        fg = folium.FeatureGroup(name=image.name, show=True)
        m.add_child(fg)
        folium.Polygon(envelope[:, :2][:, [1, 0]],
                       color=color,
                       fill_color=color,
                       tooltip=image.name).add_to(fg)

    return m

m = draw_envelope(image1, geomodel1, "blue")
m = draw_envelope(image2, geomodel2, "red", m)
if m is not None:
    folium.LayerControl().add_to(m)
    st_data = st_folium(m, width=725)
