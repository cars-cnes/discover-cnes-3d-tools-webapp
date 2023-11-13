import os
from glob import glob
import shutil
import tempfile
import numpy as np
import streamlit as st
from bulldozer.pipeline.bulldozer_pipeline import dsm_to_dtm
import rasterio as rio
import plotly.graph_objects as go
import plotly.express as px

FAVICON ="https://cnes.fr/sites/all/themes/web3/favicon.ico"
st.set_page_config(page_title="CNES 3D | Bulldozer, a DTM from DSM extraction tool", page_icon=FAVICON)

PICTO_BULLDO = "https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/bulldozer_logo.png"
left, right = st.columns((1, 4))
with left:
    st.markdown("")
    st.markdown("")
    bulldo = '<div style="text-align: center; display: block;"><link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Audiowide"><p style="font-family:Audiowide"><img src="'+PICTO_BULLDO+'"; height=60; style="margin-bottom: 5px"; alt="bulldozer;"> <br> bulldozer</p></div>'
    st.markdown(bulldo, unsafe_allow_html=True)

with right:
    st.markdown("<h1 style='text-align: center;'>Bulldozer, a DTM from DSM extraction tool</h1>", unsafe_allow_html=True)


st.header("1. Upload the DSM")
dsm = st.file_uploader("Upload your DSM",
                       accept_multiple_files=False,
                       label_visibility="collapsed")

st.header("2. Launch Bulldozer")
if st.button("Run Bulldozer"):
    if dsm is not None:
        dsm_suffix = os.path.splitext(dsm.name)[-1]
        temp_dir = tempfile.mkdtemp()
        dsm_path = os.path.join(temp_dir, "DSM"+dsm_suffix)
        with open(dsm_path, "wb") as f:
            f.write(dsm.getbuffer())

        params = {"nb_max_workers": 1,
                  "max_object_width": 16,
                  "check_intersection": False,
                  "min_valid_height": 0.0,
                  "no_data": None,
                  "developper_mode": False,
                  "keep_inter_dtm" : False,
                  "slope_threshold" : 2.0,
                  "four_connexity" : True,
                  "uniform_filter_size" : 3,
                  "prevent_unhook_iter" : 10,
                  "num_outer_iter" : 50,
                  "num_inner_iter" : 10,
                  "mp_tile_size" : 1500}
        try:
            dsm_to_dtm(dsm_path=dsm_path,
                       output_dir=temp_dir,
                       **params)

        except FileNotFoundError:
            st.rerun()
            dsm_to_dtm(dsm_path=dsm_path,
                       output_dir=temp_dir,
                       **params)

        st.info("Bulldozer has successfully completed the pipeline")

        rasters_list = [dsm_path] + glob(os.path.join(temp_dir, "D*.tif"))
        rasters_list = list(set(rasters_list))

        st.session_state["bulldozer"] = dict()
        for raster in rasters_list:
            with open(raster, "rb") as reader:
                st.session_state["bulldozer"][os.path.basename(raster)] = reader.read()

        shutil.rmtree(temp_dir, ignore_errors=True)

st.header("3. See DSM / DHM / DTM etc.")
if "bulldozer" in st.session_state:
    temp_dir = tempfile.mkdtemp()
    left, right = st.columns((3, 1))
    rasters_list = st.session_state["bulldozer"].keys()

    with left:
        st.image("https://raw.githubusercontent.com/cars-cnes/discover-cnes-3d-tools/gh-pages/images/dsm_dtm_dhm_illustration.png")
    with right:
        for raster in rasters_list:
            st.download_button("Download "+raster,
                               data=st.session_state["bulldozer"][raster],
                               file_name=raster)

    fig_list = []
    for raster in rasters_list:
        image = os.path.join(temp_dir, raster)
        with open(image, "wb") as writer:
            writer.write(st.session_state["bulldozer"][raster])
        with rio.open(image) as src:
            array = src.read(1).astype(float)
            array[array==src.nodata] = np.nan
            array = np.flip(array, 0)
            fig_data = px.imshow(array).data[0]
            fig_list.append(fig_data)

    nb_rasters = len(rasters_list)
    fig = go.Figure(fig_list)
    for idx in range(nb_rasters):
        fig.data[idx].visible = False
    fig.data[0].visible = True
    fig.update_coloraxes(colorscale="Greys")
    fig.update_layout(width=500, height=500,
                      xaxis_visible=False,
                      yaxis_visible=False)
    fig.update_yaxes(scaleanchor="x",
                     scaleratio=1)

    buttons = list()
    for idx, name in enumerate(rasters_list):
        visible = [False] * nb_rasters
        visible[idx] = True
        button = dict(label=os.path.basename(name),
                      method="restyle",
                      args=[{"visible": visible},])
        buttons.append(button)

    fig.update_layout(
        updatemenus=[
            dict(active=0,
                 buttons=buttons,
                 x=1.1,
                 xanchor="left",
                 y=1.1,
                 yanchor="top")])

    st.plotly_chart(fig, use_container_width=True)
    shutil.rmtree(temp_dir, ignore_errors=True)
