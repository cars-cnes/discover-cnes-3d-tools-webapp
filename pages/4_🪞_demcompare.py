import pandas as pd
import streamlit as st
import tempfile
import os
import plotly.express as px
from demcompare.dem_tools import load_dem
from demcompare.coregistration import Coregistration
from demcompare.dem_tools import compute_alti_diff_for_stats
from demcompare.stats_processing import StatsProcessing
import shutil

FAVICON ="https://cnes.fr/sites/all/themes/web3/favicon.ico"
st.set_page_config(page_title="CNES 3D | DEMCOMPARE, a DSM / DTM comparison tool", page_icon=FAVICON)

PICTO_DEMCOMPARE = "https://raw.githubusercontent.com/CNES/demcompare/master/docs/source/images/demcompare_picto.png"
left, right = st.columns((1, 4))
with left:
    st.markdown("")
    demcompare = '<div style="text-align: center; display: block;"><link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Orbitron"><p style="font-family:Orbitron; color:rgb(0,47,108)"><img src="'+PICTO_DEMCOMPARE+'"; height=80"; alt="demcompare;"> <br> demcompare</p></div>'
    st.markdown(demcompare, unsafe_allow_html=True)

with right:
    st.markdown("<h1 style='text-align: center;'>DEMCOMPARE, a DSM / DTM comparison tool</h1>", unsafe_allow_html=True)

st.header("1. Upload the DEMs")

col1, col2, = st.columns(2)
with col1:
    st.subheader("DEM 1")
    dem1 = st.file_uploader("Upload your DEM 1",
                            accept_multiple_files=False,
                            label_visibility="collapsed",
                            type=["tif"])
with col2:
    st.subheader("DEM 2")
    dem2 = st.file_uploader("Upload your DEM 2",
                            accept_multiple_files=False,
                            label_visibility="collapsed",
                            type=["tif"])

st.header("2. Launch DEMCOMPARE")
if st.button("Run DEMCOMPARE"):
    if dem1 is not None and dem2 is not None:
        temp_dir = tempfile.mkdtemp()
        dem1_path = os.path.join(temp_dir, "DEM1.tif")
        dem2_path = os.path.join(temp_dir, "DEM2.tif")
        with open(dem1_path, "wb") as f:
            f.write(dem1.getbuffer())
        with open(dem2_path, "wb") as f:
            f.write(dem2.getbuffer())

        input1 = load_dem(path=dem1_path)
        input2 = load_dem(path=dem2_path)

        coreg_cfg = {
            "method_name": "nuth_kaab_internal",
            "number_of_iterations": 6,
            "estimated_initial_shift_x": 0,
            "estimated_initial_shift_y": 0
        }

        coregistration = Coregistration(coreg_cfg)
        transformation = coregistration.compute_coregistration(input2, input1)

        input2_coreg = transformation.apply_transform(input2)

        reproj_ref = coregistration.reproj_ref
        reproj_sec = coregistration.reproj_sec
        altitude_diff = compute_alti_diff_for_stats(reproj_ref, reproj_sec)
        statistics_cfg = {}
        stats_processing = StatsProcessing(statistics_cfg, altitude_diff, input_diff=True)
        stats_dataset = stats_processing.compute_stats()
        stats_metrics = stats_dataset.get_classification_layer_metrics(classification_layer="global")

        list_metrics = [["Metric's name", "Measured metrics"]]
        values = []
        metric = []
        for metric in stats_metrics:
            values.append(stats_dataset.get_classification_layer_metric(classification_layer="global", metric=metric)[0])

        df = pd.DataFrame(values, index=stats_metrics, columns =['Values'])
        array = altitude_diff["image"].data

        st.session_state["demcompare"] = {"image": array,
                                          "transformation": transformation,
                                          "df": df}
        shutil.rmtree(temp_dir)

st.header("3. See statistics")

if "demcompare" in st.session_state:
    fig = px.imshow(st.session_state["demcompare"]["image"])
    fig.update_coloraxes(colorscale="RdBu")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Shift: "+str(st.session_state["demcompare"]["transformation"]))
    st.table(st.session_state["demcompare"]["df"])
