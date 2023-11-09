import streamlit as st

FAVICON ="https://cnes.fr/sites/all/themes/web3/favicon.ico"
st.set_page_config(page_title="CNES 3D | DEMCOMPARE, a DSM / DTM comparison tool", page_icon=FAVICON)

PICTO_DEMCOMPARE = "https://raw.githubusercontent.com/CNES/demcompare/master/docs/source/images/demcompare_picto.png"
left, right = st.columns((1, 4))
with left:
    st.markdown("")
    demcompare = '<div style="text-align: center; display: block;"><link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Orbitron"><p style="font-family:Orbitron; color:rgb(0,47,108)"><img src="'+PICTO_DEMCOMPARE+'"; height=80"; alt="demcompare;"> <br> demcompare</p></div>'
    st.markdown(demcompare, unsafe_allow_html=True)

with right:
    st.markdown("<h1 style='text-align: center;'>Bulldozer, a DTM from DSM extraction tool</h1>", unsafe_allow_html=True)

st.info("Work in progress...")
