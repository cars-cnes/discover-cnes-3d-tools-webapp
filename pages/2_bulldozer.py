import streamlit as st

FAVICON ="https://cnes.fr/sites/all/themes/web3/favicon.ico"
st.set_page_config(page_title="CNES 3D | Bulldozer,a DTM from DSM extraction tool", page_icon=FAVICON)

PICTO_BULLDO = "https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/bulldozer_logo.png"
left, right = st.columns((1, 4))
with left:
    st.markdown("")
    st.markdown("")
    bulldo = '<div style="text-align: center; display: block;"><link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Audiowide"><p style="font-family:Audiowide"><img src="'+PICTO_BULLDO+'"; height=60; style="margin-bottom: 5px"; alt="bulldozer;"> <br> bulldozer</p></div>'
    st.markdown(bulldo, unsafe_allow_html=True)

with right:
    st.markdown("<h1 style='text-align: center;'>Bulldozer, a DTM from DSM extraction tool</h1>", unsafe_allow_html=True)

st.info("Work in progress...")
