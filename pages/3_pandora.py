import streamlit as st

FAVICON ="https://cnes.fr/sites/all/themes/web3/favicon.ico"
st.set_page_config(page_title="CNES 3D | PANDORA, a stereo matching framework", page_icon=FAVICON)

PICTO_PANDORA = "https://raw.githubusercontent.com/CNES/pandora/master/docs/source/Images/logo/logo_typo_large.png"
left, right = st.columns((1, 4))
with left:
    pandora = '<div style="text-align: center;"><br><img src="'+PICTO_PANDORA+'"; height=80; alt="pandora"></div>'
    st.markdown(pandora, unsafe_allow_html=True)
with right:
    st.markdown("<h1 style='text-align: center;'>PANDORA, a stereo matching framework</h1>", unsafe_allow_html=True)

st.info("Work in progress...")
