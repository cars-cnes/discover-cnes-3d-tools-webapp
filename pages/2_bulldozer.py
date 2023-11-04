import streamlit as st

PICTO_BULLDO = "https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/bulldozer_logo.png"
left, right = st.columns((1, 4))
with left:
    bulldo = '<div style="text-align: center;"><link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Audiowide"><p style="font-family:Audiowide"><img src="'+PICTO_BULLDO+'"; width=120; alt="bulldozer"> bulldozer</p></div>'
    st.markdown("")
    st.markdown("")
    st.markdown(bulldo, unsafe_allow_html=True)

with right:
    st.title("Bulldozer, a DTM from DSM extraction tool")

st.info("Work in progress...")
