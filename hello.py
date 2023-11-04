import streamlit as st

FAVICON ="https://cnes.fr/sites/all/themes/web3/favicon.ico"
st.set_page_config(page_title="cnes-3d-tools", page_icon=FAVICON)

LOGO_CNES = "https://cnes.fr/sites/default/files/styles/large/public/drupal/201707/image/is_logo_2017_logo_carre_bleu.jpg?itok=UtsAO5qP"
LOGO_REP = "https://upload.wikimedia.org/wikipedia/fr/thumb/2/22/Republique-francaise-logo.svg/532px-Republique-francaise-logo.svg.png"

# title
left, center, right = st.columns((1, 1, 3))
with left:
    st.image(LOGO_REP, width=120)
with center:
    st.image(LOGO_CNES, width=100)
with right:
    st.title(("Discover CNES 3D open-source tools"))

st.header("""In a nutshell""")

st.markdown("""
In the frame of a future dedicated CO3D constellation, CNES (French space agency) has developed open source 3D tools around CARS main pipeline tool. These tools are made available for massive Digital Surface Model production with a robust and performant design, but also for research aims.
""")
st.markdown("""
**👈 Select a page from the sidebar** to discover this open source 3D toolbox!
""")
st.header("""Want to learn more?""")

PICTO_CO3D = "https://co3d.cnes.fr/sites/default/files/drupal/202309/image/logo.jpg"
left, right = st.columns((1, 4))
with left:
    st.image(PICTO_CO3D, width=140)
with right:
    st.subheader("CO3D, A constellation to map the world in 3D")
    st.markdown("""
    - CESBIO Website: https://labo.obs-mip.fr/multitemp/co3d-the-very-high-resolution-mission-dedicated-to-3d/
    - CNES Website: https://co3d.cnes.fr/en/co3d-0
    - eoPortal article: https://www.eoportal.org/satellite-missions/co3d-constellation#space-and-hardware-components
    - ISPRS Article: https://isprs-archives.copernicus.org/articles/XLIII-B1-2020/299/2020/isprs-archives-XLIII-B1-2020-299-2020.pdf
    """)

PICTO_CARS = "https://raw.githubusercontent.com/CNES/cars/master/docs/source/images/picto_transparent_mini.png"
left, right = st.columns((1, 4))
with left:
    st.image(PICTO_CARS, width=140)
with right:
    st.subheader("CARS, a satellite multi view stereo framework")
    st.markdown("""
    - GitHub page: https://github.com/CNES/cars
    - PyPI package: https://pypi.org/project/cars/
    - ReadTheDocs documentation: https://cars.readthedocs.io/
""")
