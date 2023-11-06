import streamlit as st

FAVICON ="https://cnes.fr/sites/all/themes/web3/favicon.ico"
st.set_page_config(page_title="CNES 3D | Discover CNES 3D open-source tools", page_icon=FAVICON)

LOGO_CNES = "https://cnes.fr/sites/default/files/styles/large/public/drupal/201707/image/is_logo_2017_logo_carre_bleu.jpg?itok=UtsAO5qP"
LOGO_REP = "https://upload.wikimedia.org/wikipedia/fr/thumb/2/22/Republique-francaise-logo.svg/532px-Republique-francaise-logo.svg.png"

# title
left, right = st.columns((1, 1.5))
with left:
    cnes = '<div style="text-align: center;"><img src="'+LOGO_REP+'"; width=120; alt="rep"><img src="'+LOGO_CNES+'"; width=100; alt="cnes"></div>'
    st.markdown(cnes, unsafe_allow_html=True)

with right:
    st.markdown("<h1 style='text-align: center;'>Discover CNES 3D open-source tools</h1>", unsafe_allow_html=True)

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
    co3d = '<div style="text-align: center;"><img src="'+PICTO_CO3D+'"; height=140; alt="co3d"></div>'
    st.markdown(co3d, unsafe_allow_html=True)
with right:
    st.markdown("")
    st.markdown("<h3 style='text-align: center;'>CO3D, A constellation to map <br> the world in 3D</h3>", unsafe_allow_html=True)
    st.markdown("")
st.markdown("""
- CNES webpage: https://co3d.cnes.fr/en/co3d-0
- CESBIO blog: [https://labo.obs-mip.fr/.../co3d-the-very-high-resolution-mission-dedicated-to-3d/](https://labo.obs-mip.fr/multitemp/co3d-the-very-high-resolution-mission-dedicated-to-3d/)
- CO3D mission on eoPortal: https://www.eoportal.org/satellite-missions/co3d-constellation
- ISPRS article: https://isprs-archives.copernicus.org/articles/XLIII-B1-2020/299/2020
""")

PICTO_CARS = "https://raw.githubusercontent.com/CNES/cars/master/docs/source/images/picto_transparent_mini.png"
left, right = st.columns((1, 4))
with left:
    cars = '<div style="text-align: center;"><img src="'+PICTO_CARS+'"; height=140; alt="cars"></div>'
    st.markdown(cars, unsafe_allow_html=True)
with right:
    st.markdown("")
    st.markdown("<h3 style='text-align: center;'>CARS, a satellite multi view <br> stereo framework</h3>", unsafe_allow_html=True)
    st.markdown("")
st.markdown("""
    - GitHub page: https://github.com/CNES/cars
    - PyPI package: https://pypi.org/project/cars/
    - ReadTheDocs documentation: https://cars.readthedocs.io/
""")


PICTO_BULLDO = "https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/bulldozer_logo.png"
left, right = st.columns((1, 4))
with left:
    st.markdown("")
    st.markdown("")
    bulldo = '<div style="text-align: center; display: block;"><link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Audiowide"><p style="font-family:Audiowide"><img src="'+PICTO_BULLDO+'"; height=60; style="margin-bottom: 5px"; alt="bulldozer;"> <br> bulldozer</p></div>'
    st.markdown(bulldo, unsafe_allow_html=True)

with right:
    st.markdown("")
    st.markdown("<h3 style='text-align: center;'>Bulldozer, a DTM from DSM <br> extraction tool</h3>", unsafe_allow_html=True)
    st.markdown("")
st.markdown("""
- GitHub page: https://github.com/CNES/bulldozer
- PyPI package: https://pypi.org/project/bulldozer-dtm
""")
