import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
import data_visualization
import model.model_fitter

import upload



EXAMPLE_NO = 1
def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="PdM",  # required
                options=["Upload Data", "Data Visualization","Model Training", "Model Evaluation","Model Inference"],  # required
                icons=["cloud-upload", "bar-chart-fill", "robot","clipboard2-check"],  # optional
                menu_icon="tools",  # optional
                default_index=0,  # optional
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Upload Data", "Data Visualization","Model Training", "Model Evaluation","Model Inference"],  # required
            icons=["cloud-upload", "bar-chart-fill", "robot","clipboard2-check"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 2. horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Upload Data", "Data Visualization","Model Training", "Model Evaluation","Model Inference"],  # required
            icons=["cloud-upload", "bar-chart-fill", "robot","clipboard2-check"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
        if example == 4:
            # 2. horizontal menu with custom style
            selected = option_menu(
                menu_title=None,  # required
                options=["Upload Data", "Data Visualization", "Model Training", "Model Evaluation", "Model Inference"],
                # required
                icons=["cloud-upload", "bar-chart-fill", "robot", "clipboard2-check"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
                orientation="horizontal",
                styles={
                    "container": {"padding": "0!important", "background-color": "#fafafa"},
                    "icon": {"color": "orange", "font-size": "25px"},
                    "nav-link": {
                        "font-size": "25px",
                        "text-align": "left",
                        "margin": "0px",
                        "--hover-color": "#eee",
                    },
                    "nav-link-selected": {"background-color": "green"},
                },
            )
        return selected


selected = streamlit_menu(example=EXAMPLE_NO)


if selected == "Upload Data":
    upload.data_uploader()


def histogram_intersection(a, b):
    v = np.minimum(a, b).sum().round(decimals=1)
    return v

if selected == 'Data Visualization':
    data_visualization.visualization()


if selected == "Model Training":
    model.model_fitter.fit_models()


if selected == "Model Evaluation":
    st.title('Model Evaluation')
    if 'model' in st.session_state:
        st.dataframe(st.session_state.model)

if selected == "Model Inference":
    st.title('Model Inference')