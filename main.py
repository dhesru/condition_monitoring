import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from streamlit_option_menu import option_menu
import os
import glob
import plotly.express as px
import pysurvival
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from pysurvival.pysurvival.models.multi_task import LinearMultiTaskModel
from pysurvival.pysurvival.models.survival_forest import ConditionalSurvivalForestModel, ExtraSurvivalTreesModel,RandomSurvivalForestModel
from pysurvival.pysurvival.utils.metrics import concordance_index,integrated_brier_score as ibs
from pysurvival.pysurvival.utils.display import integrated_brier_score, correlation_matrix, compare_to_actual

EXAMPLE_NO = 1
def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="PdM",  # required
                options=["Upload Data", "Data Visualization","Model Training", "Model Evaluation"],  # required
                icons=["cloud-upload", "bar-chart-fill", "robot","graph-up"],  # optional
                menu_icon="tools",  # optional
                default_index=0,  # optional
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Upload Data", "Data Visualization","Model Training", "Model Evaluation"],  # required
            icons=["cloud-upload", "bar-chart-fill", "robot","graph-up"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 2. horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Upload Data", "Data Visualization","Model Training", "Model Evaluation"],  # required
            icons=["cloud-upload", "bar-chart-fill", "robot","graph-up"],  # optional
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

def save_uploaded_file(uploadedfile):
  with open(os.path.join("./",uploadedfile.name),"wb") as f:
     f.write(uploadedfile.getbuffer())
  return st.success("Saved file locally")

if selected == "Upload Data":
    st.title('Welcome to Condition Monitoring')

    uploaded_file = st.file_uploader("Upload a CSV file for condition monitoring..")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()

        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.dataframe(dataframe)
        st.write("Filename: ", uploaded_file.name)
        save_uploaded_file(uploaded_file)
    else:
        files_path = os.path.join('.', '*')
        files = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True)
        csvs = [x for x in files if ".csv" in x]
        if len(csvs) > 0:
            csv_name = csvs[0]
            csv_name = csv_name.split('\\')[1]
            df = pd.read_csv(csv_name)
            st.dataframe(df)


def remove_ttf_event(df):
    col_names = list(df.columns)
    col_names.remove('ttf')
    col_names.remove('event')
    return col_names

def histogram_intersection(a, b):
    v = np.minimum(a, b).sum().round(decimals=1)
    return v

if selected == 'Data Visualization':
    st.title('Data Viz')
    files_path = os.path.join('.', '*')
    files = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True)
    csvs = [x for x in files if ".csv" in x]
    if len(csvs) > 0:
        df = pd.read_csv(csvs[0])
        features = remove_ttf_event(df)
        corr_plt = correlation_matrix(df[features], figure_size=(3, 3))
        # Create distplot with custom bin_size
        data_list = list()
        for feat in features:
            data_list.append(df[feat].to_numpy())

        fig = ff.create_distplot(data_list, features, bin_size=[.1, .1, .1])

        corr_val = df[features].corr()

        corr_plt = px.imshow(corr_val, text_auto=True)

        # Plot!
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(corr_plt, use_container_width=True)


def model_fitting(option,X_train,T_train,E_train):
    if option == 'Linear MTLR':
        mdl_type = LinearMultiTaskModel(bins=300)
        mdl = mdl_type.fit(X_train, T_train, E_train, num_epochs=1000,
                              init_method='orthogonal', optimizer='rmsprop',
                              lr=1e-3, l2_reg=3, l2_smooth=3, )
    if option == 'Conditional Survival Forest':
        mdl_type = ConditionalSurvivalForestModel(num_trees=10)
        mdl = mdl_type.fit(X_train, T_train, E_train, max_features = 'sqrt', max_depth = 5,
                        min_node_size = 10, alpha = 0.05, minprop= 0.1, num_threads = -1,
                        weights = None, sample_size_pct = 0.63,
                        importance_mode = 'normalized_permutation', seed = None,
                        save_memory=False ),
    if option == 'Extra Survival Trees':
        mdl_type = ExtraSurvivalTreesModel(num_trees=10)
        mdl = mdl_type.fit(X_train, T_train, E_train,max_features = 'sqrt', max_depth = 5,
                        min_node_size = 10, num_random_splits = 100, num_threads = -1,
                        weights = None, sample_size_pct = 0.63,
                        importance_mode = 'normalized_permutation',  seed = None,
                        save_memory=False )
    if option == 'Random Survival Forest':
        mdl_type = RandomSurvivalForestModel(num_trees=10)
        mdl = mdl_type.fit(X_train, T_train, E_train, max_features = 'sqrt', max_depth = 5,
                        min_node_size = 10, alpha = 0.05, minprop= 0.1, num_threads = -1,
                        weights = None, sample_size_pct = 0.63,
                        importance_mode = 'normalized_permutation', seed = None,
                        save_memory=False )
    return mdl,mdl_type

if selected == "Model Training":
    st.title('Model Training')
    files_path = os.path.join('.', '*')
    files = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True)
    csvs = [x for x in files if ".csv" in x]

    option = st.selectbox('Model Type',('Linear MTLR', 'Conditional Survival Forest', 'Extra Survival Trees','Random Survival Forest'))

    if len(csvs) > 0:
        df = pd.read_csv(csvs[0])
        train = st.slider('Train Test Split', 0, 100, 70)
        test = 100 - train
        st.write('Selected Training period: ', train)
        st.write('Selected Testing period: ', test)
        if st.button('Begin Model Training'):
            N = len(df)
            index_train, index_test = train_test_split(range(N), test_size=test)
            st.write('Model Training has begun using ', option)
            features = np.setdiff1d(df.columns, ['ttf', 'event']).tolist()

            time_column = 'ttf'
            event_column = 'event'

            data_train = df.loc[index_train].reset_index(drop=True)
            data_test = df.loc[index_test].reset_index(drop=True)

            # Creating the X, T and E inputs
            X_train, X_test = data_train[features], data_test[features]
            T_train, T_test = data_train[time_column], data_test[time_column]
            E_train, E_test = data_train[event_column], data_test[event_column]
            mdl,mdl_type = model_fitting(option,X_train,T_train,E_train)
            print('MDL',mdl_type)

            st.write('Completed.')

            c_index = concordance_index(mdl_type, X_test, T_test, E_test)

            fig = integrated_brier_score(mdl_type, X_test, T_test, E_test, t_max=100,
                                   figure_size=(20, 6.5))
            st.pyplot(fig)

            results = compare_to_actual(mdl_type, X_test, T_test, E_test,
                                        is_at_risk=False, figure_size=(16, 6),
                                        metrics=['rmse', 'mean', 'median'])
            st.pyplot(results)
            ibs_score = ibs(mdl_type, X_test, T_test, E_test, t_max=None)
            if len(st.session_state.model) == 0:
                st.session_state.model = [option,train,test,c_index,ibs_score]
            else:
                st.session_state.model.append([option,train,test,c_index,ibs_score])


if selected == "Model Evaluation":
    st.title('Model Evaluation')
    if len(st.session_state.model) > 0:
        st.write(st.session_state.model)
