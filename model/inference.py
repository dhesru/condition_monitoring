import streamlit as st
import pandas as pd
import numpy as np
import os
import os.path
import glob
import platform

import plotly.express as px
import plotly.graph_objs as go

if platform.system() == 'Windows':
    from pysurvival.pysurvival.utils import load_model
else:
    from pysurvival.utils import load_model


model_dict = {'Linear MTLR':'LMTLR','Conditional Survival Forest': 'CSF',"Extra Survival Trees":'EST','Random Survival Forest':'RSF'}

def get_key(val,model_dict):
    for key, value in model_dict.items():
        if val == value:
            return key
    return "key doesn't exist"

def get_model_name(mdl):
    model_name = mdl.split('\\')
    last_index = len(model_name) - 1
    v = model_name[last_index].split('.')
    return v[0]



def infer():
    st.title('Model Inference')
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'trained_models')
    if not os.path.exists(final_directory):
        st.title('No models to infer')
    else:
        files_path = os.path.join(final_directory, '*')
        files = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True)
        models = [x for x in files if ".zip" in x]
        trained_models = list()
        for mdl in models:
            mdl_sn = get_model_name(mdl)
            key = get_key(mdl_sn,model_dict)
            trained_models.append(key)
        option = st.selectbox('Select the Model to infer',trained_models)
        uploaded_file = st.file_uploader("Upload CSV file to infer")
        if uploaded_file is not None:
            dataframe = pd.read_csv(uploaded_file)
            #st_df = st.dataframe(dataframe)
            threshold = st.slider('Select the threshold to apply on survival function',0.0, 1.0,0.8)
            if st.button('Begin Inference'):
                selected_model = model_dict.get(option)
                current_directory = os.getcwd()
                final_directory = os.path.join(current_directory, r'trained_models')
                mdl_loc = final_directory + '/' + str(selected_model) +'.zip'
                st.write(mdl_loc)
                mdl_infer = load_model(mdl_loc)
                categories = st.session_state.categorical
                dataset = pd.get_dummies(dataframe, columns=categories, drop_first=True)
                surv_curves = mdl_infer.predict_survival(dataset)
                haz_rates = mdl_infer.predict_hazard(dataset)
                with st.expander("Click here to view Survival Curves"):
                    for idx,surv_curve in enumerate(surv_curves):
                        row_name = 'Row_' + str(idx)
                        fig = px.line(surv_curve,title=row_name)
                        fig.update_traces(line_color='#AA4A44', line_width=1)
                        st.plotly_chart(fig, use_container_width=True)
                with st.expander("Click here to view Hazard Rates"):
                    for idx,haz_rate in enumerate(haz_rates):
                        row_name = 'Row_' + str(idx)
                        fig = px.line(haz_rate,title=row_name)
                        fig.update_traces(line_color='#AA4A44', line_width=1)
                        st.plotly_chart(fig, use_container_width=True)

                t = np.where(surv_curves < threshold, 1 , 0)
                column_name = 'ttf_' + str(threshold)
                dataframe[column_name] = np.argmax(t,axis=1)
                st.dataframe(dataframe)


