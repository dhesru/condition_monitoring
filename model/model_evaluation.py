import streamlit as st
import os
import os.path
import glob
from model.inference import get_model_name,get_key,model_dict
from model.model_fitter import get_csvs
import pandas as pd
import numpy as np
import os

import platform
if platform.system() == 'Windows':
    from pysurvival.pysurvival.utils import load_model
else:
    from pysurvival.pysurvival.utils import load_model

def get_optimal_threshold(df,model):
    range_vals = np.linspace(0,1,41)
    categories = st.session_state.categorical
    dataset = pd.get_dummies(df, columns=categories, drop_first=True)
    surv_curves = model.predict_survival(dataset)
    filtered_arr_len = list()
    for ind,thresh in enumerate(range_vals):
        t = np.where(surv_curves < thresh, 1, 0)
        pred_ttf = np.argmax(t, axis=1)
        excess_ttf = df.lifetime - pred_ttf
        excess_ttf = excess_ttf.to_numpy()
        bel_7 = excess_ttf[excess_ttf < 7]
        abv_7 = excess_ttf[excess_ttf > 7]
        outside_7_day = len(bel_7) + len(abv_7)
        filtered_arr_len.append(outside_7_day)
    opt_idx = np.argmin(filtered_arr_len)
    return range_vals[opt_idx] * 100




def evaluate_models():
    st.title('Model Evaluation')
    if 'model' in st.session_state:
        st.dataframe(st.session_state.model)
    st.title('Get Optimal Threshold')

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
        option = st.selectbox('Select the Model to optimize', trained_models)
        csvs = get_csvs()
        if len(csvs) > 0:
            df = pd.read_csv(csvs[0])
            selected_model = model_dict.get(option)
            current_directory = os.getcwd()
            final_directory = os.path.join(current_directory, r'trained_models')
            mdl_loc = final_directory + '/' + str(selected_model) + '.zip'
            mdl_infer = load_model(mdl_loc)
            opt_thresh = get_optimal_threshold(df,mdl_infer)
            st.write('Optimal Threshold for the selected model ',opt_thresh)


