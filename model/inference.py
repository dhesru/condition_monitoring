import streamlit as st
import pandas as pd
import numpy as np
import os
import os.path
import glob
import platform
import matplotlib.pyplot as plt

import plotly.express as px

if platform.system() == 'Windows':
    from pysurvival.pysurvival.utils import load_model
    from pysurvival.pysurvival.utils.display import create_risk_groups, create_risk_groups_custom
else:
    from pysurvival.utils import load_model
    from pysurvival.utils.display import create_risk_groups


model_dict = {'Linear MTLR':'LMTLR','Conditional Survival Forest': 'CSF',"Extra Survival Trees":'EST','Random Survival Forest':'RSF'}

def get_key(val,model_dict):
    for key, value in model_dict.items():
        if val == value:
            return key
    return "key doesn't exist"

def get_model_name(mdl):
    if platform.system() == 'Windows':
        model_name = mdl.split('\\')
    else:
        model_name = mdl.split('/')
    last_index = len(model_name) - 1
    v = model_name[last_index].split('.')
    return v[0]

def plt_risk_profile(mdl,X):
    risk = np.log(mdl.predict_risk(X))
    risk_hist = np.histogram(risk)


    np.max(risk_hist[1]) + ((np.max(risk_hist[1]) - np.min(risk_hist[1])) / 3)

    low_l_bound = np.min(risk_hist[1])
    low_u_bound = low_l_bound + ((np.max(risk_hist[1]) - np.min(risk_hist[1])) / 3)

    med_l_bound = low_u_bound
    med_u_bound = med_l_bound + ((np.max(risk_hist[1]) - np.min(risk_hist[1])) / 3)

    high_l_bound = med_u_bound
    high_u_bound = np.max(risk_hist[1])

    risk_groups = create_risk_groups_custom(model=mdl, X=X,
                                     use_log=True, num_bins=50, figure_size=(20, 4),
                                     low={'lower_bound': low_l_bound, 'upper_bound': low_u_bound, 'color': 'red'},
                                     medium={'lower_bound': med_l_bound, 'upper_bound': med_u_bound, 'color': 'green'},
                                     high={'lower_bound': high_l_bound, 'upper_bound': high_u_bound, 'color': 'blue'}
                                     )

    with st.expander("Click here to view Risk groups"):
        st.pyplot(risk_groups[0])
    risk_vals = risk_hist[1]
    risk_groups = risk_groups[1]

    risk_labels = list()

    asset_count = 0
    with st.expander("Click here to view Risk profiles"):
        for i, (label, (color, indexes)) in enumerate(risk_groups.items()):
            if len(indexes) == 0:
                continue
            X_ = X.values[indexes, :]

            for x in range(len(X_)):
                plt.figure().clear()
                survival = mdl.predict_survival(X_[x, :]).flatten()
                label_ = '{} risk'.format(label)
                plt.plot(mdl.times, survival, color=color, label=label_, lw=2)
                title = "Risk profile for asset " + str(asset_count)
                plt.legend(fontsize=12)
                plt.title(title, fontsize=15)
                plt.ylim(0, 1.05)
                st.pyplot(plt)
                plt.figure().clear()
                asset_count += 1
                risk_labels.append(label)
    return risk_labels

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
                mdl_infer = load_model(mdl_loc)
                try:
                    categories = st.session_state.categorical
                    dataset = pd.get_dummies(dataframe, columns=categories, drop_first=True)
                except Exception as AttributeError:
                    dataset = dataframe
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

                risk_labels = plt_risk_profile(mdl_infer,dataset)
                t = np.where(surv_curves < threshold, 1 , 0)
                column_name = 'ttf_' + str(threshold)
                dataframe[column_name] = np.argmax(t,axis=1)
                dataframe['Risk Level'] = risk_labels
                st.dataframe(dataframe)


