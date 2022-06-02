import streamlit as st
import pandas as pd
import os
import glob
import plotly.express as px
import plotly.figure_factory as ff
from sksurv.nonparametric import kaplan_meier_estimator
import matplotlib.pyplot as plt

def remove_ttf_event_categorical(df):
    col_names = list(df.columns)
    col_names.remove(st.session_state.ttf)
    col_names.remove(st.session_state.event)
    try:
        for cat in st.session_state.categorical:
            col_names.remove(cat)
    except Exception as AttributeError:
        pass
    return col_names

def kaplan_meier_plot(df,cat_var):
    df[st.session_state.event] = df[st.session_state.event].astype(bool)
    for value in df[cat_var].unique():
        mask = df[cat_var] == value
        ttf, event = kaplan_meier_estimator(df[st.session_state.event][mask], df[st.session_state.ttf][mask])
        plt.step(ttf, event, where="post", label="%s (n = %d)" % (value, mask.sum()))

    plt.ylabel("est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$")
    plt.legend(loc="best")
    return plt
def visualization():
    st.title('Data Visualization')
    files_path = os.path.join('.', '*')
    files = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True)
    csvs = [x for x in files if ".csv" in x]
    if len(csvs) > 0:
        df = pd.read_csv(csvs[0])
        if 'ttf' in st.session_state:
            features = remove_ttf_event_categorical(df)
            # Create distplot with custom bin_size
            data_list = list()
            for feat in features:
                data_list.append(df[feat].to_numpy())

            col1, col2 = st.columns([2,2])

            bin_size = [1 for x in range(len(features))]
            fig = ff.create_distplot(data_list, features, bin_size=bin_size)
            col1.caption("Distribution Plot")
            col1.plotly_chart(fig, use_container_width=True)

            corr_val = df[features].corr()
            corr_plt = px.imshow(corr_val, text_auto=True)
            corr_plt.update_coloraxes(showscale=False)
            col2.caption("Correlation Plot")
            col2.plotly_chart(corr_plt, use_container_width=True)
            try:
                catgeroical_vars = st.session_state.categorical
                v = st.columns(len(catgeroical_vars))
                for idx, cat_var in enumerate(catgeroical_vars):
                    plt = kaplan_meier_plot(df, cat_var)
                    header_str = "Kaplan Meier Plot for " + str(cat_var)
                    v[idx].caption(header_str)
                    v[idx].pyplot(plt)
                    plt.close()
            except Exception as AttributeError:
                pass


        else:
            st.write('Please select the variable types')
