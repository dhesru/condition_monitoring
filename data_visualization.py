import streamlit as st
import pandas as pd
import os
import glob
import plotly.express as px
import plotly.figure_factory as ff

def remove_ttf_event(df):
    col_names = list(df.columns)
    col_names.remove('ttf')
    col_names.remove('event')
    return col_names

def visualization():
    st.title('Data Visualization')
    files_path = os.path.join('.', '*')
    files = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True)
    csvs = [x for x in files if ".csv" in x]
    if len(csvs) > 0:
        df = pd.read_csv(csvs[0])
        features = remove_ttf_event(df)

        # Create distplot with custom bin_size
        data_list = list()
        for feat in features:
            data_list.append(df[feat].to_numpy())

        col1, col2 = st.columns([2,1])


        fig = ff.create_distplot(data_list, features, bin_size=[.1, .1, .1])
        col1.subheader("Distribution Plot")
        col1.plotly_chart(fig, use_container_width=True)

        corr_val = df[features].corr()
        corr_plt = px.imshow(corr_val, text_auto=True)
        col2.subheader("Correlation Plot")
        col2.plotly_chart(corr_plt, use_container_width=True)
