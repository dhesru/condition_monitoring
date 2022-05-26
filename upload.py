import streamlit as st
import pandas as pd
import os
import glob


def save_uploaded_file(uploadedfile):
  with open(os.path.join("./",uploadedfile.name),"wb") as f:
     f.write(uploadedfile.getbuffer())
  return st.success("Saved file locally")


def data_uploader():
    st.title('Welcome to Condition Monitoring')

    uploaded_file = st.file_uploader("Upload a CSV file for condition monitoring..")
    if uploaded_file is not None:
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
            col_names = list(df.columns)
            ttf = st.selectbox(
                'please select the TTF variable',
                col_names)
            event = st.selectbox(
                'please select the event variable',
                col_names)
            categorical = st.multiselect(
                'Please select the categorical variables',
                col_names)

            if st.button('Confirm variables selected'):
                st.session_state.ttf = ttf
                st.session_state.event = event
                st.session_state.categorical = categorical