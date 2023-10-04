import streamlit as st
import pandas as pd
import numpy as np
import os
import os.path
import pickle
import glob
from sklearn.model_selection import train_test_split
import platform
if platform.system() == 'Windows':
    from pysurvival.pysurvival.utils.metrics import concordance_index,integrated_brier_score as ibs
    from pysurvival.pysurvival.utils.display import integrated_brier_score
    from pysurvival.pysurvival.models.multi_task import LinearMultiTaskModel
    from pysurvival.pysurvival.models.survival_forest import ConditionalSurvivalForestModel, ExtraSurvivalTreesModel,RandomSurvivalForestModel
    from pysurvival.pysurvival.utils import save_model
else:
    from pysurvival.utils.metrics import concordance_index,integrated_brier_score as ibs
    from pysurvival.utils.display import integrated_brier_score
    from pysurvival.models.multi_task import LinearMultiTaskModel
    from pysurvival.models.survival_forest import ConditionalSurvivalForestModel, ExtraSurvivalTreesModel,RandomSurvivalForestModel
    from pysurvival.utils import save_model

model_dict = {'Linear MTLR':'LMTLR','Conditional Survival Forest': 'CSF',"Extra Survival Trees":'EST','Random Survival Forest':'RSF'}

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
                        save_memory=False )
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

def create_trained_model():
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'trained_models')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    return final_directory

def get_csvs():
    files_path = os.path.join('.', '*')
    files = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True)
    csvs = [x for x in files if ".csv" in x]
    return csvs

def fit_models():
    st.title('Model Training')
    csvs = get_csvs()
    #option = st.selectbox('Model Type',('Linear MTLR', 'Conditional Survival Forest', 'Extra Survival Trees','Random Survival Forest'))
    option = st.selectbox('Model Type', ('Conditional Survival Forest', 'Extra Survival Trees', 'Random Survival Forest'))
    final_directory = create_trained_model()
    if len(csvs) > 0:
        df = pd.read_csv(csvs[0])
        train = st.slider('Train Test Split', 0, 100, 70)
        test = 100 - train
        st.write('Selected Training period: ', train)
        st.write('Selected Testing period: ', test)
        if st.button('Begin Model Training'):
            N = df.shape[0]
            index_train, index_test = train_test_split(range(N), test_size=test)
            st.write('Model Training has begun using ', option)
            time_column = st.session_state.ttf
            event_column = st.session_state.event

            # Encoding the categorical variables as one-hot vectors
            try:
                df = pd.get_dummies(df, columns=st.session_state.categorical, drop_first=True)
            except Exception as AttributeError:
                pass

            features = np.setdiff1d(df.columns, [time_column, event_column]).tolist()
            data_train = df.loc[index_train].reset_index(drop=True)
            data_test = df.loc[index_test].reset_index(drop=True)

            # Creating the X, T and E inputs
            X_train, X_test = data_train[features], data_test[features]
            T_train, T_test = data_train[time_column], data_test[time_column]
            E_train, E_test = data_train[event_column], data_test[event_column]
            mdl,mdl_type = model_fitting(option,X_train,T_train,E_train)

            st.write('Completed.')

            c_index = concordance_index(mdl_type, X_test, T_test, E_test)

            fig = integrated_brier_score(mdl_type, X_test, T_test, E_test, t_max=100,
                                   figure_size=(20, 6.5))
            st.pyplot(fig)

            ibs_score = ibs(mdl_type, X_test, T_test, E_test, t_max=None)

            if 'model' not in st.session_state:
                data_ = {
                    'model_type': [option],
                    'train_split': [train],
                    'test_split' : [test],
                    'concordance_index': [c_index],
                    'integrated_brier_score':[ibs_score]

                }

                st.session_state.model = pd.DataFrame(data_)


            else:
                data_ = {
                    'model_type': [option],
                    'train_split': [train],
                    'test_split': [test],
                    'concordance_index': [c_index],
                    'integrated_brier_score': [ibs_score]

                }
                new_df = pd.DataFrame(data_)
                result_df = pd.concat([new_df, st.session_state.model], ignore_index=True)

                st.session_state.model = result_df

            mdl_name = model_dict.get(option)

            if platform.system() == 'Windows':
                mdl_directory = str(final_directory) + '\\' + str(mdl_name) + '.zip'
            else:
                mdl_directory = str(final_directory) + '/' + str(mdl_name) + '.zip'

            save_model(mdl,mdl_directory)