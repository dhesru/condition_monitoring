import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from pysurvival.utils.metrics import concordance_index,integrated_brier_score as ibs
from pysurvival.pysurvival.utils.display import integrated_brier_score, correlation_matrix, compare_to_actual
from pysurvival.pysurvival.models.multi_task import LinearMultiTaskModel
from pysurvival.pysurvival.models.survival_forest import ConditionalSurvivalForestModel, ExtraSurvivalTreesModel,RandomSurvivalForestModel

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

def fit_models():
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
            N = df.shape[0]
            index_train, index_test = train_test_split(range(N), test_size=test)
            st.write('Model Training has begun using ', option)
            time_column = st.session_state.ttf
            event_column = st.session_state.event

            # Encoding the categorical variables as one-hot vectors
            df = pd.get_dummies(df, columns=st.session_state.categorical, drop_first=True)
            features = np.setdiff1d(df.columns, [time_column,event_column]).tolist()
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

            # results = compare_to_actual(mdl_type, X_test, T_test, E_test,
            #                             is_at_risk=False, figure_size=(16, 6),
            #                             metrics=['rmse', 'mean', 'median'])
            # st.pyplot(results)
            ibs_score = ibs(mdl_type, X_test, T_test, E_test, t_max=None)

            if 'model' not in st.session_state:
                model_params_df = pd.DataFrame(columns=['model_type','train_split','test_split','concordance_index','integrated_brier_score'])
                st.session_state.model = model_params_df.append({'model_type':option,'train_split':train,'test_split':test,'concordance_index':c_index,
                                                                 'integrated_brier_score':ibs_score},ignore_index=True)

            else:
                res_dict = {'model_type':option,'train_split':train,'test_split':test,'concordance_index':c_index,'integrated_brier_score':ibs_score}
                st.session_state.model = st.session_state.model.append(res_dict,ignore_index=True)