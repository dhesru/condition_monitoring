## Condition Monitoring App PoC
Proof-of-Concept Dashboard for future work on this area

#### This application comes with 5 key features. An explanation of these features can be found below. 

<strong>User Actions</strong> – User is expected to perform these actions 
<br><strong>App functions</strong> – Once User Action is met, app functions will be activated

_** Not all requires User Action_ 

<strong>Upload Data</strong> – Upload predictive maintenance data for visualization & model training purposes 

    User Actions

    - Upload data using Browse files option
    - Select TTF, event variable & categorical variables using the drop-down options
    - Click Confirm Variables selected to process the data and variables

    App Functions 

    - Uploaded data will be saved locally for visualization 
    - App will process the data based on given inputs

<strong>Data Visualization</strong> – Pre-defined charts are used to visualize the data  

    App Functions 

    - Histogram plots on numerical dataset 
    - Correlation plot
    - Kaplan Meier plot 

<strong>Model Training </strong> – Train your model using defined parameters.  

	User Actions 

    - Dropdown options for model type to train on
    - Train Test Split ratio using slider
    - Lastly click Begin Model Training to train your model 

	App functions 

    - Model begins training and stores the artifact locally

**Model Evaluation** – Evaluate the model using metrics. Optimal Threshold will be generated 

	App functions 

    - Shows Model results e.g. C-index, Brier Score
    - Obtain the optimal threshold for each model

**Model Inference** – Used to get predictions using a trained model 

	User actions 

    - Select the trained model to infer
    - Upload data for predictive maintenance
    - Select the appropriate threshold and click Begin Inference  

	App Functions 

    - Survival Curves and Hazard Rates for each asset will be generated
    - Time to failure will be generated for each asset 



#### Key tools used
- Python
- Streamlit
- PySurvival Package
- Plotly