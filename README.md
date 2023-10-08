<!DOCTYPE html>
<html>



<body>

  <h1>COVID-19 Prediction Machine Learning Model</h1>

  <img src="https://github.com/zak-soussi/Covid19_ML/tree/main/Visualisations/corona19.jpg" alt="COVID-19 Image">

  <h2>Overview</h2>

  <p>This repository contains a machine learning model for predicting whether a person will test positive for COVID-19 or not. The model is built using anonymized data from patients seen at the Hospital Israelita Albert Einstein in São Paulo, Brazil. The project is motivated by the need to make predictions in an overwhelmed healthcare system, considering limitations in performing tests for the detection of SARS-CoV-2. The dataset has undergone thorough analysis, preprocessing, and modeling phases.</p>

  <h2>Table of Contents</h2>

  <ul>
    <li><a href="#data">Data</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#model-performance">Model Performance</a></li>
    <li><a href="#exploratory-data-analysis">Exploratory Data Analysis</a></li>
    <li><a href="#preprocessing">Preprocessing</a></li>
    <li><a href="#modeling">Modeling</a></li>
    
  </ul>

  <h2>Data</h2>

  <p>The data used for training and testing the model is sourced from patients at the Hospital Israelita Albert Einstein in São Paulo, Brazil. This dataset contains anonymized data from patients who had samples collected for SARS-CoV-2 testing during their hospital visits. The data has been standardized following international practices and recommendations.</p>

  <h2>Installation</h2>

  <p>To run the code and use the model, you will need to set up a Python environment and install the required dependencies. You can do this using the following steps:</p>

  <ol>
    <li>Clone this repository to your local machine:</li>
    <pre>git clone https://github.com/zak-soussi/Covid19_ML.git</pre>
    <li>Create a virtual environment and activate it:</li>
    <pre>virtualenv venv
source venv/bin/activate</pre>
    <li>Install the required dependencies using <code>pip</code>:</li>
    <pre>pip install -r requirements.txt</pre>
  </ol>

  <h2>Usage</h2>

  <p>You can use the provided Jupyter Notebook or Python scripts to explore, preprocess, and model the data. Here's how you can get started:</p>

  <ol>
    <li>Follow the step-by-step instructions in the EDA / PreProcessing scripts to explore the data, understand the features, and perform necessary preprocessing.</li>
    <li>Dive into the modeling phase, where you can experiment with different machine learning models.</li>
  </ol>

  <h2>Model Performance</h2>

  <p>The model's performance was rigorously evaluated using recall and F1-score as the primary evaluation metrics. The final model achieved an 88% recall score and an F1-score of 51%. Further details and insights into the model's performance can be found in the Modeling script.</p>

  <img src="https://github.com/zak-soussi/Covid19_ML/tree/main/Visualisations/Modeling/FinalResult.png">

  <h2>Exploratory Data Analysis</h2>

  <p>In the exploratory data analysis (EDA) phase, the dataset was meticulously analyzed to uncover insights and patterns. This phase involved:</p>

  <ul>
    <li>Data visualization to understand feature distributions and relationships.</li>
    <li>Identifying potential correlations between variables.</li>
    <li>Gaining a deeper understanding of the data's context and characteristics.</li>
  </ul>

  <h2>Preprocessing</h2>

  <p>The preprocessing phase played a crucial role in ensuring that the data was ready for modeling. It encompassed:</p>

  <ul>
    <li>Data cleaning to handle missing or inconsistent values.</li>
    <li>Feature engineering to create relevant new variables.</li>
    <li>Standardization of clinical data to improve model accuracy.</li>
    <li>Feature encoding to convert categorical variables into a numerical format, allowing the model to work with them effectively.</li>
  </ul>

  <h2>Modeling</h2>

  <p>The modeling phase involved the exploration and selection of machine learning models. After experimenting with various models, the Support Vector Classifier was chosen as the final model. It was tuned and trained to make predictions on COVID-19 test results.</p>

</body>

</html>
