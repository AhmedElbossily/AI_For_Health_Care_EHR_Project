## Problem defination: 
A healthcare startup that has created a groundbreaking diabetes drug that is ready for clinical trial testing. It is a very unique and sensitive drug that requires administering the drug over **at least 5-7 days** of time in the hospital with frequent monitoring/testing and patient medication adherence training with a mobile application. You have been provided *a patient dataset* from a client partner and **are tasked with** *building a predictive model* that can *identify* which type of patients the company should focus their efforts testing this drug on. **Target patients** are people that are likely to be in the hospital for *this duration* of time and will not incur significant additional costs for administering this drug to the patient and monitoring.  



## Expected Hospitalization Time Regression Model:
Utilizing a synthetic dataset(denormalized at the line level augmentation) built off of the UCI Diabetes readmission dataset, a **regression model** was buit that **predicts** the expected days of hospitalization time and **then convert** this to a **binary prediction** of whether to include or exclude that patient from the clinical trial.

   - This project uses the **Tensorflow Dataset API** to scalably extract, transform, and load datasets and build datasets aggregated at the line, encounter, and patient data levels(longitudinal)
   - **Analyzes EHR datasets** to check for common issues (data leakage, statistical properties, missing values, high cardinality) by performing exploratory data analysis with Tensorflow Data Analysis and Validation library.
   - Creates **categorical features** from Key Industry Code Sets (ICD, CPT, NDC) and reduce dimensionality for high cardinality features by using embeddings 
   - Create **derived features**(bucketing, cross-features, embeddings) utilizing Tensorflow feature columns on both continuous and categorical input features
   - Uses the **Tensorflow Probability library** to train a model that provides uncertainty range predictions that allow for risk adjustment/prioritization and triaging of predictions
   - **Analyze and determine biases** for a model for key demographic groups by evaluating performance metrics across groups by using the Aequitas framework 

## Dataset
- A [dataset](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008) from UC Irvine were used in this project.
- The **Data Schema** can be found [here](https://github.com/udacity/nd320-c1-emr-data-starter/tree/master/data_schema_references/). There are two CSVs that provide more details on the fields and some of the mapped values.
