# Ship Collision Risk Assessment using AIS and Weather Data through Fuzzy Logic and Deep Learning

### Official Python implementation of the SAINT deep learning model on the Piraeus AIS Dataset.

# Usage
In order to reproduce the results obtained in this study, execute the code in the following notebooks in the same order- ```1_Pira_data_preprocessing.ipynb```, ```2_Pira_VCRA.ipynb```, ```3_Pira_Enfactors.ipynb```, ```4_Pira_Consequence_Fuzzy_logic.ipynb```, ```5_Pira_SAINT_model_training.ipynb```.


To perform the data quality checks and clean the dynamic AIS dataset, please consult the notebook [`1. Pira_data_preprocessing.ipynb`](./1_Pira_data_preprocessing.ipynb)

The next step of the data pre-processing stage involves calculating the basic collision risk index for all own-ship-target-ship pairs using the dynamic AIS data. The code that performs this procedure can be run using the notebook [`2_Pira_VCRA.ipynb`](./2_Pira_VCRA.ipynb)`. 

The notebook [`3_Pira_Enfactors.ipynb`](./3_Pira_Enfactors.ipynb) contains the code to pre-process the weather dataset and calculate the enfactor index which considers the weather parameters at both the own ship and the target ship.

The final stage of preparing the training dataset involves using fuzzy logic to combine the basic collision risk index and enfactor index to obtain the final collision risk index (or Consequence). This section of code can be executed by running the notebook [`4_Pira_Consequence_Fuzzy_logic.ipynb`](./4_Pira_Consequence_Fuzzy_logic.ipynb).

Having obtained the final dataset, the next steps are to split the dataset, train the SAINT model by running the [`5_Pira_SAINT_model_training.ipynb`](./5_Pira_SAINT_model_training.ipynb) notebook. This notebook also allows for model pretraining, hyperparameter tuning and visualising the performance results.

Finally, in order to compare the performance of the model with other machine learning models, consult the notebook [`6_Comparison_VCRA_MLP.ipynb`](./6_Comparison_VCRA_MLP.ipynb).

# Contributors
Veer Kapadia; Engineering Science, University of Toronto

Anil Kumar Korupoju; Research and Development Division, Indian Register of Shipping
