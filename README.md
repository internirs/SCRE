# Ship Collision Risk Evaluation using AIS and Weather Data through Fuzzy Logic and Deep Learning

### Official Python implementation of the SAINT deep learning model on the Piraeus AIS Dataset.

# Usage
The dataset used in this study is the [Piraeus AIS Dataset](https://zenodo.org/records/6323416). In order to reproduce the results obtained in this study, execute the code in the following notebooks in the same order- ```1-pira-data-preprocessing.ipynb```, ```2-pira-scre.ipynb```, ```3-pira-enfactors.ipynb```, ```4-pira-ecriw-fuzzy-logic.ipynb```, ```5-pira-saint-model-training.ipynb```.


To perform the data quality checks and clean the dynamic AIS dataset, please consult the notebook [`1-pira-data-preprocessing.ipynb`](./1-pira-data-preprocessing.ipynb) At the start of this notebook, it is required to upload the July 2018 data in .csv format. In case the user wishes to skip performing the quality check and cleansing, they can directly use the data in the file [`gdf_sub_moving.zip`](./Data/gdf_sub_moving.zip) (under the Data folder), which is what they would have obtained at the end of the first notebook. Thus, the user can upload this file at the beginning of the second notebook and begin with the data pre-processing.

The next step of the data pre-processing stage involves calculating the basic collision risk index for all own-ship-target-ship pairs using the dynamic AIS data. The code that performs this procedure can be run using the notebook [`2-pira-scre.ipynb`](./2-pira-scre.ipynb). 

The notebook [`3-pira-enfactors.ipynb`](./3-pira-enfactors.ipynb) contains the code to pre-process the weather dataset and calculate the enfactor index which considers the weather parameters at both the own ship and the target ship.

The final stage of preparing the training dataset involves using fuzzy logic to combine the basic collision risk index and enfactor index to obtain the Enhanced CRI with Weather (ECRI-W). This section of code can be executed by running the notebook [`4-pira-ecriw-fuzzy-logic.ipynb`](./4-pira-ecriw-fuzzy-logic.ipynb). Additionally, code to evaluate correlation between the features of the dataset is available in [`correlation.ipynb`](./correlation.ipynb).

Having obtained the final dataset, the next steps are to split the dataset, train the SAINT model by running the [`5-pira-saint-model-training.ipynb`](./5-pira-saint-model-training.ipynb) notebook. This notebook also allows for model pretraining, hyperparameter tuning and visualising the performance results.

Finally, in order to compare the performance of the model with MLP Regressor model, consult the notebook [`6-comparison-scre-mlp.ipynb`](./6-comparison-scre-mlp.ipynb) and to compare with SVM, RVM and MLP models consult [`6-comparison-scre-svm-rvm-mlp.ipynb`](./6-comparison-scre-svm-rvm-mlp.ipynb).

# Contributors
Veer Kapadia; Engineering Science, University of Toronto

Anil Kumar Korupoju; Research and Development Division, Indian Register of Shipping
