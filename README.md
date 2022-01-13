# Thesis Replication Kit
This repository serves as replication kit for the master thesis: *"Are performance metrics suitable indicators for the cost saving potential of defect prediction models? An exploratory analysis"*.

## Thesis Abstract
Defect prediction models try to lead the software quality assurance resources by predicting the location of defects in a software project.
Through this, defect prediction models have the potential to be cost saving, which is the main reason to apply them in practice.
For the assessment of the quality of defect prediction models, a variety of performance metrics can be used.
As the purpose of defect prediction is cost saving, it is vital to know which of these metrics can be used as an indicator for the cost saving potential.
Thus, we try to establish a relationship between commonly used performance metrics and the cost saving potential of such models.
Since there is not enough evidence to have a clear expectation about the outcome, we conduct an exploratory analysis.
Therefore, we execute numerous bootstrap-based defect prediction experiments to gather data.
Based on this, we model the relationship with three different machine learning models.
Subsequently, we try to confirm the established relationship for defect prediction models in general by applying the found relationship to data of other realistic defect prediction models.
For the defect prediction models in our exploratory analysis, we were able to establish relationships between two groups of performance metrics and the cost saving potential.
One of the groups is related to whether there is any cost saving potential at all, while the other is able to categorize the size of the potential to a certain degree.
However, we were not able to confirm these relationships for defect prediction models in general.
Finally, it can be said that performance metrics can be suitable indicators for the cost saving potential of defect prediction models, but the relationship depends on the characteristics of the particular defect prediction model.

## Content
- `configCrossPare/` contains the configuration files for the CrossPare experiments.
- `data/` contains our (intermediate) result data.
- `plots/` contains the plot files created by the analysis (by `replication_notebook.ipynb`).
- `convert_dataframe.ipynb` has the purpose to manipulate the CrossPare result dataframe to fit the variables of our study.
- `db_credentials.env` is the credential file for a MySQL database for the storage of CrossPare results.
- `logit_optimizer.py` is a script for the optimization of a logit model based on training data.
- `replication_notebook.ipynb` is the notebook containing the analysis of the data.

## Instructions
For the replication of our experiments there are multiple entry points.
These are possible through the intermediate data stored in `data/`.
For the full replication continue reading. 
For using our results of the CrossPare experiments go to [Create Dataset from CrossPare Results](#create-dataset-from-crosspare-results).
For the analysis of our _performance metrics vs. cost saving potential_ dataset continue at [Run Analysis](#run-analysis).

#### Run Defect Prediction Experiments in CrossPare
1. Setup the [CrossPare](https://github.com/sherbold/CrossPare) tool.
2. Load the code metric [dataset](https://user.informatik.uni-goettingen.de/~sherbol/replicationkits/replication-kit-emse-2020-defect-prediction-data/release-level-data.tar.gz) and unpack it (archive: 587MB, unpacked: 6.57GB).
3. Fill up the `Crosspare/testdata/mynbou/` directory with the full dataset.
4. Setup a MySQL database and define the settings in a `mysql.cred` file in the CrossPare directory.
5. Run CrossPare with the folder `configCrossPare/bootstrap` as argument for the bootstrap experiment. These configurations need to be run 100 times each.
6. Run CrossPare with the folder `configCrosPare/generalization` as argument for the generalization experiment.

#### Create Dataset from CrossPare Results
The results from our run of the CrossPare experiments are saved in the files `data/database_metrics_vs_costsaving_bt_exp.csv` and `data/database_metrics_vs_costsaving_real.csv`.
To convert them into the dataframe for the evaluation, execute the `convert_dataframe.ipynb` notebook. 
For the conversion of CrossPare results directly from a MySQL database you have to make a few adjustments.
Overwrite the credentials in `db_credentials.env` with the credentials of your database.
Further, set the `USE_DATABASE` flag in the notebook to `True` and adjust the database table names before executing `convert_dataframe.ipynb`.
The resulting datasets with the variables for the study then overwrite our data in the files `data/metrics_vs_costsaving_bootstrap_experiment.csv` and `data/metrics_vs_costsaving_realistic_settings.csv`.
#### Run Analysis
Run the `replication_notebook.ipynb` notebook. Optionally, you can disable the `RERUN_LOGIT_OPTIMIZER` flag in order to directly use the optimization results stored in `data/logit_optim_grid_search.csv`. Otherwise, the `logit_optimizer.py` script is called by the notebook.

## Requirements
For the CrossPare experiments:
- Java 8
- MySQL database

For the notebooks of our study:
- Python 3.8 or higher
- jupyter-notebook or jupyter-lab
- graphviz

For Ubuntu 20.04 the setup to start a jupyter lab for the execution of the notebooks can be:
    
    sudo apt-get update    
    sudo apt-get install python3-venv build-essential python3-dev graphviz
    git clone https://github.com/steffentunkel/thesis_replication_kit
    cd thesis_replication_kit/
    python3 -m venv venv
    source venv/bin/activate
    pip install jupyterlab
    jupyter lab