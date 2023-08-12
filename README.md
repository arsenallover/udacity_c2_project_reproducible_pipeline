# Building an end-to-end ML Pipeline for Short-Term Rental Prices in NYC

**Project description:** In this project, we will be deploying an end to end property rental price predictions using scikit-learn, mlflow and weights and biases. The pipeline estimates the typical price of a given property based on the price of similar properties. The focus on the project is on the MLops process such as the tracking of experiments, pipeline artifacts and the deployment of the inference pipeline rather than on the EDA and modelling. The latter are of secondary importance in this project.

Source: [Udacity - Machine Learning DevOps Engineer Nano-degree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821)


## Pipeline Configuration
The parameters controlling the pipeline are defined in the ``config.yaml`` file defined in
the root of the project. Hydra is used to manage this configuration file. 

The pipeline is defined in the ``main.py`` file in the root of the project. The file already
contains some boilerplate code as well as the download step. Your task will be to develop the
needed additional step, and then add them to the ``main.py`` file.

## Running the entire pipeline or just a selection of steps
Ar the root of the project execute:

```bash
>  mlflow run .
```
This will run the entire pipeline.

To be able to run one step at the time. Say you want to run only
the ``download`` step. The `main.py` is written so that the steps are defined at the top of the file, in the 
``_steps`` list, and can be selected by using the `steps` parameter on the command line:

```bash
> mlflow run . -P steps=download
```

If you want to run the ``download`` and the ``basic_cleaning`` steps, you can similarly do:
```bash
> mlflow run . -P steps=download,basic_cleaning
```

Override any other parameter in the configuration file using the Hydra syntax, by
providing it as a ``hydra_options`` parameter. For example, say that we want to set the parameter
modeling -> random_forest -> n_estimators to 10 and etl->min_price to 50:

```bash
> mlflow run . \
  -P steps=download,basic_cleaning \
  -P hydra_options="modeling.random_forest.n_estimators=10 etl.min_price=50"
```

## Environment clean-up
To clean up all conda environments created by `mlflow`  a list of the environments you are about to remove by executing:

```
> conda info --envs | grep mlflow | cut -f1 -d" "
```

Check the list and then execute the following to clean up (will iterate over all the environments created by `mlflow` and remove them):

**_NOTE_**: this will remove *ALL* the environments with a name starting with `mlflow`. Use at your own risk

```
> for e in $(conda info --envs | grep mlflow | cut -f1 -d" "); do conda uninstall --name $e --all -y;done
```

# 1.Exploratory Data Analysis (EDA)
* Component ``get_data`` 

Basic EDA performed in a notebook. The raw dataset has the following characteristics:
```
RangeIndex: 20000 entries, 0 to 19999
Data columns (total 16 columns):
 #   Column                          Non-Null Count  Dtype  
---  ------                          --------------  -----  
 0   id                              20000 non-null  int64  
 1   name                            19993 non-null  object 
 2   host_id                         20000 non-null  int64  
 3   host_name                       19992 non-null  object 
 4   neighbourhood_group             20000 non-null  object 
 5   neighbourhood                   20000 non-null  object 
 6   latitude                        20000 non-null  float64
 7   longitude                       20000 non-null  float64
 8   room_type                       20000 non-null  object 
 9   price                           20000 non-null  int64  
 10  minimum_nights                  20000 non-null  int64  
 11  number_of_reviews               20000 non-null  int64  
 12  last_review                     15877 non-null  object 
 13  reviews_per_month               15877 non-null  float64
 14  calculated_host_listings_count  20000 non-null  int64  
 15  availability_365                20000 non-null  int64  
dtypes: float64(3), int64(7), object(6)
```

**To run this pipeline component:**
```bash
> mlflow run . -P steps=download
```

# 2.Data Cleaning
* Component ``basic_cleaning`` 

Performs basic cleaning and filtering on the price feature and it range:
```
min_price: 10  # dollars
max_price: 350  # dollars
```

**To run this pipeline component:**
```bash
> mlflow run . -P steps=basic_cleaning
```


# 3.Data Testing
* Component ``data_check`` 

After the cleaning, a set of test validate the resulting dataset characteristics

**To run this pipeline component:**
```bash
> mlflow run . -P steps=data_check
```

# 4.Data Splitting
* Component ``train_val_test_split``

Splits the date is an training, validation and testing set with the following default parameters:
```
  # Fraction of data to use for test (the remaining will be used for train and validation)
  test_size: 0.2
  # Fraction of remaining data to use for validation
  val_size: 0.2
  # Fix this for reproducibility, change to have new splits
  random_seed: 42
  # Column to use for stratification (use "none" for no stratification)
  stratify_by: "neighbourhood_group"
```

**To run this pipeline component:**
```bash
> mlflow run . -P steps=data_split
```

# 5.Train Random Forest
* Component ``train_random_forest``

Train the Random Forest model with the with the following best tuned hyper parameters

**To run this pipeline component:**
```bash
> mlflow run . -P steps=train_random_forest
```

# 6.Optimize Hyper-parameters
Re-runs the entire pipeline varying the hyper-parameters of the Random Forest model. 
We use the multi-run feature (adding the `-m` option 
at the end of the `hydra_options` specification), and a grid search with the parameters: `modeling.max_tfidf_features` and `modeling.random_forest.max_features`

**To run this pipeline component:**
```bash
> mlflow run . \
   -P steps=train_random_forest \
   -P hydra_options="modeling.random_forest.max_depth=10,50,100 modeling.random_forest.max_features=0.1,0.33,0.5,0.75,1 modeling.max_tfidf_features=10,15,30 -m"
```

# 7.Select Best Model

From W&B interface and we select the best performing model considering the Mean Absolute Error as the main evaluation metric

# 8.Test & Evaluate Model
* Component ``test_regression_model``
Tests the production model against the test set. 

> Mean Absolute Error (MAE): MAE measures the average magnitude of the errors in a set of predictions, without considering their direction. It’s the average over the test sample of the absolute differences between prediction and actual observation where all individual differences have equal weight.

> R-squared is a goodness-of-fit measure for linear regression models. This statistic indicates the percentage of the variance in the dependent variable that the independent variables explain collectively. R-squared measures the strength of the relationship between your model and the dependent variable on a convenient 0 – 100% scale.

**To run this pipeline component:**
```bash
> mlflow run . -P steps=test_regression_model
```

# 9.Visualize the pipeline

From W&B interface and visualize the end to end pipeline


# 10.Release the pipeline

Using github, a release of the entire pipeline is created