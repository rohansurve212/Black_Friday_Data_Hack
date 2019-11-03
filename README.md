Black_Friday_Data_Hack (Python)
==============================

**Business Scenario**
A retail company "ABC Private Limited" wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month.

They want to build a model to predict the purchase amount of customer against various products which will help them to create personalized offer for customers against different products.

**Frame The Problem**
Before we dive into the data and start analyzing it, it's imperative to understand how does the company expect to use and benefit from this model?
We can categorize this problem as:

* **Supervised Learning task**: we are given labeled training data (e.g. we already know how much a customer spent on a specific product)

* **Regression task**: our algorithm is expected to predict the purchase amount a client is expected to spend on this day.

* **Plain batch learning**: since there is no continuous flow of data coming into our system, there is no particular need to adjust to changing data rapidly, and the data is small enough to fit in memory, so plain batch learning should work.

**Approach To Solution**
I have used the following steps to approach the best Machine Learning algorithm:
* Step 1. Get The Data
* Step 2. Build Features
* Step 3. Visualize The Data
* Step 4. Pre-Process Data
* Step 5. Train Machine Learning Models
* Step 6. Stacking and Blending Models
* Step 7. Evaluate Best Model on Unseen Data

**Project File Structure**
I have used the Cookie Cutter Data Science Project Template to structure my project.



    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been shuffled and sliced to suit my hardware requirements.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── main.ipynb         <- Jupyter Notebook file where the entire project resides  
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │       
    │   ├── make_dataset.py          <- Scripts to download or generate data
    │   │     
    │   ├── build_features.py        <- Scripts to split data into train, stacking, test sets and build new features that can help optimize our models
    │   │   
    │   ├── visualize_data.py        <- Scripts to generate plots to visualize distribution of different features and also the relationships between features and target
    │   │      
    │   ├── preprocess_data.py       <- Scripts to make the data ready to be processed by ML models 
    │   │
    │   ├── train_models.py          <- Scripts to train models and then evaluate the cross-validated prediction scores of these trained models 
    │   │                
    │   ├── blend_stacked_models.py  <- Script to stack previously trained models and blend them using a Random Forest Regressor  
    │   │   
    │   └── evaluate_models.py       <- Script to evaluate a model on previously unseen data    
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
