Overview
--------
[Surprisica](https://github.com/gitlamp/Surprisica) is a package based on [Surprise](http://surpriselib.com) for building and analyzing
context-aware recommender systems.

[Surprisica](https://github.com/gitlamp/Surprisica) **is specifically designed for location-based recommender systems.**

Please note that Surprisica does not support implicit ratings or content-based
information.

Installation
------------
```
$ git clone https://github.com/gitlamp/Surprisica.git
$ cd Surprisica
$ python setup.py install
```

Quick Start
-----------
Below, you can find an example showing how you can define a custom context-based Dataset and split it for 5-fold cross-validation, and then compute RMSE and MAE accuracy measures.
Some hybrid collaborative filtering models and custom similarity measures are added in this package that works appropriately for designing a Context-Aware Recommender in the social media.

```python
from surprisica import Reader
from surprisica import Dataset
from surprisica import CSR
from surprisica.model_selection import cross_validate

# Instantiate a Reader to read data features in your custom dataset.
reader = Reader(line_format='user location timestamp context-1 context-2 ...')

# Load dataset
data = Dataset.load_from_file(path='path-to-the-dataset', reader=reader)

# Add similarity options and create a model
sim_options = {'name': 'asymmetric_msd'}
model = CSR(k=10, min_k=5, sim_options=sim_options)

# Run 5-fold cross-validation and show results
cross_validate(algo=model, data=data, measures=['RMSE', 'MAE'], cv=5)
```  
You can also load the dataset and define your model by command line.
```commandline
 surprisica -algo CSR -params "{'name': 'cosine'}" -reader "Reader(line_format='user location timestamp cnx-1 cnx-2 ...')" -load-data 'path-to-the-dataset'
```

**Output:**

```
Evaluating RMSE, MAE of algorithm CSR on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
RMSE (testset)    0.3866  0.3405  0.8011  0.3605  1.1186  0.6015  0.3097  
MAE (testset)     0.1697  0.1789  0.2882  0.1548  0.4615  0.2506  0.1155  
Fit time          0.12    0.10    0.14    0.12    0.20    0.13    0.03    
Test time         0.04    0.05    0.09    0.04    0.05    0.05    0.02  
```
