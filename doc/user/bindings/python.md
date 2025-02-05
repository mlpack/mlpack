# mlpack Python binding documentation

## mlpack overview

mlpack is an intuitive, fast, and flexible header-only C++ machine learning library with bindings to other languages.  It aims to provide fast, lightweight implementations of both common and cutting-edge machine learning algorithms.

This reference page details mlpack's bindings to Python.

Further useful mlpack documentation links are given below.

 - [mlpack homepage](https://www.mlpack.org/)
 - [mlpack on Github](https://github.com/mlpack/mlpack)
 - [mlpack main documentation page](https://www.mlpack.org/doc/index.html)

See also the quickstart guide for Python:

 - [Python Quickstart](../../quickstart/python.md)

## Data Formats

<div id="data-formats-div" markdown="1">
mlpack bindings for Python take and return a restricted set of types, for simplicity.  These include primitive types, matrix/vector types, categorical matrix types, and model types. Each type is detailed below.

 - `int`{: #doc_int}: An integer (i.e., "1").
 - `float`{: #doc_float }: A floating-point number (i.e., "0.5").
 - `bool`{: #doc_bool }: A boolean flag option (True or False).
 - `str`{: #doc_str }: A character string (i.e., "hello").
 - `list of ints`{: #doc_list_of_ints }: A list of integers; i.e., `[0, 1, 2]`.
 - `list of strs`{: #doc_list_of_strs }: A list of strings; i.e., `["hello", "goodbye"]`.
 - `matrix`{: #doc_matrix }: A 2-d arraylike containing data.  This can be a list of lists, a numpy ndarray, or a pandas DataFrame.  If the dtype is not already float64, it will be converted.
 - `int matrix`{: #doc_int_matrix }: A 2-d arraylike containing data with a uint64 dtype.  This can be a list of lists, a numpy ndarray, or a pandas DataFrame.  If the dtype is not already uint64, it will be converted.
 - `vector`{: #doc_vector }: A 1-d arraylike containing data.  This can be a 2-d matrix where one dimension has size 1, or it can also be a list, a numpy 1-d ndarray, or a 1-d pandas DataFrame.  If the dtype is not already float64, it will be converted.
 - `int vector`{: #doc_int_vector }: A 1-d arraylike containing data with a uint64 dtype.  This can be a 2-d matrix where one dimension has size 1, or it can also be a list, a numpy 1-d ndarray, or a 1-d pandas DataFrame.  If the dtype is not already uint64, it will be converted.
 - `categorical matrix`{: #doc_categorical_matrix }: A 2-d arraylike containing data.  Like the regular 2-d matrices, this can be a list of lists, a numpy ndarray, or a pandas DataFrame. However, this type can also accept a pandas DataFrame that has columns of type 'CategoricalDtype'.  These categorical values will be converted to numeric indices before being passed to mlpack, and then inside mlpack they will be properly treated as categorical variables, so there is no need to do one-hot encoding for this matrix type.  If the dtype of the given matrix is not already float64, it will be converted.
 - `mlpackModelType`{: #doc_model }: An mlpack model pointer.  This type can be pickled to or from disk, and internally holds a pointer to C++ memory containing the mlpack model.  This model pointer has 2 methods with which the parameters of the model can be inspected as well as changed through Python.  The `get_cpp_params()` method returns a python ordered dictionary that contains all the parameters of the model.  These parameters can be inspected and changed.  To set new parameters for a model, pass the modified dictionary (without deleting any keys) to the `set_cpp_params()` method.
</div>


## approx_kfn()
{: #approx_kfn }

#### Approximate furthest neighbor search
{: #approx_kfn_descr }

```python
>>> from mlpack import approx_kfn
>>> d = approx_kfn(algorithm='ds', calculate_error=False,
        check_input_matrices=False, copy_all_inputs=False,
        exact_distances=np.empty([0, 0]), input_model=None, k=0,
        num_projections=5, num_tables=5, query=np.empty([0, 0]),
        reference=np.empty([0, 0]), verbose=False)
>>> distances = d['distances']
>>> neighbors = d['neighbors']
>>> output_model = d['output_model']
```

An implementation of two strategies for furthest neighbor search.  This can be used to compute the furthest neighbor of query point(s) from a set of points; furthest neighbor models can be saved and reused with future query point(s). [Detailed documentation](#approx_kfn_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `algorithm` | [`str`](#doc_str) | Algorithm to use: 'ds' or 'qdafn'. | `'ds'` |
| `calculate_error` | [`bool`](#doc_bool) | If set, calculate the average distance error for the first furthest neighbor only. | `False` |
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `exact_distances` | [`matrix`](#doc_matrix) | Matrix containing exact distances to furthest neighbors; this can be used to avoid explicit calculation when --calculate_error is set. | `np.empty([0, 0])` |
| `input_model` | [`ApproxKFNModelType`](#doc_model) | File containing input model. | `None` |
| `k` | [`int`](#doc_int) | Number of furthest neighbors to search for. | `0` |
| `num_projections` | [`int`](#doc_int) | Number of projections to use in each hash table. | `5` |
| `num_tables` | [`int`](#doc_int) | Number of hash tables to use. | `5` |
| `query` | [`matrix`](#doc_matrix) | Matrix containing query points. | `np.empty([0, 0])` |
| `reference` | [`matrix`](#doc_matrix) | Matrix containing the reference dataset. | `np.empty([0, 0])` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `distances` | [`matrix`](#doc_matrix) | Matrix to save furthest neighbor distances to. | 
| `neighbors` | [`int matrix`](#doc_int_matrix) | Matrix to save neighbor indices to. | 
| `output_model` | [`ApproxKFNModelType`](#doc_model) | File to save output model to. | 

### Detailed documentation
{: #approx_kfn_detailed-documentation }

This program implements two strategies for furthest neighbor search. These strategies are:

 - The 'qdafn' algorithm from "Approximate Furthest Neighbor in High Dimensions" by R. Pagh, F. Silvestri, J. Sivertsen, and M. Skala, in Similarity Search and Applications 2015 (SISAP).
 - The 'DrusillaSelect' algorithm from "Fast approximate furthest neighbors with data-dependent candidate selection", by R.R. Curtin and A.B. Gardner, in Similarity Search and Applications 2016 (SISAP).

These two strategies give approximate results for the furthest neighbor search problem and can be used as fast replacements for other furthest neighbor techniques such as those found in the mlpack_kfn program.  Note that typically, the 'ds' algorithm requires far fewer tables and projections than the 'qdafn' algorithm.

Specify a reference set (set to search in) with `reference`, specify a query set with `query`, and specify algorithm parameters with `num_tables` and `num_projections` (or don't and defaults will be used).  The algorithm to be used (either 'ds'---the default---or 'qdafn')  may be specified with `algorithm`.  Also specify the number of neighbors to search for with `k`.

Note that for 'qdafn' in lower dimensions, `num_projections` may need to be set to a high value in order to return results for each query point.

If no query set is specified, the reference set will be used as the query set.  The `output_model` output parameter may be used to store the built model, and an input model may be loaded instead of specifying a reference set with the `input_model` option.

Results for each query point can be stored with the `neighbors` and `distances` output parameters.  Each row of these output matrices holds the k distances or neighbor indices for each query point.

### Example
For example, to find the 5 approximate furthest neighbors with `'reference_set'` as the reference set and `'query_set'` as the query set using DrusillaSelect, storing the furthest neighbor indices to `'neighbors'` and the furthest neighbor distances to `'distances'`, one could call

```python
>>> output = approx_kfn(query=query_set, reference=reference_set, k=5,
  algorithm='ds')
>>> neighbors = output['neighbors']
>>> distances = output['distances']
```

and to perform approximate all-furthest-neighbors search with k=1 on the set `'data'` storing only the furthest neighbor distances to `'distances'`, one could call

```python
>>> output = approx_kfn(reference=reference_set, k=1)
>>> distances = output['distances']
```

A trained model can be re-used.  If a model has been previously saved to `'model'`, then we may find 3 approximate furthest neighbors on a query set `'new_query_set'` using that model and store the furthest neighbor indices into `'neighbors'` by calling

```python
>>> output = approx_kfn(input_model=model, query=new_query_set, k=3)
>>> neighbors = output['neighbors']
```

### See also

 - [k-furthest-neighbor search](#kfn)
 - [k-nearest-neighbor search](#knn)
 - [Fast approximate furthest neighbors with data-dependent candidate selection (pdf)](http://ratml.org/pub/pdf/2016fast.pdf)
 - [Approximate furthest neighbor in high dimensions (pdf)](https://www.rasmuspagh.net/papers/approx-furthest-neighbor-SISAP15.pdf)
 - [QDAFN class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/approx_kfn/qdafn.hpp)
 - [DrusillaSelect class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/approx_kfn/drusilla_select.hpp)

## bayesian_linear_regression()
{: #bayesian_linear_regression }

#### BayesianLinearRegression
{: #bayesian_linear_regression_descr }

```python
>>> from mlpack import bayesian_linear_regression
>>> d = bayesian_linear_regression(center=False,
        check_input_matrices=False, copy_all_inputs=False, input_=np.empty([0,
        0]), input_model=None, responses=np.empty([0]), scale=False,
        test=np.empty([0, 0]), verbose=False)
>>> output_model = d['output_model']
>>> predictions = d['predictions']
>>> stds = d['stds']
```

An implementation of the bayesian linear regression. [Detailed documentation](#bayesian_linear_regression_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `center` | [`bool`](#doc_bool) | Center the data and fit the intercept if enabled. | `False` |
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `input_` | [`matrix`](#doc_matrix) | Matrix of covariates (X). | `np.empty([0, 0])` |
| `input_model` | [`BayesianLinearRegression<>Type`](#doc_model) | Trained BayesianLinearRegression model to use. | `None` |
| `responses` | [`vector`](#doc_vector) | Matrix of responses/observations (y). | `np.empty([0])` |
| `scale` | [`bool`](#doc_bool) | Scale each feature by their standard deviations if enabled. | `False` |
| `test` | [`matrix`](#doc_matrix) | Matrix containing points to regress on (test points). | `np.empty([0, 0])` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output_model` | [`BayesianLinearRegression<>Type`](#doc_model) | Output BayesianLinearRegression model. | 
| `predictions` | [`matrix`](#doc_matrix) | If --test_file is specified, this file is where the predicted responses will be saved. | 
| `stds` | [`matrix`](#doc_matrix) | If specified, this is where the standard deviations of the predictive distribution will be saved. | 

### Detailed documentation
{: #bayesian_linear_regression_detailed-documentation }

An implementation of the bayesian linear regression.
This model is a probabilistic view and implementation of the linear regression. The final solution is obtained by computing a posterior distribution from gaussian likelihood and a zero mean gaussian isotropic  prior distribution on the solution. 
Optimization is AUTOMATIC and does not require cross validation. The optimization is performed by maximization of the evidence function. Parameters are tuned during the maximization of the marginal likelihood. This procedure includes the Ockham's razor that penalizes over complex solutions. 

This program is able to train a Bayesian linear regression model or load a model from file, output regression predictions for a test set, and save the trained model to a file.

To train a BayesianLinearRegression model, the `input_` and `responses`parameters must be given. The `center`and `scale` parameters control the centering and the normalizing options. A trained model can be saved with the `output_model`. If no training is desired at all, a model can be passed via the `input_model` parameter.

The program can also provide predictions for test data using either the trained model or the given input model.  Test points can be specified with the `test` parameter.  Predicted responses to the test points can be saved with the `predictions` output parameter. The corresponding standard deviation can be save by precising the `stds` parameter.

### Example
For example, the following command trains a model on the data `'data'` and responses `'responses'`with center set to true and scale set to false (so, Bayesian linear regression is being solved, and then the model is saved to `'blr_model'`:

```python
>>> output = bayesian_linear_regression(input_=data, responses=responses,
  center=1, scale=0)
>>> blr_model = output['output_model']
```

The following command uses the `'blr_model'` to provide predicted  responses for the data `'test'` and save those  responses to `'test_predictions'`: 

```python
>>> output = bayesian_linear_regression(input_model=blr_model, test=test)
>>> test_predictions = output['predictions']
```

Because the estimator computes a predictive distribution instead of a simple point estimate, the `stds` parameter allows one to save the prediction uncertainties: 

```python
>>> output = bayesian_linear_regression(input_model=blr_model, test=test)
>>> test_predictions = output['predictions']
>>> stds = output['stds']
```

### See also

 - [Bayesian Interpolation](https://cs.uwaterloo.ca/~mannr/cs886-w10/mackay-bayesian.pdf)
 - [Bayesian Linear Regression, Section 3.3](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
 - [BayesianLinearRegression C++ class documentation](../../user/methods/bayesian_linear_regression.md)

## cf()
{: #cf }

#### Collaborative Filtering
{: #cf_descr }

```python
>>> from mlpack import cf
>>> d = cf(algorithm='NMF', all_user_recommendations=False,
        check_input_matrices=False, copy_all_inputs=False, input_model=None,
        interpolation='average', iteration_only_termination=False,
        max_iterations=1000, min_residue=1e-05, neighbor_search='euclidean',
        neighborhood=5, normalization='none', query=np.empty([0, 0],
        dtype=np.uint64), rank=0, recommendations=5, seed=0, test=np.empty([0,
        0]), training=np.empty([0, 0]), verbose=False)
>>> output = d['output']
>>> output_model = d['output_model']
```

An implementation of several collaborative filtering (CF) techniques for recommender systems.  This can be used to train a new CF model, or use an existing CF model to compute recommendations. [Detailed documentation](#cf_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `algorithm` | [`str`](#doc_str) | Algorithm used for matrix factorization. | `'NMF'` |
| `all_user_recommendations` | [`bool`](#doc_bool) | Generate recommendations for all users. | `False` |
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `input_model` | [`CFModelType`](#doc_model) | Trained CF model to load. | `None` |
| `interpolation` | [`str`](#doc_str) | Algorithm used for weight interpolation. | `'average'` |
| `iteration_only_termination` | [`bool`](#doc_bool) | Terminate only when the maximum number of iterations is reached. | `False` |
| `max_iterations` | [`int`](#doc_int) | Maximum number of iterations. If set to zero, there is no limit on the number of iterations. | `1000` |
| `min_residue` | [`float`](#doc_float) | Residue required to terminate the factorization (lower values generally mean better fits). | `1e-05` |
| `neighbor_search` | [`str`](#doc_str) | Algorithm used for neighbor search. | `'euclidean'` |
| `neighborhood` | [`int`](#doc_int) | Size of the neighborhood of similar users to consider for each query user. | `5` |
| `normalization` | [`str`](#doc_str) | Normalization performed on the ratings. | `'none'` |
| `query` | [`int matrix`](#doc_int_matrix) | List of query users for which recommendations should be generated. | `np.empty([0, 0], dtype=np.uint64)` |
| `rank` | [`int`](#doc_int) | Rank of decomposed matrices (if 0, a heuristic is used to estimate the rank). | `0` |
| `recommendations` | [`int`](#doc_int) | Number of recommendations to generate for each query user. | `5` |
| `seed` | [`int`](#doc_int) | Set the random seed (0 uses std::time(NULL)). | `0` |
| `test` | [`matrix`](#doc_matrix) | Test set to calculate RMSE on. | `np.empty([0, 0])` |
| `training` | [`matrix`](#doc_matrix) | Input dataset to perform CF on. | `np.empty([0, 0])` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output` | [`int matrix`](#doc_int_matrix) | Matrix that will store output recommendations. | 
| `output_model` | [`CFModelType`](#doc_model) | Output for trained CF model. | 

### Detailed documentation
{: #cf_detailed-documentation }

This program performs collaborative filtering (CF) on the given dataset. Given a list of user, item and preferences (the `training` parameter), the program will perform a matrix decomposition and then can perform a series of actions related to collaborative filtering.  Alternately, the program can load an existing saved CF model with the `input_model` parameter and then use that model to provide recommendations or predict values.

The input matrix should be a 3-dimensional matrix of ratings, where the first dimension is the user, the second dimension is the item, and the third dimension is that user's rating of that item.  Both the users and items should be numeric indices, not names. The indices are assumed to start from 0.

A set of query users for which recommendations can be generated may be specified with the `query` parameter; alternately, recommendations may be generated for every user in the dataset by specifying the `all_user_recommendations` parameter.  In addition, the number of recommendations per user to generate can be specified with the `recommendations` parameter, and the number of similar users (the size of the neighborhood) to be considered when generating recommendations can be specified with the `neighborhood` parameter.

For performing the matrix decomposition, the following optimization algorithms can be specified via the `algorithm` parameter:

 - 'RegSVD' -- Regularized SVD using a SGD optimizer
 - 'NMF' -- Non-negative matrix factorization with alternating least squares update rules
 - 'BatchSVD' -- SVD batch learning
 - 'SVDIncompleteIncremental' -- SVD incomplete incremental learning
 - 'SVDCompleteIncremental' -- SVD complete incremental learning
 - 'BiasSVD' -- Bias SVD using a SGD optimizer
 - 'SVDPP' -- SVD++ using a SGD optimizer
 - 'RandSVD' -- RandomizedSVD learning
 - 'QSVD' -- QuicSVD learning
 - 'BKSVD' -- Block Krylov SVD learning


The following neighbor search algorithms can be specified via the `neighbor_search` parameter:

 - 'cosine'  -- Cosine Search Algorithm
 - 'euclidean'  -- Euclidean Search Algorithm
 - 'pearson'  -- Pearson Search Algorithm


The following weight interpolation algorithms can be specified via the `interpolation` parameter:

 - 'average'  -- Average Interpolation Algorithm
 - 'regression'  -- Regression Interpolation Algorithm
 - 'similarity'  -- Similarity Interpolation Algorithm


The following ranking normalization algorithms can be specified via the `normalization` parameter:

 - 'none'  -- No Normalization
 - 'item_mean'  -- Item Mean Normalization
 - 'overall_mean'  -- Overall Mean Normalization
 - 'user_mean'  -- User Mean Normalization
 - 'z_score'  -- Z-Score Normalization

A trained model may be saved to with the `output_model` output parameter.

### Example
To train a CF model on a dataset `'training_set'` using NMF for decomposition and saving the trained model to `'model'`, one could call: 

```python
>>> output = cf(training=training_set, algorithm='NMF')
>>> model = output['output_model']
```

Then, to use this model to generate recommendations for the list of users in the query set `'users'`, storing 5 recommendations in `'recommendations'`, one could call 

```python
>>> output = cf(input_model=model, query=users, recommendations=5)
>>> recommendations = output['output']
```

### See also

 - [Collaborative Filtering on Wikipedia](https://en.wikipedia.org/wiki/Collaborative_filtering)
 - [Matrix factorization on Wikipedia](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems))
 - [Matrix factorization techniques for recommender systems (pdf)](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=cf17f85a0a7991fa01dbfb3e5878fbf71ea4bdc5)
 - [CFType class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/cf/cf.hpp)

## dbscan()
{: #dbscan }

#### DBSCAN clustering
{: #dbscan_descr }

```python
>>> from mlpack import dbscan
>>> d = dbscan(check_input_matrices=False, copy_all_inputs=False,
        epsilon=1, input_=np.empty([0, 0]), min_size=5, naive=False,
        selection_type='ordered', single_mode=False, tree_type='kd',
        verbose=False)
>>> assignments = d['assignments']
>>> centroids = d['centroids']
```

An implementation of DBSCAN clustering.  Given a dataset, this can compute and return a clustering of that dataset. [Detailed documentation](#dbscan_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `epsilon` | [`float`](#doc_float) | Radius of each range search. | `1` |
| `input_` | [`matrix`](#doc_matrix) | Input dataset to cluster. | `**--**` |
| `min_size` | [`int`](#doc_int) | Minimum number of points for a cluster. | `5` |
| `naive` | [`bool`](#doc_bool) | If set, brute-force range search (not tree-based) will be used. | `False` |
| `selection_type` | [`str`](#doc_str) | If using point selection policy, the type of selection to use ('ordered', 'random'). | `'ordered'` |
| `single_mode` | [`bool`](#doc_bool) | If set, single-tree range search (not dual-tree) will be used. | `False` |
| `tree_type` | [`str`](#doc_str) | If using single-tree or dual-tree search, the type of tree to use ('kd', 'r', 'r-star', 'x', 'hilbert-r', 'r-plus', 'r-plus-plus', 'cover', 'ball'). | `'kd'` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `assignments` | [`int vector`](#doc_int_vector) | Output matrix for assignments of each point. | 
| `centroids` | [`matrix`](#doc_matrix) | Matrix to save output centroids to. | 

### Detailed documentation
{: #dbscan_detailed-documentation }

This program implements the DBSCAN algorithm for clustering using accelerated tree-based range search.  The type of tree that is used may be parameterized, or brute-force range search may also be used.

The input dataset to be clustered may be specified with the `input_` parameter; the radius of each range search may be specified with the `epsilon` parameters, and the minimum number of points in a cluster may be specified with the `min_size` parameter.

The `assignments` and `centroids` output parameters may be used to save the output of the clustering. `assignments` contains the cluster assignments of each point, and `centroids` contains the centroids of each cluster.

The range search may be controlled with the `tree_type`, `single_mode`, and `naive` parameters.  `tree_type` can control the type of tree used for range search; this can take a variety of values: 'kd', 'r', 'r-star', 'x', 'hilbert-r', 'r-plus', 'r-plus-plus', 'cover', 'ball'. The `single_mode` parameter will force single-tree search (as opposed to the default dual-tree search), and '`naive` will force brute-force range search.

### Example
An example usage to run DBSCAN on the dataset in `'input'` with a radius of 0.5 and a minimum cluster size of 5 is given below:

```python
>>> dbscan(input_=input, epsilon=0.5, min_size=5)
```

### See also

 - [DBSCAN on Wikipedia](https://en.wikipedia.org/wiki/DBSCAN)
 - [A density-based algorithm for discovering clusters in large spatial databases with noise (pdf)](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf)
 - [DBSCAN class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/dbscan/dbscan.hpp)

## decision_tree()
{: #decision_tree }

#### Decision tree
{: #decision_tree_descr }

```python
>>> from mlpack import decision_tree
>>> d = decision_tree(check_input_matrices=False, copy_all_inputs=False,
        input_model=None, labels=np.empty([0], dtype=np.uint64),
        maximum_depth=0, minimum_gain_split=1e-07, minimum_leaf_size=20,
        print_training_accuracy=False, test=np.empty([0, 0]),
        test_labels=np.empty([0], dtype=np.uint64), training=np.empty([0, 0]),
        verbose=False, weights=np.empty([0, 0]))
>>> output_model = d['output_model']
>>> predictions = d['predictions']
>>> probabilities = d['probabilities']
```

An implementation of an ID3-style decision tree for classification, which supports categorical data.  Given labeled data with numeric or categorical features, a decision tree can be trained and saved; or, an existing decision tree can be used for classification on new points. [Detailed documentation](#decision_tree_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `input_model` | [`DecisionTreeModelType`](#doc_model) | Pre-trained decision tree, to be used with test points. | `None` |
| `labels` | [`int vector`](#doc_int_vector) | Training labels. | `np.empty([0], dtype=np.uint64)` |
| `maximum_depth` | [`int`](#doc_int) | Maximum depth of the tree (0 means no limit). | `0` |
| `minimum_gain_split` | [`float`](#doc_float) | Minimum gain for node splitting. | `1e-07` |
| `minimum_leaf_size` | [`int`](#doc_int) | Minimum number of points in a leaf. | `20` |
| `print_training_accuracy` | [`bool`](#doc_bool) | Print the training accuracy. | `False` |
| `test` | [`categorical matrix`](#doc_categorical_matrix) | Testing dataset (may be categorical). | `np.empty([0, 0])` |
| `test_labels` | [`int vector`](#doc_int_vector) | Test point labels, if accuracy calculation is desired. | `np.empty([0], dtype=np.uint64)` |
| `training` | [`categorical matrix`](#doc_categorical_matrix) | Training dataset (may be categorical). | `np.empty([0, 0])` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |
| `weights` | [`matrix`](#doc_matrix) | The weight of labels | `np.empty([0, 0])` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output_model` | [`DecisionTreeModelType`](#doc_model) | Output for trained decision tree. | 
| `predictions` | [`int vector`](#doc_int_vector) | Class predictions for each test point. | 
| `probabilities` | [`matrix`](#doc_matrix) | Class probabilities for each test point. | 

### Detailed documentation
{: #decision_tree_detailed-documentation }

Train and evaluate using a decision tree.  Given a dataset containing numeric or categorical features, and associated labels for each point in the dataset, this program can train a decision tree on that data.

The training set and associated labels are specified with the `training` and `labels` parameters, respectively.  The labels should be in the range `[0, num_classes - 1]`. Optionally, if `labels` is not specified, the labels are assumed to be the last dimension of the training dataset.

When a model is trained, the `output_model` output parameter may be used to save the trained model.  A model may be loaded for predictions with the `input_model` parameter.  The `input_model` parameter may not be specified when the `training` parameter is specified.  The `minimum_leaf_size` parameter specifies the minimum number of training points that must fall into each leaf for it to be split.  The `minimum_gain_split` parameter specifies the minimum gain that is needed for the node to split.  The `maximum_depth` parameter specifies the maximum depth of the tree.  If `print_training_accuracy` is specified, the training accuracy will be printed.

Test data may be specified with the `test` parameter, and if performance numbers are desired for that test set, labels may be specified with the `test_labels` parameter.  Predictions for each test point may be saved via the `predictions` output parameter.  Class probabilities for each prediction may be saved with the `probabilities` output parameter.

### Example
For example, to train a decision tree with a minimum leaf size of 20 on the dataset contained in `'data'` with labels `'labels'`, saving the output model to `'tree'` and printing the training error, one could call

```python
>>> output = decision_tree(training=data, labels=labels, minimum_leaf_size=20,
  minimum_gain_split=0.001, print_training_accuracy=True)
>>> tree = output['output_model']
```

Then, to use that model to classify points in `'test_set'` and print the test error given the labels `'test_labels'` using that model, while saving the predictions for each point to `'predictions'`, one could call 

```python
>>> output = decision_tree(input_model=tree, test=test_set,
  test_labels=test_labels)
>>> predictions = output['predictions']
```

### See also

 - [Random forest](#random_forest)
 - [Decision trees on Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning)
 - [Induction of Decision Trees (pdf)](https://www.hunch.net/~coms-4771/quinlan.pdf)
 - [DecisionTree C++ class documentation](../../user/methods/decision_tree.md)

## det()
{: #det }

#### Density Estimation With Density Estimation Trees
{: #det_descr }

```python
>>> from mlpack import det
>>> d = det(check_input_matrices=False, copy_all_inputs=False, folds=10,
        input_model=None, max_leaf_size=10, min_leaf_size=5, path_format='lr',
        skip_pruning=False, test=np.empty([0, 0]), training=np.empty([0, 0]),
        verbose=False)
>>> output_model = d['output_model']
>>> tag_counters_file = d['tag_counters_file']
>>> tag_file = d['tag_file']
>>> test_set_estimates = d['test_set_estimates']
>>> training_set_estimates = d['training_set_estimates']
>>> vi = d['vi']
```

An implementation of density estimation trees for the density estimation task.  Density estimation trees can be trained or used to predict the density at locations given by query points. [Detailed documentation](#det_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `folds` | [`int`](#doc_int) | The number of folds of cross-validation to perform for the estimation (0 is LOOCV) | `10` |
| `input_model` | [`DTree<>Type`](#doc_model) | Trained density estimation tree to load. | `None` |
| `max_leaf_size` | [`int`](#doc_int) | The maximum size of a leaf in the unpruned, fully grown DET. | `10` |
| `min_leaf_size` | [`int`](#doc_int) | The minimum size of a leaf in the unpruned, fully grown DET. | `5` |
| `path_format` | [`str`](#doc_str) | The format of path printing: 'lr', 'id-lr', or 'lr-id'. | `'lr'` |
| `skip_pruning` | [`bool`](#doc_bool) | Whether to bypass the pruning process and output the unpruned tree only. | `False` |
| `test` | [`matrix`](#doc_matrix) | A set of test points to estimate the density of. | `np.empty([0, 0])` |
| `training` | [`matrix`](#doc_matrix) | The data set on which to build a density estimation tree. | `np.empty([0, 0])` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output_model` | [`DTree<>Type`](#doc_model) | Output to save trained density estimation tree to. | 
| `tag_counters_file` | [`str`](#doc_str) | The file to output the number of points that went to each leaf. | 
| `tag_file` | [`str`](#doc_str) | The file to output the tags (and possibly paths) for each sample in the test set. | 
| `test_set_estimates` | [`matrix`](#doc_matrix) | The output estimates on the test set from the final optimally pruned tree. | 
| `training_set_estimates` | [`matrix`](#doc_matrix) | The output density estimates on the training set from the final optimally pruned tree. | 
| `vi` | [`matrix`](#doc_matrix) | The output variable importance values for each feature. | 

### Detailed documentation
{: #det_detailed-documentation }

This program performs a number of functions related to Density Estimation Trees.  The optimal Density Estimation Tree (DET) can be trained on a set of data (specified by `training`) using cross-validation (with number of folds specified with the `folds` parameter).  This trained density estimation tree may then be saved with the `output_model` output parameter.

The variable importances (that is, the feature importance values for each dimension) may be saved with the `vi` output parameter, and the density estimates for each training point may be saved with the `training_set_estimates` output parameter.

Enabling path printing for each node outputs the path from the root node to a leaf for each entry in the test set, or training set (if a test set is not provided).  Strings like 'LRLRLR' (indicating that traversal went to the left child, then the right child, then the left child, and so forth) will be output. If 'lr-id' or 'id-lr' are given as the `path_format` parameter, then the ID (tag) of every node along the path will be printed after or before the L or R character indicating the direction of traversal, respectively.

This program also can provide density estimates for a set of test points, specified in the `test` parameter.  The density estimation tree used for this task will be the tree that was trained on the given training points, or a tree given as the parameter `input_model`.  The density estimates for the test points may be saved using the `test_set_estimates` output parameter.

### See also

 - [Density estimation on Wikipedia](https://en.wikipedia.org/wiki/Density_estimation)
 - [Density estimation trees (pdf)](https://www.mlpack.org/papers/det.pdf)
 - [DTree class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/det/dtree.hpp)

## emst()
{: #emst }

#### Fast Euclidean Minimum Spanning Tree
{: #emst_descr }

```python
>>> from mlpack import emst
>>> d = emst(check_input_matrices=False, copy_all_inputs=False,
        input_=np.empty([0, 0]), leaf_size=1, naive=False, verbose=False)
>>> output = d['output']
```

An implementation of the Dual-Tree Boruvka algorithm for computing the Euclidean minimum spanning tree of a set of input points. [Detailed documentation](#emst_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `input_` | [`matrix`](#doc_matrix) | Input data matrix. | `**--**` |
| `leaf_size` | [`int`](#doc_int) | Leaf size in the kd-tree.  One-element leaves give the empirically best performance, but at the cost of greater memory requirements. | `1` |
| `naive` | [`bool`](#doc_bool) | Compute the MST using O(n^2) naive algorithm. | `False` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output` | [`matrix`](#doc_matrix) | Output data.  Stored as an edge list. | 

### Detailed documentation
{: #emst_detailed-documentation }

This program can compute the Euclidean minimum spanning tree of a set of input points using the dual-tree Boruvka algorithm.

The set to calculate the minimum spanning tree of is specified with the `input_` parameter, and the output may be saved with the `output` output parameter.

The `leaf_size` parameter controls the leaf size of the kd-tree that is used to calculate the minimum spanning tree, and if the `naive` option is given, then brute-force search is used (this is typically much slower in low dimensions).  The leaf size does not affect the results, but it may have some effect on the runtime of the algorithm.

### Example
For example, the minimum spanning tree of the input dataset `'data'` can be calculated with a leaf size of 20 and stored as `'spanning_tree'` using the following command:

```python
>>> output = emst(input_=data, leaf_size=20)
>>> spanning_tree = output['output']
```

The output matrix is a three-dimensional matrix, where each row indicates an edge.  The first dimension corresponds to the lesser index of the edge; the second dimension corresponds to the greater index of the edge; and the third column corresponds to the distance between the two points.

### See also

 - [Minimum spanning tree on Wikipedia](https://en.wikipedia.org/wiki/Minimum_spanning_tree)
 - [Fast Euclidean Minimum Spanning Tree: Algorithm, Analysis, and Applications (pdf)](https://www.mlpack.org/papers/emst.pdf)
 - [DualTreeBoruvka class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/emst/dtb.hpp)

## fastmks()
{: #fastmks }

#### FastMKS (Fast Max-Kernel Search)
{: #fastmks_descr }

```python
>>> from mlpack import fastmks
>>> d = fastmks(bandwidth=1, base=2, check_input_matrices=False,
        copy_all_inputs=False, degree=2, input_model=None, k=0, kernel='linear',
        naive=False, offset=0, query=np.empty([0, 0]), reference=np.empty([0,
        0]), scale=1, single=False, verbose=False)
>>> indices = d['indices']
>>> kernels = d['kernels']
>>> output_model = d['output_model']
```

An implementation of the single-tree and dual-tree fast max-kernel search (FastMKS) algorithm.  Given a set of reference points and a set of query points, this can find the reference point with maximum kernel value for each query point; trained models can be reused for future queries. [Detailed documentation](#fastmks_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `bandwidth` | [`float`](#doc_float) | Bandwidth (for Gaussian, Epanechnikov, and triangular kernels). | `1` |
| `base` | [`float`](#doc_float) | Base to use during cover tree construction. | `2` |
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `degree` | [`float`](#doc_float) | Degree of polynomial kernel. | `2` |
| `input_model` | [`FastMKSModelType`](#doc_model) | Input FastMKS model to use. | `None` |
| `k` | [`int`](#doc_int) | Number of maximum kernels to find. | `0` |
| `kernel` | [`str`](#doc_str) | Kernel type to use: 'linear', 'polynomial', 'cosine', 'gaussian', 'epanechnikov', 'triangular', 'hyptan'. | `'linear'` |
| `naive` | [`bool`](#doc_bool) | If true, O(n^2) naive mode is used for computation. | `False` |
| `offset` | [`float`](#doc_float) | Offset of kernel (for polynomial and hyptan kernels). | `0` |
| `query` | [`matrix`](#doc_matrix) | The query dataset. | `np.empty([0, 0])` |
| `reference` | [`matrix`](#doc_matrix) | The reference dataset. | `np.empty([0, 0])` |
| `scale` | [`float`](#doc_float) | Scale of kernel (for hyptan kernel). | `1` |
| `single` | [`bool`](#doc_bool) | If true, single-tree search is used (as opposed to dual-tree search. | `False` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `indices` | [`int matrix`](#doc_int_matrix) | Output matrix of indices. | 
| `kernels` | [`matrix`](#doc_matrix) | Output matrix of kernels. | 
| `output_model` | [`FastMKSModelType`](#doc_model) | Output for FastMKS model. | 

### Detailed documentation
{: #fastmks_detailed-documentation }

This program will find the k maximum kernels of a set of points, using a query set and a reference set (which can optionally be the same set). More specifically, for each point in the query set, the k points in the reference set with maximum kernel evaluations are found.  The kernel function used is specified with the `kernel` parameter.

### Example
For example, the following command will calculate, for each point in the query set `'query'`, the five points in the reference set `'reference'` with maximum kernel evaluation using the linear kernel.  The kernel evaluations may be saved with the  `'kernels'` output parameter and the indices may be saved with the `'indices'` output parameter.

```python
>>> output = fastmks(k=5, reference=reference, query=query, kernel='linear')
>>> indices = output['indices']
>>> kernels = output['kernels']
```

The output matrices are organized such that row i and column j in the indices matrix corresponds to the index of the point in the reference set that has j'th largest kernel evaluation with the point in the query set with index i.  Row i and column j in the kernels matrix corresponds to the kernel evaluation between those two points.

This program performs FastMKS using a cover tree.  The base used to build the cover tree can be specified with the `base` parameter.

### See also

 - [k-nearest-neighbor search](#knn)
 - [Dual-tree Fast Exact Max-Kernel Search (pdf)](https://mlpack.org/papers/fmks.pdf)
 - [FastMKS class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/fastmks/fastmks.hpp)

## gmm_train()
{: #gmm_train }

#### Gaussian Mixture Model (GMM) Training
{: #gmm_train_descr }

```python
>>> from mlpack import gmm_train
>>> d = gmm_train(check_input_matrices=False, copy_all_inputs=False,
        diagonal_covariance=False, gaussians=0, input_=np.empty([0, 0]),
        input_model=None, kmeans_max_iterations=1000, max_iterations=250,
        no_force_positive=False, noise=0, percentage=0.02, refined_start=False,
        samplings=100, seed=0, tolerance=1e-10, trials=1, verbose=False)
>>> output_model = d['output_model']
```

An implementation of the EM algorithm for training Gaussian mixture models (GMMs).  Given a dataset, this can train a GMM for future use with other tools. [Detailed documentation](#gmm_train_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `diagonal_covariance` | [`bool`](#doc_bool) | Force the covariance of the Gaussians to be diagonal.  This can accelerate training time significantly. | `False` |
| `gaussians` | [`int`](#doc_int) | Number of Gaussians in the GMM. | `**--**` |
| `input_` | [`matrix`](#doc_matrix) | The training data on which the model will be fit. | `**--**` |
| `input_model` | [`GMMType`](#doc_model) | Initial input GMM model to start training with. | `None` |
| `kmeans_max_iterations` | [`int`](#doc_int) | Maximum number of iterations for the k-means algorithm (used to initialize EM). | `1000` |
| `max_iterations` | [`int`](#doc_int) | Maximum number of iterations of EM algorithm (passing 0 will run until convergence). | `250` |
| `no_force_positive` | [`bool`](#doc_bool) | Do not force the covariance matrices to be positive definite. | `False` |
| `noise` | [`float`](#doc_float) | Variance of zero-mean Gaussian noise to add to data. | `0` |
| `percentage` | [`float`](#doc_float) | If using --refined_start, specify the percentage of the dataset used for each sampling (should be between 0.0 and 1.0). | `0.02` |
| `refined_start` | [`bool`](#doc_bool) | During the initialization, use refined initial positions for k-means clustering (Bradley and Fayyad, 1998). | `False` |
| `samplings` | [`int`](#doc_int) | If using --refined_start, specify the number of samplings used for initial points. | `100` |
| `seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `tolerance` | [`float`](#doc_float) | Tolerance for convergence of EM. | `1e-10` |
| `trials` | [`int`](#doc_int) | Number of trials to perform in training GMM. | `1` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output_model` | [`GMMType`](#doc_model) | Output for trained GMM model. | 

### Detailed documentation
{: #gmm_train_detailed-documentation }

This program takes a parametric estimate of a Gaussian mixture model (GMM) using the EM algorithm to find the maximum likelihood estimate.  The model may be saved and reused by other mlpack GMM tools.

The input data to train on must be specified with the `input_` parameter, and the number of Gaussians in the model must be specified with the `gaussians` parameter.  Optionally, many trials with different random initializations may be run, and the result with highest log-likelihood on the training data will be taken.  The number of trials to run is specified with the `trials` parameter.  By default, only one trial is run.

The tolerance for convergence and maximum number of iterations of the EM algorithm are specified with the `tolerance` and `max_iterations` parameters, respectively.  The GMM may be initialized for training with another model, specified with the `input_model` parameter. Otherwise, the model is initialized by running k-means on the data.  The k-means clustering initialization can be controlled with the `kmeans_max_iterations`, `refined_start`, `samplings`, and `percentage` parameters.  If `refined_start` is specified, then the Bradley-Fayyad refined start initialization will be used.  This can often lead to better clustering results.

The 'diagonal_covariance' flag will cause the learned covariances to be diagonal matrices.  This significantly simplifies the model itself and causes training to be faster, but restricts the ability to fit more complex GMMs.

If GMM training fails with an error indicating that a covariance matrix could not be inverted, make sure that the `no_force_positive` parameter is not specified.  Alternately, adding a small amount of Gaussian noise (using the `noise` parameter) to the entire dataset may help prevent Gaussians with zero variance in a particular dimension, which is usually the cause of non-invertible covariance matrices.

The `no_force_positive` parameter, if set, will avoid the checks after each iteration of the EM algorithm which ensure that the covariance matrices are positive definite.  Specifying the flag can cause faster runtime, but may also cause non-positive definite covariance matrices, which will cause the program to crash.

### Example
As an example, to train a 6-Gaussian GMM on the data in `'data'` with a maximum of 100 iterations of EM and 3 trials, saving the trained GMM to `'gmm'`, the following command can be used:

```python
>>> output = gmm_train(input_=data, gaussians=6, trials=3)
>>> gmm = output['output_model']
```

To re-train that GMM on another set of data `'data2'`, the following command may be used: 

```python
>>> output = gmm_train(input_model=gmm, input_=data2, gaussians=6)
>>> new_gmm = output['output_model']
```

### See also

 - [gmm_generate()](#gmm_generate)
 - [gmm_probability()](#gmm_probability)
 - [Gaussian Mixture Models on Wikipedia](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model)
 - [GMM class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/gmm/gmm.hpp)

## gmm_generate()
{: #gmm_generate }

#### GMM Sample Generator
{: #gmm_generate_descr }

```python
>>> from mlpack import gmm_generate
>>> d = gmm_generate(check_input_matrices=False, copy_all_inputs=False,
        input_model=None, samples=0, seed=0, verbose=False)
>>> output = d['output']
```

A sample generator for pre-trained GMMs.  Given a pre-trained GMM, this can sample new points randomly from that distribution. [Detailed documentation](#gmm_generate_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `input_model` | [`GMMType`](#doc_model) | Input GMM model to generate samples from. | `**--**` |
| `samples` | [`int`](#doc_int) | Number of samples to generate. | `**--**` |
| `seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output` | [`matrix`](#doc_matrix) | Matrix to save output samples in. | 

### Detailed documentation
{: #gmm_generate_detailed-documentation }

This program is able to generate samples from a pre-trained GMM (use gmm_train to train a GMM).  The pre-trained GMM must be specified with the `input_model` parameter.  The number of samples to generate is specified by the `samples` parameter.  Output samples may be saved with the `output` output parameter.

### Example
The following command can be used to generate 100 samples from the pre-trained GMM `'gmm'` and store those generated samples in `'samples'`:

```python
>>> output = gmm_generate(input_model=gmm, samples=100)
>>> samples = output['output']
```

### See also

 - [gmm_train()](#gmm_train)
 - [gmm_probability()](#gmm_probability)
 - [Gaussian Mixture Models on Wikipedia](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model)
 - [GMM class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/gmm/gmm.hpp)

## gmm_probability()
{: #gmm_probability }

#### GMM Probability Calculator
{: #gmm_probability_descr }

```python
>>> from mlpack import gmm_probability
>>> d = gmm_probability(check_input_matrices=False,
        copy_all_inputs=False, input_=np.empty([0, 0]), input_model=None,
        verbose=False)
>>> output = d['output']
```

A probability calculator for GMMs.  Given a pre-trained GMM and a set of points, this can compute the probability that each point is from the given GMM. [Detailed documentation](#gmm_probability_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `input_` | [`matrix`](#doc_matrix) | Input matrix to calculate probabilities of. | `**--**` |
| `input_model` | [`GMMType`](#doc_model) | Input GMM to use as model. | `**--**` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output` | [`matrix`](#doc_matrix) | Matrix to store calculated probabilities in. | 

### Detailed documentation
{: #gmm_probability_detailed-documentation }

This program calculates the probability that given points came from a given GMM (that is, P(X \| gmm)).  The GMM is specified with the `input_model` parameter, and the points are specified with the `input_` parameter.  The output probabilities may be saved via the `output` output parameter.

### Example
So, for example, to calculate the probabilities of each point in `'points'` coming from the pre-trained GMM `'gmm'`, while storing those probabilities in `'probs'`, the following command could be used:

```python
>>> output = gmm_probability(input_model=gmm, input_=points)
>>> probs = output['output']
```

### See also

 - [gmm_train()](#gmm_train)
 - [gmm_generate()](#gmm_generate)
 - [Gaussian Mixture Models on Wikipedia](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model)
 - [GMM class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/gmm/gmm.hpp)

## hmm_train()
{: #hmm_train }

#### Hidden Markov Model (HMM) Training
{: #hmm_train_descr }

```python
>>> from mlpack import hmm_train
>>> d = hmm_train(batch=False, check_input_matrices=False,
        copy_all_inputs=False, gaussians=0, input_file='', input_model=None,
        labels_file='', seed=0, states=0, tolerance=1e-05, type='gaussian',
        verbose=False)
>>> output_model = d['output_model']
```

An implementation of training algorithms for Hidden Markov Models (HMMs). Given labeled or unlabeled data, an HMM can be trained for further use with other mlpack HMM tools. [Detailed documentation](#hmm_train_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `batch` | [`bool`](#doc_bool) | If true, input_file (and if passed, labels_file) are expected to contain a list of files to use as input observation sequences (and label sequences). | `False` |
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `gaussians` | [`int`](#doc_int) | Number of gaussians in each GMM (necessary when type is 'gmm'). | `0` |
| `input_file` | [`str`](#doc_str) | File containing input observations. | `**--**` |
| `input_model` | [`HMMModelType`](#doc_model) | Pre-existing HMM model to initialize training with. | `None` |
| `labels_file` | [`str`](#doc_str) | Optional file of hidden states, used for labeled training. | `''` |
| `seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `states` | [`int`](#doc_int) | Number of hidden states in HMM (necessary, unless model_file is specified). | `0` |
| `tolerance` | [`float`](#doc_float) | Tolerance of the Baum-Welch algorithm. | `1e-05` |
| `type` | [`str`](#doc_str) | Type of HMM: discrete \| gaussian \| diag_gmm \| gmm. | `'gaussian'` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output_model` | [`HMMModelType`](#doc_model) | Output for trained HMM. | 

### Detailed documentation
{: #hmm_train_detailed-documentation }

This program allows a Hidden Markov Model to be trained on labeled or unlabeled data.  It supports four types of HMMs: Discrete HMMs, Gaussian HMMs, GMM HMMs, or Diagonal GMM HMMs

Either one input sequence can be specified (with `input_file`), or, a file containing files in which input sequences can be found (when `input_file`and`batch` are used together).  In addition, labels can be provided in the file specified by `labels_file`, and if `batch` is used, the file given to `labels_file` should contain a list of files of labels corresponding to the sequences in the file given to `input_file`.

The HMM is trained with the Baum-Welch algorithm if no labels are provided.  The tolerance of the Baum-Welch algorithm can be set with the `tolerance`option.  By default, the transition matrix is randomly initialized and the emission distributions are initialized to fit the extent of the data.

Optionally, a pre-created HMM model can be used as a guess for the transition matrix and emission probabilities; this is specifiable with `output_model`.

### See also

 - [hmm_generate()](#hmm_generate)
 - [hmm_loglik()](#hmm_loglik)
 - [hmm_viterbi()](#hmm_viterbi)
 - [Hidden Mixture Models on Wikipedia](https://en.wikipedia.org/wiki/Hidden_Markov_model)
 - [HMM class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/hmm/hmm.hpp)

## hmm_generate()
{: #hmm_generate }

#### Hidden Markov Model (HMM) Sequence Generator
{: #hmm_generate_descr }

```python
>>> from mlpack import hmm_generate
>>> d = hmm_generate(check_input_matrices=False, copy_all_inputs=False,
        length=0, model=None, seed=0, start_state=0, verbose=False)
>>> output = d['output']
>>> state = d['state']
```

A utility to generate random sequences from a pre-trained Hidden Markov Model (HMM).  The length of the desired sequence can be specified, and a random sequence of observations is returned. [Detailed documentation](#hmm_generate_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `length` | [`int`](#doc_int) | Length of sequence to generate. | `**--**` |
| `model` | [`HMMModelType`](#doc_model) | Trained HMM to generate sequences with. | `**--**` |
| `seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `start_state` | [`int`](#doc_int) | Starting state of sequence. | `0` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output` | [`matrix`](#doc_matrix) | Matrix to save observation sequence to. | 
| `state` | [`int matrix`](#doc_int_matrix) | Matrix to save hidden state sequence to. | 

### Detailed documentation
{: #hmm_generate_detailed-documentation }

This utility takes an already-trained HMM, specified as the `model` parameter, and generates a random observation sequence and hidden state sequence based on its parameters. The observation sequence may be saved with the `output` output parameter, and the internal state  sequence may be saved with the `state` output parameter.

The state to start the sequence in may be specified with the `start_state` parameter.

### Example
For example, to generate a sequence of length 150 from the HMM `'hmm'` and save the observation sequence to `'observations'` and the hidden state sequence to `'states'`, the following command may be used: 

```python
>>> output = hmm_generate(model=hmm, length=150)
>>> observations = output['output']
>>> states = output['state']
```

### See also

 - [hmm_train()](#hmm_train)
 - [hmm_loglik()](#hmm_loglik)
 - [hmm_viterbi()](#hmm_viterbi)
 - [Hidden Mixture Models on Wikipedia](https://en.wikipedia.org/wiki/Hidden_Markov_model)
 - [HMM class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/hmm/hmm.hpp)

## hmm_loglik()
{: #hmm_loglik }

#### Hidden Markov Model (HMM) Sequence Log-Likelihood
{: #hmm_loglik_descr }

```python
>>> from mlpack import hmm_loglik
>>> d = hmm_loglik(check_input_matrices=False, copy_all_inputs=False,
        input_=np.empty([0, 0]), input_model=None, verbose=False)
>>> log_likelihood = d['log_likelihood']
```

A utility for computing the log-likelihood of a sequence for Hidden Markov Models (HMMs).  Given a pre-trained HMM and an observation sequence, this computes and returns the log-likelihood of that sequence being observed from that HMM. [Detailed documentation](#hmm_loglik_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `input_` | [`matrix`](#doc_matrix) | File containing observations, | `**--**` |
| `input_model` | [`HMMModelType`](#doc_model) | File containing HMM. | `**--**` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `log_likelihood` | [`float`](#doc_float) | Log-likelihood of the sequence. | 

### Detailed documentation
{: #hmm_loglik_detailed-documentation }

This utility takes an already-trained HMM, specified with the `input_model` parameter, and evaluates the log-likelihood of a sequence of observations, given with the `input_` parameter.  The computed log-likelihood is given as output.

### Example
For example, to compute the log-likelihood of the sequence `'seq'` with the pre-trained HMM `'hmm'`, the following command may be used: 

```python
>>> hmm_loglik(input_=seq, input_model=hmm)
```

### See also

 - [hmm_train()](#hmm_train)
 - [hmm_generate()](#hmm_generate)
 - [hmm_viterbi()](#hmm_viterbi)
 - [Hidden Mixture Models on Wikipedia](https://en.wikipedia.org/wiki/Hidden_Markov_model)
 - [HMM class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/hmm/hmm.hpp)

## hmm_viterbi()
{: #hmm_viterbi }

#### Hidden Markov Model (HMM) Viterbi State Prediction
{: #hmm_viterbi_descr }

```python
>>> from mlpack import hmm_viterbi
>>> d = hmm_viterbi(check_input_matrices=False, copy_all_inputs=False,
        input_=np.empty([0, 0]), input_model=None, verbose=False)
>>> output = d['output']
```

A utility for computing the most probable hidden state sequence for Hidden Markov Models (HMMs).  Given a pre-trained HMM and an observed sequence, this uses the Viterbi algorithm to compute and return the most probable hidden state sequence. [Detailed documentation](#hmm_viterbi_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `input_` | [`matrix`](#doc_matrix) | Matrix containing observations, | `**--**` |
| `input_model` | [`HMMModelType`](#doc_model) | Trained HMM to use. | `**--**` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output` | [`int matrix`](#doc_int_matrix) | File to save predicted state sequence to. | 

### Detailed documentation
{: #hmm_viterbi_detailed-documentation }

This utility takes an already-trained HMM, specified as `input_model`, and evaluates the most probable hidden state sequence of a given sequence of observations (specified as '`input_`, using the Viterbi algorithm.  The computed state sequence may be saved using the `output` output parameter.

### Example
For example, to predict the state sequence of the observations `'obs'` using the HMM `'hmm'`, storing the predicted state sequence to `'states'`, the following command could be used:

```python
>>> output = hmm_viterbi(input_=obs, input_model=hmm)
>>> states = output['output']
```

### See also

 - [hmm_train()](#hmm_train)
 - [hmm_generate()](#hmm_generate)
 - [hmm_loglik()](#hmm_loglik)
 - [Hidden Mixture Models on Wikipedia](https://en.wikipedia.org/wiki/Hidden_Markov_model)
 - [HMM class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/hmm/hmm.hpp)

## hoeffding_tree()
{: #hoeffding_tree }

#### Hoeffding trees
{: #hoeffding_tree_descr }

```python
>>> from mlpack import hoeffding_tree
>>> d = hoeffding_tree(batch_mode=False, bins=10,
        check_input_matrices=False, confidence=0.95, copy_all_inputs=False,
        info_gain=False, input_model=None, labels=np.empty([0],
        dtype=np.uint64), max_samples=5000, min_samples=100,
        numeric_split_strategy='binary', observations_before_binning=100,
        passes=1, test=np.empty([0, 0]), test_labels=np.empty([0],
        dtype=np.uint64), training=np.empty([0, 0]), verbose=False)
>>> output_model = d['output_model']
>>> predictions = d['predictions']
>>> probabilities = d['probabilities']
```

An implementation of Hoeffding trees, a form of streaming decision tree for classification.  Given labeled data, a Hoeffding tree can be trained and saved for later use, or a pre-trained Hoeffding tree can be used for predicting the classifications of new points. [Detailed documentation](#hoeffding_tree_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `batch_mode` | [`bool`](#doc_bool) | If true, samples will be considered in batch instead of as a stream.  This generally results in better trees but at the cost of memory usage and runtime. | `False` |
| `bins` | [`int`](#doc_int) | If the 'domingos' split strategy is used, this specifies the number of bins for each numeric split. | `10` |
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `confidence` | [`float`](#doc_float) | Confidence before splitting (between 0 and 1). | `0.95` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `info_gain` | [`bool`](#doc_bool) | If set, information gain is used instead of Gini impurity for calculating Hoeffding bounds. | `False` |
| `input_model` | [`HoeffdingTreeModelType`](#doc_model) | Input trained Hoeffding tree model. | `None` |
| `labels` | [`int vector`](#doc_int_vector) | Labels for training dataset. | `np.empty([0], dtype=np.uint64)` |
| `max_samples` | [`int`](#doc_int) | Maximum number of samples before splitting. | `5000` |
| `min_samples` | [`int`](#doc_int) | Minimum number of samples before splitting. | `100` |
| `numeric_split_strategy` | [`str`](#doc_str) | The splitting strategy to use for numeric features: 'domingos' or 'binary'. | `'binary'` |
| `observations_before_binning` | [`int`](#doc_int) | If the 'domingos' split strategy is used, this specifies the number of samples observed before binning is performed. | `100` |
| `passes` | [`int`](#doc_int) | Number of passes to take over the dataset. | `1` |
| `test` | [`categorical matrix`](#doc_categorical_matrix) | Testing dataset (may be categorical). | `np.empty([0, 0])` |
| `test_labels` | [`int vector`](#doc_int_vector) | Labels of test data. | `np.empty([0], dtype=np.uint64)` |
| `training` | [`categorical matrix`](#doc_categorical_matrix) | Training dataset (may be categorical). | `np.empty([0, 0])` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output_model` | [`HoeffdingTreeModelType`](#doc_model) | Output for trained Hoeffding tree model. | 
| `predictions` | [`int vector`](#doc_int_vector) | Matrix to output label predictions for test data into. | 
| `probabilities` | [`matrix`](#doc_matrix) | In addition to predicting labels, provide rediction probabilities in this matrix. | 

### Detailed documentation
{: #hoeffding_tree_detailed-documentation }

This program implements Hoeffding trees, a form of streaming decision tree suited best for large (or streaming) datasets.  This program supports both categorical and numeric data.  Given an input dataset, this program is able to train the tree with numerous training options, and save the model to a file.  The program is also able to use a trained model or a model from file in order to predict classes for a given test set.

The training file and associated labels are specified with the `training` and `labels` parameters, respectively. Optionally, if `labels` is not specified, the labels are assumed to be the last dimension of the training dataset.

The training may be performed in batch mode (like a typical decision tree algorithm) by specifying the `batch_mode` option, but this may not be the best option for large datasets.

When a model is trained, it may be saved via the `output_model` output parameter.  A model may be loaded from file for further training or testing with the `input_model` parameter.

Test data may be specified with the `test` parameter, and if performance statistics are desired for that test set, labels may be specified with the `test_labels` parameter.  Predictions for each test point may be saved with the `predictions` output parameter, and class probabilities for each prediction may be saved with the `probabilities` output parameter.

### Example
For example, to train a Hoeffding tree with confidence 0.99 with data `'dataset'`, saving the trained tree to `'tree'`, the following command may be used:

```python
>>> output = hoeffding_tree(training=dataset, confidence=0.99)
>>> tree = output['output_model']
```

Then, this tree may be used to make predictions on the test set `'test_set'`, saving the predictions into `'predictions'` and the class probabilities into `'class_probs'` with the following command: 

```python
>>> output = hoeffding_tree(input_model=tree, test=test_set)
>>> predictions = output['predictions']
>>> class_probs = output['probabilities']
```

### See also

 - [decision_tree()](#decision_tree)
 - [random_forest()](#random_forest)
 - [Mining High-Speed Data Streams (pdf)](http://dm.cs.washington.edu/papers/vfdt-kdd00.pdf)
 - [HoeffdingTree class documentation](../../user/methods/hoeffding_tree.md)

## kde()
{: #kde }

#### Kernel Density Estimation
{: #kde_descr }

```python
>>> from mlpack import kde
>>> d = kde(abs_error=0, algorithm='dual-tree', bandwidth=1,
        check_input_matrices=False, copy_all_inputs=False,
        initial_sample_size=100, input_model=None, kernel='gaussian',
        mc_break_coef=0.4, mc_entry_coef=3, mc_probability=0.95,
        monte_carlo=False, query=np.empty([0, 0]), reference=np.empty([0, 0]),
        rel_error=0.05, tree='kd-tree', verbose=False)
>>> output_model = d['output_model']
>>> predictions = d['predictions']
```

An implementation of kernel density estimation with dual-tree algorithms. Given a set of reference points and query points and a kernel function, this can estimate the density function at the location of each query point using trees; trees that are built can be saved for later use. [Detailed documentation](#kde_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `abs_error` | [`float`](#doc_float) | Relative error tolerance for the prediction. | `0` |
| `algorithm` | [`str`](#doc_str) | Algorithm to use for the prediction.('dual-tree', 'single-tree'). | `'dual-tree'` |
| `bandwidth` | [`float`](#doc_float) | Bandwidth of the kernel. | `1` |
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `initial_sample_size` | [`int`](#doc_int) | Initial sample size for Monte Carlo estimations. | `100` |
| `input_model` | [`KDEModelType`](#doc_model) | Contains pre-trained KDE model. | `None` |
| `kernel` | [`str`](#doc_str) | Kernel to use for the prediction.('gaussian', 'epanechnikov', 'laplacian', 'spherical', 'triangular'). | `'gaussian'` |
| `mc_break_coef` | [`float`](#doc_float) | Controls what fraction of the amount of node's descendants is the limit for the sample size before it recurses. | `0.4` |
| `mc_entry_coef` | [`float`](#doc_float) | Controls how much larger does the amount of node descendants has to be compared to the initial sample size in order to be a candidate for Monte Carlo estimations. | `3` |
| `mc_probability` | [`float`](#doc_float) | Probability of the estimation being bounded by relative error when using Monte Carlo estimations. | `0.95` |
| `monte_carlo` | [`bool`](#doc_bool) | Whether to use Monte Carlo estimations when possible. | `False` |
| `query` | [`matrix`](#doc_matrix) | Query dataset to KDE on. | `np.empty([0, 0])` |
| `reference` | [`matrix`](#doc_matrix) | Input reference dataset use for KDE. | `np.empty([0, 0])` |
| `rel_error` | [`float`](#doc_float) | Relative error tolerance for the prediction. | `0.05` |
| `tree` | [`str`](#doc_str) | Tree to use for the prediction.('kd-tree', 'ball-tree', 'cover-tree', 'octree', 'r-tree'). | `'kd-tree'` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output_model` | [`KDEModelType`](#doc_model) | If specified, the KDE model will be saved here. | 
| `predictions` | [`vector`](#doc_vector) | Vector to store density predictions. | 

### Detailed documentation
{: #kde_detailed-documentation }

This program performs a Kernel Density Estimation. KDE is a non-parametric way of estimating probability density function. For each query point the program will estimate its probability density by applying a kernel function to each reference point. The computational complexity of this is O(N^2) where there are N query points and N reference points, but this implementation will typically see better performance as it uses an approximate dual or single tree algorithm for acceleration.

Dual or single tree optimization avoids many barely relevant calculations (as kernel function values decrease with distance), so it is an approximate computation. You can specify the maximum relative error tolerance for each query value with `rel_error` as well as the maximum absolute error tolerance with the parameter `abs_error`. This program runs using an Euclidean metric. Kernel function can be selected using the `kernel` option. You can also choose what which type of tree to use for the dual-tree algorithm with `tree`. It is also possible to select whether to use dual-tree algorithm or single-tree algorithm using the `algorithm` option.

Monte Carlo estimations can be used to accelerate the KDE estimate when the Gaussian Kernel is used. This provides a probabilistic guarantee on the the error of the resulting KDE instead of an absolute guarantee.To enable Monte Carlo estimations, the `monte_carlo` flag can be used, and success probability can be set with the `mc_probability` option. It is possible to set the initial sample size for the Monte Carlo estimation using `initial_sample_size`. This implementation will only consider a node, as a candidate for the Monte Carlo estimation, if its number of descendant nodes is bigger than the initial sample size. This can be controlled using a coefficient that will multiply the initial sample size and can be set using `mc_entry_coef`. To avoid using the same amount of computations an exact approach would take, this program recurses the tree whenever a fraction of the amount of the node's descendant points have already been computed. This fraction is set using `mc_break_coef`.

### Example
For example, the following will run KDE using the data in `'ref_data'` for training and the data in `'qu_data'` as query data. It will apply an Epanechnikov kernel with a 0.2 bandwidth to each reference point and use a KD-Tree for the dual-tree optimization. The returned predictions will be within 5% of the real KDE value for each query point.

```python
>>> output = kde(reference=ref_data, query=qu_data, bandwidth=0.2,
  kernel='epanechnikov', tree='kd-tree', rel_error=0.05)
>>> out_data = output['predictions']
```

the predicted density estimations will be stored in `'out_data'`.
If no `query` is provided, then KDE will be computed on the `reference` dataset.
It is possible to select either a reference dataset or an input model but not both at the same time. If an input model is selected and parameter values are not set (e.g. `bandwidth`) then default parameter values will be used.

In addition to the last program call, it is also possible to activate Monte Carlo estimations if a Gaussian kernel is used. This can provide faster results, but the KDE will only have a probabilistic guarantee of meeting the desired error bound (instead of an absolute guarantee). The following example will run KDE using a Monte Carlo estimation when possible. The results will be within a 5% of the real KDE value with a 95% probability. Initial sample size for the Monte Carlo estimation will be 200 points and a node will be a candidate for the estimation only when it contains 700 (i.e. 3.5*200) points. If a node contains 700 points and 420 (i.e. 0.6*700) have already been sampled, then the algorithm will recurse instead of keep sampling.

```python
>>> output = kde(reference=ref_data, query=qu_data, bandwidth=0.2,
  kernel='gaussian', tree='kd-tree', rel_error=0.05, monte_carlo=,
  mc_probability=0.95, initial_sample_size=200, mc_entry_coef=3.5,
  mc_break_coef=0.6)
>>> out_data = output['predictions']
```

### See also

 - [knn()](#knn)
 - [Kernel density estimation on Wikipedia](https://en.wikipedia.org/wiki/Kernel_density_estimation)
 - [Tree-Independent Dual-Tree Algorithms](https://arxiv.org/pdf/1304.4327)
 - [Fast High-dimensional Kernel Summations Using the Monte Carlo Multipole Method](https://proceedings.neurips.cc/paper_files/paper/2008/file/39059724f73a9969845dfe4146c5660e-Paper.pdf)
 - [KDE C++ class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/kde/kde.hpp)

## kernel_pca()
{: #kernel_pca }

#### Kernel Principal Components Analysis
{: #kernel_pca_descr }

```python
>>> from mlpack import kernel_pca
>>> d = kernel_pca(bandwidth=1, center=False,
        check_input_matrices=False, copy_all_inputs=False, degree=1,
        input_=np.empty([0, 0]), kernel='', kernel_scale=1,
        new_dimensionality=0, nystroem_method=False, offset=0,
        sampling='kmeans', verbose=False)
>>> output = d['output']
```

An implementation of Kernel Principal Components Analysis (KPCA).  This can be used to perform nonlinear dimensionality reduction or preprocessing on a given dataset. [Detailed documentation](#kernel_pca_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `bandwidth` | [`float`](#doc_float) | Bandwidth, for 'gaussian' and 'laplacian' kernels. | `1` |
| `center` | [`bool`](#doc_bool) | If set, the transformed data will be centered about the origin. | `False` |
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `degree` | [`float`](#doc_float) | Degree of polynomial, for 'polynomial' kernel. | `1` |
| `input_` | [`matrix`](#doc_matrix) | Input dataset to perform KPCA on. | `**--**` |
| `kernel` | [`str`](#doc_str) | The kernel to use; see the above documentation for the list of usable kernels. | `**--**` |
| `kernel_scale` | [`float`](#doc_float) | Scale, for 'hyptan' kernel. | `1` |
| `new_dimensionality` | [`int`](#doc_int) | If not 0, reduce the dimensionality of the output dataset by ignoring the dimensions with the smallest eigenvalues. | `0` |
| `nystroem_method` | [`bool`](#doc_bool) | If set, the Nystroem method will be used. | `False` |
| `offset` | [`float`](#doc_float) | Offset, for 'hyptan' and 'polynomial' kernels. | `0` |
| `sampling` | [`str`](#doc_str) | Sampling scheme to use for the Nystroem method: 'kmeans', 'random', 'ordered' | `'kmeans'` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output` | [`matrix`](#doc_matrix) | Matrix to save modified dataset to. | 

### Detailed documentation
{: #kernel_pca_detailed-documentation }

This program performs Kernel Principal Components Analysis (KPCA) on the specified dataset with the specified kernel.  This will transform the data onto the kernel principal components, and optionally reduce the dimensionality by ignoring the kernel principal components with the smallest eigenvalues.

For the case where a linear kernel is used, this reduces to regular PCA.

The kernels that are supported are listed below:

 * 'linear': the standard linear dot product (same as normal PCA):
    `K(x, y) = x^T y`

 * 'gaussian': a Gaussian kernel; requires bandwidth:
    `K(x, y) = exp(-(\|\| x - y \|\| ^ 2) / (2 * (bandwidth ^ 2)))`

 * 'polynomial': polynomial kernel; requires offset and degree:
    `K(x, y) = (x^T y + offset) ^ degree`

 * 'hyptan': hyperbolic tangent kernel; requires scale and offset:
    `K(x, y) = tanh(scale * (x^T y) + offset)`

 * 'laplacian': Laplacian kernel; requires bandwidth:
    `K(x, y) = exp(-(\|\| x - y \|\|) / bandwidth)`

 * 'epanechnikov': Epanechnikov kernel; requires bandwidth:
    `K(x, y) = max(0, 1 - \|\| x - y \|\|^2 / bandwidth^2)`

 * 'cosine': cosine distance:
    `K(x, y) = 1 - (x^T y) / (\|\| x \|\| * \|\| y \|\|)`

The parameters for each of the kernels should be specified with the options `bandwidth`, `kernel_scale`, `offset`, or `degree` (or a combination of those parameters).

Optionally, the Nystroem method ("Using the Nystroem method to speed up kernel machines", 2001) can be used to calculate the kernel matrix by specifying the `nystroem_method` parameter. This approach works by using a subset of the data as basis to reconstruct the kernel matrix; to specify the sampling scheme, the `sampling` parameter is used.  The sampling scheme for the Nystroem method can be chosen from the following list: 'kmeans', 'random', 'ordered'.

### Example
For example, the following command will perform KPCA on the dataset `'input'` using the Gaussian kernel, and saving the transformed data to `'transformed'`: 

```python
>>> output = kernel_pca(input_=input, kernel='gaussian')
>>> transformed = output['output']
```

### See also

 - [Kernel principal component analysis on Wikipedia](https://en.wikipedia.org/wiki/Kernel_principal_component_analysis)
 - [Nonlinear Component Analysis as a Kernel Eigenvalue Problem](https://www.mlpack.org/papers/kpca.pdf)
 - [KernelPCA class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/kernel_pca/kernel_pca.hpp)

## kmeans()
{: #kmeans }

#### K-Means Clustering
{: #kmeans_descr }

```python
>>> from mlpack import kmeans
>>> d = kmeans(algorithm='naive', allow_empty_clusters=False,
        check_input_matrices=False, clusters=0, copy_all_inputs=False,
        in_place=False, initial_centroids=np.empty([0, 0]), input_=np.empty([0,
        0]), kill_empty_clusters=False, kmeans_plus_plus=False,
        labels_only=False, max_iterations=1000, percentage=0.02,
        refined_start=False, samplings=100, seed=0, verbose=False)
>>> centroid = d['centroid']
>>> output = d['output']
```

An implementation of several strategies for efficient k-means clustering. Given a dataset and a value of k, this computes and returns a k-means clustering on that data. [Detailed documentation](#kmeans_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `algorithm` | [`str`](#doc_str) | Algorithm to use for the Lloyd iteration ('naive', 'pelleg-moore', 'elkan', 'hamerly', 'dualtree', or 'dualtree-covertree'). | `'naive'` |
| `allow_empty_clusters` | [`bool`](#doc_bool) | Allow empty clusters to be persist. | `False` |
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `clusters` | [`int`](#doc_int) | Number of clusters to find (0 autodetects from initial centroids). | `**--**` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `in_place` | [`bool`](#doc_bool) | If specified, a column containing the learned cluster assignments will be added to the input dataset file.  In this case, --output_file is overridden. (Do not use in Python.) | `False` |
| `initial_centroids` | [`matrix`](#doc_matrix) | Start with the specified initial centroids. | `np.empty([0, 0])` |
| `input_` | [`matrix`](#doc_matrix) | Input dataset to perform clustering on. | `**--**` |
| `kill_empty_clusters` | [`bool`](#doc_bool) | Remove empty clusters when they occur. | `False` |
| `kmeans_plus_plus` | [`bool`](#doc_bool) | Use the k-means++ initialization strategy to choose initial points. | `False` |
| `labels_only` | [`bool`](#doc_bool) | Only output labels into output file. | `False` |
| `max_iterations` | [`int`](#doc_int) | Maximum number of iterations before k-means terminates. | `1000` |
| `percentage` | [`float`](#doc_float) | Percentage of dataset to use for each refined start sampling (use when --refined_start is specified). | `0.02` |
| `refined_start` | [`bool`](#doc_bool) | Use the refined initial point strategy by Bradley and Fayyad to choose initial points. | `False` |
| `samplings` | [`int`](#doc_int) | Number of samplings to perform for refined start (use when --refined_start is specified). | `100` |
| `seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `centroid` | [`matrix`](#doc_matrix) | If specified, the centroids of each cluster will  be written to the given file. | 
| `output` | [`matrix`](#doc_matrix) | Matrix to store output labels or labeled data to. | 

### Detailed documentation
{: #kmeans_detailed-documentation }

This program performs K-Means clustering on the given dataset.  It can return the learned cluster assignments, and the centroids of the clusters.  Empty clusters are not allowed by default; when a cluster becomes empty, the point furthest from the centroid of the cluster with maximum variance is taken to fill that cluster.

Optionally, the strategy to choose initial centroids can be specified.  The k-means++ algorithm can be used to choose initial centroids with the `kmeans_plus_plus` parameter.  The Bradley and Fayyad approach ("Refining initial points for k-means clustering", 1998) can be used to select initial points by specifying the `refined_start` parameter.  This approach works by taking random samplings of the dataset; to specify the number of samplings, the `samplings` parameter is used, and to specify the percentage of the dataset to be used in each sample, the `percentage` parameter is used (it should be a value between 0.0 and 1.0).

There are several options available for the algorithm used for each Lloyd iteration, specified with the `algorithm`  option.  The standard O(kN) approach can be used ('naive').  Other options include the Pelleg-Moore tree-based algorithm ('pelleg-moore'), Elkan's triangle-inequality based algorithm ('elkan'), Hamerly's modification to Elkan's algorithm ('hamerly'), the dual-tree k-means algorithm ('dualtree'), and the dual-tree k-means algorithm using the cover tree ('dualtree-covertree').

The behavior for when an empty cluster is encountered can be modified with the `allow_empty_clusters` option.  When this option is specified and there is a cluster owning no points at the end of an iteration, that cluster's centroid will simply remain in its position from the previous iteration. If the `kill_empty_clusters` option is specified, then when a cluster owns no points at the end of an iteration, the cluster centroid is simply filled with DBL_MAX, killing it and effectively reducing k for the rest of the computation.  Note that the default option when neither empty cluster option is specified can be time-consuming to calculate; therefore, specifying either of these parameters will often accelerate runtime.

Initial clustering assignments may be specified using the `initial_centroids` parameter, and the maximum number of iterations may be specified with the `max_iterations` parameter.

### Example
As an example, to use Hamerly's algorithm to perform k-means clustering with k=10 on the dataset `'data'`, saving the centroids to `'centroids'` and the assignments for each point to `'assignments'`, the following command could be used:

```python
>>> output = kmeans(input_=data, clusters=10)
>>> assignments = output['output']
>>> centroids = output['centroid']
```

To run k-means on that same dataset with initial centroids specified in `'initial'` with a maximum of 500 iterations, storing the output centroids in `'final'` the following command may be used:

```python
>>> output = kmeans(input_=data, initial_centroids=initial, clusters=10,
  max_iterations=500)
>>> final = output['centroid']
```

### See also

 - [dbscan()](#dbscan)
 - [k-means++](https://en.wikipedia.org/wiki/K-means%2B%2B)
 - [Using the triangle inequality to accelerate k-means (pdf)](https://cdn.aaai.org/ICML/2003/ICML03-022.pdf)
 - [Making k-means even faster (pdf)](https://www.cse.iitd.ac.in/~rjaiswal/2015/col870/Project/Faster-k-means/Hamerly.pdf)
 - [Accelerating exact k-means algorithms with geometric reasoning (pdf)](http://reports-archive.adm.cs.cmu.edu/anon/anon/usr/ftp/usr0/ftp/2000/CMU-CS-00-105.pdf)
 - [A dual-tree algorithm for fast k-means clustering with large k (pdf)](http://www.ratml.org/pub/pdf/2017dual.pdf)
 - [KMeans class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/kmeans/kmeans.hpp)

## lars()
{: #lars }

#### LARS
{: #lars_descr }

```python
>>> from mlpack import lars
>>> d = lars(check_input_matrices=False, copy_all_inputs=False,
        input_=np.empty([0, 0]), input_model=None, lambda1=0, lambda2=0,
        no_intercept=False, no_normalize=False, responses=np.empty([0, 0]),
        test=np.empty([0, 0]), use_cholesky=False, verbose=False)
>>> output_model = d['output_model']
>>> output_predictions = d['output_predictions']
```

An implementation of Least Angle Regression (Stagewise/laSso), also known as LARS.  This can train a LARS/LASSO/Elastic Net model and use that model or a pre-trained model to output regression predictions for a test set. [Detailed documentation](#lars_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `input_` | [`matrix`](#doc_matrix) | Matrix of covariates (X). | `np.empty([0, 0])` |
| `input_model` | [`LARS<>Type`](#doc_model) | Trained LARS model to use. | `None` |
| `lambda1` | [`float`](#doc_float) | Regularization parameter for l1-norm penalty. | `0` |
| `lambda2` | [`float`](#doc_float) | Regularization parameter for l2-norm penalty. | `0` |
| `no_intercept` | [`bool`](#doc_bool) | Do not fit an intercept in the model. | `False` |
| `no_normalize` | [`bool`](#doc_bool) | Do not normalize data to unit variance before modeling. | `False` |
| `responses` | [`matrix`](#doc_matrix) | Matrix of responses/observations (y). | `np.empty([0, 0])` |
| `test` | [`matrix`](#doc_matrix) | Matrix containing points to regress on (test points). | `np.empty([0, 0])` |
| `use_cholesky` | [`bool`](#doc_bool) | Use Cholesky decomposition during computation rather than explicitly computing the full Gram matrix. | `False` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output_model` | [`LARS<>Type`](#doc_model) | Output LARS model. | 
| `output_predictions` | [`matrix`](#doc_matrix) | If --test_file is specified, this file is where the predicted responses will be saved. | 

### Detailed documentation
{: #lars_detailed-documentation }

An implementation of LARS: Least Angle Regression (Stagewise/laSso).  This is a stage-wise homotopy-based algorithm for L1-regularized linear regression (LASSO) and L1+L2-regularized linear regression (Elastic Net).

This program is able to train a LARS/LASSO/Elastic Net model or load a model from file, output regression predictions for a test set, and save the trained model to a file.  The LARS algorithm is described in more detail below:

Let X be a matrix where each row is a point and each column is a dimension, and let y be a vector of targets.

The Elastic Net problem is to solve

  min_beta 0.5 \|\| X * beta - y \|\|_2^2 + lambda_1 \|\|beta\|\|_1 +
      0.5 lambda_2 \|\|beta\|\|_2^2

If lambda1 > 0 and lambda2 = 0, the problem is the LASSO.
If lambda1 > 0 and lambda2 > 0, the problem is the Elastic Net.
If lambda1 = 0 and lambda2 > 0, the problem is ridge regression.
If lambda1 = 0 and lambda2 = 0, the problem is unregularized linear regression.

For efficiency reasons, it is not recommended to use this algorithm with `lambda1` = 0.  In that case, use the 'linear_regression' program, which implements both unregularized linear regression and ridge regression.

To train a LARS/LASSO/Elastic Net model, the `input_` and `responses` parameters must be given.  The `lambda1`, `lambda2`, and `use_cholesky` parameters control the training options.  A trained model can be saved with the `output_model`.  If no training is desired at all, a model can be passed via the `input_model` parameter.

The program can also provide predictions for test data using either the trained model or the given input model.  Test points can be specified with the `test` parameter.  Predicted responses to the test points can be saved with the `output_predictions` output parameter.

### Example
For example, the following command trains a model on the data `'data'` and responses `'responses'` with lambda1 set to 0.4 and lambda2 set to 0 (so, LASSO is being solved), and then the model is saved to `'lasso_model'`:

```python
>>> output = lars(input_=data, responses=responses, lambda1=0.4, lambda2=0)
>>> lasso_model = output['output_model']
```

The following command uses the `'lasso_model'` to provide predicted responses for the data `'test'` and save those responses to `'test_predictions'`: 

```python
>>> output = lars(input_model=lasso_model, test=test)
>>> test_predictions = output['output_predictions']
```

### See also

 - [linear_regression()](#linear_regression)
 - [Least angle regression (pdf)](https://mlpack.org/papers/lars.pdf)
 - [LARS C++ class documentation](../../user/methods/lars.md)

## linear_svm()
{: #linear_svm }

#### Linear SVM is an L2-regularized support vector machine.
{: #linear_svm_descr }

```python
>>> from mlpack import linear_svm
>>> d = linear_svm(check_input_matrices=False, copy_all_inputs=False,
        delta=1, epochs=50, input_model=None, labels=np.empty([0],
        dtype=np.uint64), lambda_=0.0001, max_iterations=10000,
        no_intercept=False, num_classes=0, optimizer='lbfgs', seed=0,
        shuffle=False, step_size=0.01, test=np.empty([0, 0]),
        test_labels=np.empty([0], dtype=np.uint64), tolerance=1e-10,
        training=np.empty([0, 0]), verbose=False)
>>> output_model = d['output_model']
>>> predictions = d['predictions']
>>> probabilities = d['probabilities']
```

An implementation of linear SVM for multiclass classification. Given labeled data, a model can be trained and saved for future use; or, a pre-trained model can be used to classify new points. [Detailed documentation](#linear_svm_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `delta` | [`float`](#doc_float) | Margin of difference between correct class and other classes. | `1` |
| `epochs` | [`int`](#doc_int) | Maximum number of full epochs over dataset for psgd | `50` |
| `input_model` | [`LinearSVMModelType`](#doc_model) | Existing model (parameters). | `None` |
| `labels` | [`int vector`](#doc_int_vector) | A matrix containing labels (0 or 1) for the points in the training set (y). | `np.empty([0], dtype=np.uint64)` |
| `lambda_` | [`float`](#doc_float) | L2-regularization parameter for training. | `0.0001` |
| `max_iterations` | [`int`](#doc_int) | Maximum iterations for optimizer (0 indicates no limit). | `10000` |
| `no_intercept` | [`bool`](#doc_bool) | Do not add the intercept term to the model. | `False` |
| `num_classes` | [`int`](#doc_int) | Number of classes for classification; if unspecified (or 0), the number of classes found in the labels will be used. | `0` |
| `optimizer` | [`str`](#doc_str) | Optimizer to use for training ('lbfgs' or 'psgd'). | `'lbfgs'` |
| `seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `shuffle` | [`bool`](#doc_bool) | Don't shuffle the order in which data points are visited for parallel SGD. | `False` |
| `step_size` | [`float`](#doc_float) | Step size for parallel SGD optimizer. | `0.01` |
| `test` | [`matrix`](#doc_matrix) | Matrix containing test dataset. | `np.empty([0, 0])` |
| `test_labels` | [`int vector`](#doc_int_vector) | Matrix containing test labels. | `np.empty([0], dtype=np.uint64)` |
| `tolerance` | [`float`](#doc_float) | Convergence tolerance for optimizer. | `1e-10` |
| `training` | [`matrix`](#doc_matrix) | A matrix containing the training set (the matrix of predictors, X). | `np.empty([0, 0])` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output_model` | [`LinearSVMModelType`](#doc_model) | Output for trained linear svm model. | 
| `predictions` | [`int vector`](#doc_int_vector) | If test data is specified, this matrix is where the predictions for the test set will be saved. | 
| `probabilities` | [`matrix`](#doc_matrix) | If test data is specified, this matrix is where the class probabilities for the test set will be saved. | 

### Detailed documentation
{: #linear_svm_detailed-documentation }

An implementation of linear SVMs that uses either L-BFGS or parallel SGD (stochastic gradient descent) to train the model.

This program allows loading a linear SVM model (via the `input_model` parameter) or training a linear SVM model given training data (specified with the `training` parameter), or both those things at once.  In addition, this program allows classification on a test dataset (specified with the `test` parameter) and the classification results may be saved with the `predictions` output parameter. The trained linear SVM model may be saved using the `output_model` output parameter.

The training data, if specified, may have class labels as its last dimension.  Alternately, the `labels` parameter may be used to specify a separate vector of labels.

When a model is being trained, there are many options.  L2 regularization (to prevent overfitting) can be specified with the `lambda_` option, and the number of classes can be manually specified with the `num_classes`and if an intercept term is not desired in the model, the `no_intercept` parameter can be specified.Margin of difference between correct class and other classes can be specified with the `delta` option.The optimizer used to train the model can be specified with the `optimizer` parameter.  Available options are 'psgd' (parallel stochastic gradient descent) and 'lbfgs' (the L-BFGS optimizer).  There are also various parameters for the optimizer; the `max_iterations` parameter specifies the maximum number of allowed iterations, and the `tolerance` parameter specifies the tolerance for convergence.  For the parallel SGD optimizer, the `step_size` parameter controls the step size taken at each iteration by the optimizer and the maximum number of epochs (specified with `epochs`). If the objective function for your data is oscillating between Inf and 0, the step size is probably too large.  There are more parameters for the optimizers, but the C++ interface must be used to access these.

Optionally, the model can be used to predict the labels for another matrix of data points, if `test` is specified.  The `test` parameter can be specified without the `training` parameter, so long as an existing linear SVM model is given with the `input_model` parameter.  The output predictions from the linear SVM model may be saved with the `predictions` parameter.

### Example
As an example, to train a LinaerSVM on the data '`'data'`' with labels '`'labels'`' with L2 regularization of 0.1, saving the model to '`'lsvm_model'`', the following command may be used:

```python
>>> output = linear_svm(training=data, labels=labels, lambda_=0.1, delta=1,
  num_classes=0)
>>> lsvm_model = output['output_model']
```

Then, to use that model to predict classes for the dataset '`'test'`', storing the output predictions in '`'predictions'`', the following command may be used: 

```python
>>> output = linear_svm(input_model=lsvm_model, test=test)
>>> predictions = output['predictions']
```

### See also

 - [random_forest()](#random_forest)
 - [logistic_regression()](#logistic_regression)
 - [LinearSVM on Wikipedia](https://en.wikipedia.org/wiki/Support-vector_machine)
 - [LinearSVM C++ class documentation](../../user/methods/linear_svm.md)

## lmnn()
{: #lmnn }

#### Large Margin Nearest Neighbors (LMNN)
{: #lmnn_descr }

```python
>>> from mlpack import lmnn
>>> d = lmnn(batch_size=50, center=False, check_input_matrices=False,
        copy_all_inputs=False, distance=np.empty([0, 0]), input_=np.empty([0,
        0]), k=1, labels=np.empty([0], dtype=np.uint64), linear_scan=False,
        max_iterations=100000, normalize=False, optimizer='amsgrad', passes=50,
        print_accuracy=False, rank=0, regularization=0.5, seed=0,
        step_size=0.01, tolerance=1e-07, update_interval=1, verbose=False)
>>> centered_data = d['centered_data']
>>> output = d['output']
>>> transformed_data = d['transformed_data']
```

An implementation of Large Margin Nearest Neighbors (LMNN), a distance learning technique.  Given a labeled dataset, this learns a transformation of the data that improves k-nearest-neighbor performance; this can be useful as a preprocessing step. [Detailed documentation](#lmnn_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `batch_size` | [`int`](#doc_int) | Batch size for mini-batch SGD. | `50` |
| `center` | [`bool`](#doc_bool) | Perform mean-centering on the dataset. It is useful when the centroid of the data is far from the origin. | `False` |
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `distance` | [`matrix`](#doc_matrix) | Initial distance matrix to be used as starting point | `np.empty([0, 0])` |
| `input_` | [`matrix`](#doc_matrix) | Input dataset to run LMNN on. | `**--**` |
| `k` | [`int`](#doc_int) | Number of target neighbors to use for each datapoint. | `1` |
| `labels` | [`int vector`](#doc_int_vector) | Labels for input dataset. | `np.empty([0], dtype=np.uint64)` |
| `linear_scan` | [`bool`](#doc_bool) | Don't shuffle the order in which data points are visited for SGD or mini-batch SGD. | `False` |
| `max_iterations` | [`int`](#doc_int) | Maximum number of iterations for L-BFGS (0 indicates no limit). | `100000` |
| `normalize` | [`bool`](#doc_bool) | Use a normalized starting point for optimization. Itis useful for when points are far apart, or when SGD is returning NaN. | `False` |
| `optimizer` | [`str`](#doc_str) | Optimizer to use; 'amsgrad', 'bbsgd', 'sgd', or 'lbfgs'. | `'amsgrad'` |
| `passes` | [`int`](#doc_int) | Maximum number of full passes over dataset for AMSGrad, BB_SGD and SGD. | `50` |
| `print_accuracy` | [`bool`](#doc_bool) | Print accuracies on initial and transformed dataset | `False` |
| `rank` | [`int`](#doc_int) | Rank of distance matrix to be optimized.  | `0` |
| `regularization` | [`float`](#doc_float) | Regularization for LMNN objective function  | `0.5` |
| `seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `step_size` | [`float`](#doc_float) | Step size for AMSGrad, BB_SGD and SGD (alpha). | `0.01` |
| `tolerance` | [`float`](#doc_float) | Maximum tolerance for termination of AMSGrad, BB_SGD, SGD or L-BFGS. | `1e-07` |
| `update_interval` | [`int`](#doc_int) | Number of iterations after which impostors need to be recalculated. | `1` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `centered_data` | [`matrix`](#doc_matrix) | Output matrix for mean-centered dataset. | 
| `output` | [`matrix`](#doc_matrix) | Output matrix for learned distance matrix. | 
| `transformed_data` | [`matrix`](#doc_matrix) | Output matrix for transformed dataset. | 

### Detailed documentation
{: #lmnn_detailed-documentation }

This program implements Large Margin Nearest Neighbors, a distance learning technique.  The method seeks to improve k-nearest-neighbor classification on a dataset.  The method employes the strategy of reducing distance between similar labeled data points (a.k.a target neighbors) and increasing distance between differently labeled points (a.k.a impostors) using standard optimization techniques over the gradient of the distance between data points.

To work, this algorithm needs labeled data.  It can be given as the last row of the input dataset (specified with `input_`), or alternatively as a separate matrix (specified with `labels`).  Additionally, a starting point for optimization (specified with `distance`can be given, having (r x d) dimensionality.  Here r should satisfy 1 <= r <= d, Consequently a Low-Rank matrix will be optimized. Alternatively, Low-Rank distance can be learned by specifying the `rank`parameter (A Low-Rank matrix with uniformly distributed values will be used as initial learning point). 

The program also requires number of targets neighbors to work with ( specified with `k`), A regularization parameter can also be passed, It acts as a trade of between the pulling and pushing terms (specified with `regularization`), In addition, this implementation of LMNN includes a parameter to decide the interval after which impostors must be re-calculated (specified with `update_interval`).

Output can either be the learned distance matrix (specified with `output`), or the transformed dataset  (specified with `transformed_data`), or both. Additionally mean-centered dataset (specified with `centered_data`) can be accessed given mean-centering (specified with `center`) is performed on the dataset. Accuracy on initial dataset and final transformed dataset can be printed by specifying the `print_accuracy`parameter. 

This implementation of LMNN uses AdaGrad, BigBatch_SGD, stochastic gradient descent, mini-batch stochastic gradient descent, or the L_BFGS optimizer. 

AdaGrad, specified by the value 'adagrad' for the parameter `optimizer`, uses maximum of past squared gradients. It primarily on six parameters: the step size (specified with `step_size`), the batch size (specified with `batch_size`), the maximum number of passes (specified with `passes`). Inaddition, a normalized starting point can be used by specifying the `normalize` parameter. 

BigBatch_SGD, specified by the value 'bbsgd' for the parameter `optimizer`, depends primarily on four parameters: the step size (specified with `step_size`), the batch size (specified with `batch_size`), the maximum number of passes (specified with `passes`).  In addition, a normalized starting point can be used by specifying the `normalize` parameter. 

Stochastic gradient descent, specified by the value 'sgd' for the parameter `optimizer`, depends primarily on three parameters: the step size (specified with `step_size`), the batch size (specified with `batch_size`), and the maximum number of passes (specified with `passes`).  In addition, a normalized starting point can be used by specifying the `normalize` parameter. Furthermore, mean-centering can be performed on the dataset by specifying the `center`parameter. 

The L-BFGS optimizer, specified by the value 'lbfgs' for the parameter `optimizer`, uses a back-tracking line search algorithm to minimize a function.  The following parameters are used by L-BFGS: `max_iterations`, `tolerance`(the optimization is terminated when the gradient norm is below this value).  For more details on the L-BFGS optimizer, consult either the mlpack L-BFGS documentation (in lbfgs.hpp) or the vast set of published literature on L-BFGS.  In addition, a normalized starting point can be used by specifying the `normalize` parameter.

By default, the AMSGrad optimizer is used.

### Example
Example - Let's say we want to learn distance on iris dataset with number of targets as 3 using BigBatch_SGD optimizer. A simple call for the same will look like: 

```python
>>> output = lmnn(input_=iris, labels=iris_labels, k=3, optimizer='bbsgd')
>>> output = output['output']
```

Another program call making use of update interval & regularization parameter with dataset having labels as last column can be made as: 

```python
>>> output = lmnn(input_=letter_recognition, k=5, update_interval=10,
  regularization=0.4)
>>> output = output['output']
```

### See also

 - [nca()](#nca)
 - [Large margin nearest neighbor on Wikipedia](https://en.wikipedia.org/wiki/Large_margin_nearest_neighbor)
 - [Distance metric learning for large margin nearest neighbor classification (pdf)](https://proceedings.neurips.cc/paper_files/paper/2005/file/a7f592cef8b130a6967a90617db5681b-Paper.pdf)
 - [LMNN C++ class documentation](../../user/methods/lmnn.md)

## local_coordinate_coding()
{: #local_coordinate_coding }

#### Local Coordinate Coding
{: #local_coordinate_coding_descr }

```python
>>> from mlpack import local_coordinate_coding
>>> d = local_coordinate_coding(atoms=0, check_input_matrices=False,
        copy_all_inputs=False, initial_dictionary=np.empty([0, 0]),
        input_model=None, lambda_=0, max_iterations=0, normalize=False, seed=0,
        test=np.empty([0, 0]), tolerance=0.01, training=np.empty([0, 0]),
        verbose=False)
>>> codes = d['codes']
>>> dictionary = d['dictionary']
>>> output_model = d['output_model']
```

An implementation of Local Coordinate Coding (LCC), a data transformation technique.  Given input data, this transforms each point to be expressed as a linear combination of a few points in the dataset; once an LCC model is trained, it can be used to transform points later also. [Detailed documentation](#local_coordinate_coding_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `atoms` | [`int`](#doc_int) | Number of atoms in the dictionary. | `0` |
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `initial_dictionary` | [`matrix`](#doc_matrix) | Optional initial dictionary. | `np.empty([0, 0])` |
| `input_model` | [`LocalCoordinateCoding<>Type`](#doc_model) | Input LCC model. | `None` |
| `lambda_` | [`float`](#doc_float) | Weighted l1-norm regularization parameter. | `0` |
| `max_iterations` | [`int`](#doc_int) | Maximum number of iterations for LCC (0 indicates no limit). | `0` |
| `normalize` | [`bool`](#doc_bool) | If set, the input data matrix will be normalized before coding. | `False` |
| `seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `test` | [`matrix`](#doc_matrix) | Test points to encode. | `np.empty([0, 0])` |
| `tolerance` | [`float`](#doc_float) | Tolerance for objective function. | `0.01` |
| `training` | [`matrix`](#doc_matrix) | Matrix of training data (X). | `np.empty([0, 0])` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `codes` | [`matrix`](#doc_matrix) | Output codes matrix. | 
| `dictionary` | [`matrix`](#doc_matrix) | Output dictionary matrix. | 
| `output_model` | [`LocalCoordinateCoding<>Type`](#doc_model) | Output for trained LCC model. | 

### Detailed documentation
{: #local_coordinate_coding_detailed-documentation }

An implementation of Local Coordinate Coding (LCC), which codes data that approximately lives on a manifold using a variation of l1-norm regularized sparse coding.  Given a dense data matrix X with n points and d dimensions, LCC seeks to find a dense dictionary matrix D with k atoms in d dimensions, and a coding matrix Z with n points in k dimensions.  Because of the regularization method used, the atoms in D should lie close to the manifold on which the data points lie.

The original data matrix X can then be reconstructed as D * Z.  Therefore, this program finds a representation of each point in X as a sparse linear combination of atoms in the dictionary D.

The coding is found with an algorithm which alternates between a dictionary step, which updates the dictionary D, and a coding step, which updates the coding matrix Z.

To run this program, the input matrix X must be specified (with -i), along with the number of atoms in the dictionary (-k).  An initial dictionary may also be specified with the `initial_dictionary` parameter.  The l1-norm regularization parameter is specified with the `lambda_` parameter.

### Example
For example, to run LCC on the dataset `'data'` using 200 atoms and an l1-regularization parameter of 0.1, saving the dictionary `dictionary` and the codes into `codes`, use

```python
>>> output = local_coordinate_coding(training=data, atoms=200, lambda_=0.1)
>>> dict = output['dictionary']
>>> codes = output['codes']
```

The maximum number of iterations may be specified with the `max_iterations` parameter. Optionally, the input data matrix X can be normalized before coding with the `normalize` parameter.

An LCC model may be saved using the `output_model` output parameter.  Then, to encode new points from the dataset `'points'` with the previously saved model `'lcc_model'`, saving the new codes to `'new_codes'`, the following command can be used:

```python
>>> output = local_coordinate_coding(input_model=lcc_model, test=points)
>>> new_codes = output['codes']
```

### See also

 - [sparse_coding()](#sparse_coding)
 - [Nonlinear learning using local coordinate coding (pdf)](https://proceedings.neurips.cc/paper_files/paper/2009/file/2afe4567e1bf64d32a5527244d104cea-Paper.pdf)
 - [LocalCoordinateCoding C++ class documentation](../../user/methods/local_coordinate_coding.md)

## logistic_regression()
{: #logistic_regression }

#### L2-regularized Logistic Regression and Prediction
{: #logistic_regression_descr }

```python
>>> from mlpack import logistic_regression
>>> d = logistic_regression(batch_size=64, check_input_matrices=False,
        copy_all_inputs=False, decision_boundary=0.5, input_model=None,
        labels=np.empty([0], dtype=np.uint64), lambda_=0, max_iterations=10000,
        optimizer='lbfgs', print_training_accuracy=False, step_size=0.01,
        test=np.empty([0, 0]), tolerance=1e-10, training=np.empty([0, 0]),
        verbose=False)
>>> output_model = d['output_model']
>>> predictions = d['predictions']
>>> probabilities = d['probabilities']
```

An implementation of L2-regularized logistic regression for two-class classification.  Given labeled data, a model can be trained and saved for future use; or, a pre-trained model can be used to classify new points. [Detailed documentation](#logistic_regression_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `batch_size` | [`int`](#doc_int) | Batch size for SGD. | `64` |
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `decision_boundary` | [`float`](#doc_float) | Decision boundary for prediction; if the logistic function for a point is less than the boundary, the class is taken to be 0; otherwise, the class is 1. | `0.5` |
| `input_model` | [`LogisticRegression<>Type`](#doc_model) | Existing model (parameters). | `None` |
| `labels` | [`int vector`](#doc_int_vector) | A matrix containing labels (0 or 1) for the points in the training set (y). | `np.empty([0], dtype=np.uint64)` |
| `lambda_` | [`float`](#doc_float) | L2-regularization parameter for training. | `0` |
| `max_iterations` | [`int`](#doc_int) | Maximum iterations for optimizer (0 indicates no limit). | `10000` |
| `optimizer` | [`str`](#doc_str) | Optimizer to use for training ('lbfgs' or 'sgd'). | `'lbfgs'` |
| `print_training_accuracy` | [`bool`](#doc_bool) | If set, then the accuracy of the model on the training set will be printed (verbose must also be specified). | `False` |
| `step_size` | [`float`](#doc_float) | Step size for SGD optimizer. | `0.01` |
| `test` | [`matrix`](#doc_matrix) | Matrix containing test dataset. | `np.empty([0, 0])` |
| `tolerance` | [`float`](#doc_float) | Convergence tolerance for optimizer. | `1e-10` |
| `training` | [`matrix`](#doc_matrix) | A matrix containing the training set (the matrix of predictors, X). | `np.empty([0, 0])` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output_model` | [`LogisticRegression<>Type`](#doc_model) | Output for trained logistic regression model. | 
| `predictions` | [`int vector`](#doc_int_vector) | If test data is specified, this matrix is where the predictions for the test set will be saved. | 
| `probabilities` | [`matrix`](#doc_matrix) | If test data is specified, this matrix is where the class probabilities for the test set will be saved. | 

### Detailed documentation
{: #logistic_regression_detailed-documentation }

An implementation of L2-regularized logistic regression using either the L-BFGS optimizer or SGD (stochastic gradient descent).  This solves the regression problem

  y = (1 / 1 + e^-(X * b)).

In this setting, y corresponds to class labels and X corresponds to data.

This program allows loading a logistic regression model (via the `input_model` parameter) or training a logistic regression model given training data (specified with the `training` parameter), or both those things at once.  In addition, this program allows classification on a test dataset (specified with the `test` parameter) and the classification results may be saved with the `predictions` output parameter. The trained logistic regression model may be saved using the `output_model` output parameter.

The training data, if specified, may have class labels as its last dimension.  Alternately, the `labels` parameter may be used to specify a separate matrix of labels.

When a model is being trained, there are many options.  L2 regularization (to prevent overfitting) can be specified with the `lambda_` option, and the optimizer used to train the model can be specified with the `optimizer` parameter.  Available options are 'sgd' (stochastic gradient descent) and 'lbfgs' (the L-BFGS optimizer).  There are also various parameters for the optimizer; the `max_iterations` parameter specifies the maximum number of allowed iterations, and the `tolerance` parameter specifies the tolerance for convergence.  For the SGD optimizer, the `step_size` parameter controls the step size taken at each iteration by the optimizer.  The batch size for SGD is controlled with the `batch_size` parameter. If the objective function for your data is oscillating between Inf and 0, the step size is probably too large.  There are more parameters for the optimizers, but the C++ interface must be used to access these.

For SGD, an iteration refers to a single point. So to take a single pass over the dataset with SGD, `max_iterations` should be set to the number of points in the dataset.

Optionally, the model can be used to predict the responses for another matrix of data points, if `test` is specified.  The `test` parameter can be specified without the `training` parameter, so long as an existing logistic regression model is given with the `input_model` parameter.  The output predictions from the logistic regression model may be saved with the `predictions` parameter.

This implementation of logistic regression does not support the general multi-class case but instead only the two-class case.  Any labels must be either 0 or 1.  For more classes, see the softmax regression implementation.

### Example
As an example, to train a logistic regression model on the data '`'data'`' with labels '`'labels'`' with L2 regularization of 0.1, saving the model to '`'lr_model'`', the following command may be used:

```python
>>> output = logistic_regression(training=data, labels=labels, lambda_=0.1,
  print_training_accuracy=True)
>>> lr_model = output['output_model']
```

Then, to use that model to predict classes for the dataset '`'test'`', storing the output predictions in '`'predictions'`', the following command may be used: 

```python
>>> output = logistic_regression(input_model=lr_model, test=test)
>>> predictions = output['predictions']
```

### See also

 - [softmax_regression()](#softmax_regression)
 - [random_forest()](#random_forest)
 - [Logistic regression on Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)
 - [:LogisticRegression C++ class documentation](../../user/methods/logistic_regression.md)

## lsh()
{: #lsh }

#### K-Approximate-Nearest-Neighbor Search with LSH
{: #lsh_descr }

```python
>>> from mlpack import lsh
>>> d = lsh(bucket_size=500, check_input_matrices=False,
        copy_all_inputs=False, hash_width=0, input_model=None, k=0,
        num_probes=0, projections=10, query=np.empty([0, 0]),
        reference=np.empty([0, 0]), second_hash_size=99901, seed=0, tables=30,
        true_neighbors=np.empty([0, 0], dtype=np.uint64), verbose=False)
>>> distances = d['distances']
>>> neighbors = d['neighbors']
>>> output_model = d['output_model']
```

An implementation of approximate k-nearest-neighbor search with locality-sensitive hashing (LSH).  Given a set of reference points and a set of query points, this will compute the k approximate nearest neighbors of each query point in the reference set; models can be saved for future use. [Detailed documentation](#lsh_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `bucket_size` | [`int`](#doc_int) | The size of a bucket in the second level hash. | `500` |
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `hash_width` | [`float`](#doc_float) | The hash width for the first-level hashing in the LSH preprocessing. By default, the LSH class automatically estimates a hash width for its use. | `0` |
| `input_model` | [`LSHSearch<>Type`](#doc_model) | Input LSH model. | `None` |
| `k` | [`int`](#doc_int) | Number of nearest neighbors to find. | `0` |
| `num_probes` | [`int`](#doc_int) | Number of additional probes for multiprobe LSH; if 0, traditional LSH is used. | `0` |
| `projections` | [`int`](#doc_int) | The number of hash functions for each table | `10` |
| `query` | [`matrix`](#doc_matrix) | Matrix containing query points (optional). | `np.empty([0, 0])` |
| `reference` | [`matrix`](#doc_matrix) | Matrix containing the reference dataset. | `np.empty([0, 0])` |
| `second_hash_size` | [`int`](#doc_int) | The size of the second level hash table. | `99901` |
| `seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `tables` | [`int`](#doc_int) | The number of hash tables to be used. | `30` |
| `true_neighbors` | [`int matrix`](#doc_int_matrix) | Matrix of true neighbors to compute recall with (the recall is printed when -v is specified). | `np.empty([0, 0], dtype=np.uint64)` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `distances` | [`matrix`](#doc_matrix) | Matrix to output distances into. | 
| `neighbors` | [`int matrix`](#doc_int_matrix) | Matrix to output neighbors into. | 
| `output_model` | [`LSHSearch<>Type`](#doc_model) | Output for trained LSH model. | 

### Detailed documentation
{: #lsh_detailed-documentation }

This program will calculate the k approximate-nearest-neighbors of a set of points using locality-sensitive hashing. You may specify a separate set of reference points and query points, or just a reference set which will be used as both the reference and query set. 

### Example
For example, the following will return 5 neighbors from the data for each point in `'input'` and store the distances in `'distances'` and the neighbors in `'neighbors'`:

```python
>>> output = lsh(k=5, reference=input)
>>> distances = output['distances']
>>> neighbors = output['neighbors']
```

The output is organized such that row i and column j in the neighbors output corresponds to the index of the point in the reference set which is the j'th nearest neighbor from the point in the query set with index i.  Row j and column i in the distances output file corresponds to the distance between those two points.

Because this is approximate-nearest-neighbors search, results may be different from run to run.  Thus, the `seed` parameter can be specified to set the random seed.

This program also has many other parameters to control its functionality; see the parameter-specific documentation for more information.

### See also

 - [knn()](#knn)
 - [krann()](#krann)
 - [Locality-sensitive hashing on Wikipedia](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)
 - [Locality-sensitive hashing scheme based on p-stable  distributions(pdf)](https://www.mlpack.org/papers/lsh.pdf)
 - [LSHSearch C++ class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/lsh/lsh.hpp)

## mean_shift()
{: #mean_shift }

#### Mean Shift Clustering
{: #mean_shift_descr }

```python
>>> from mlpack import mean_shift
>>> d = mean_shift(check_input_matrices=False, copy_all_inputs=False,
        force_convergence=False, in_place=False, input_=np.empty([0, 0]),
        labels_only=False, max_iterations=1000, radius=0, verbose=False)
>>> centroid = d['centroid']
>>> output = d['output']
```

A fast implementation of mean-shift clustering using dual-tree range search.  Given a dataset, this uses the mean shift algorithm to produce and return a clustering of the data. [Detailed documentation](#mean_shift_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `force_convergence` | [`bool`](#doc_bool) | If specified, the mean shift algorithm will continue running regardless of max_iterations until the clusters converge. | `False` |
| `in_place` | [`bool`](#doc_bool) | If specified, a column containing the learned cluster assignments will be added to the input dataset file.  In this case, --output_file is overridden.  (Do not use with Python.) | `False` |
| `input_` | [`matrix`](#doc_matrix) | Input dataset to perform clustering on. | `**--**` |
| `labels_only` | [`bool`](#doc_bool) | If specified, only the output labels will be written to the file specified by --output_file. | `False` |
| `max_iterations` | [`int`](#doc_int) | Maximum number of iterations before mean shift terminates. | `1000` |
| `radius` | [`float`](#doc_float) | If the distance between two centroids is less than the given radius, one will be removed.  A radius of 0 or less means an estimate will be calculated and used for the radius. | `0` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `centroid` | [`matrix`](#doc_matrix) | If specified, the centroids of each cluster will be written to the given matrix. | 
| `output` | [`matrix`](#doc_matrix) | Matrix to write output labels or labeled data to. | 

### Detailed documentation
{: #mean_shift_detailed-documentation }

This program performs mean shift clustering on the given dataset, storing the learned cluster assignments either as a column of labels in the input dataset or separately.

The input dataset should be specified with the `input_` parameter, and the radius used for search can be specified with the `radius` parameter.  The maximum number of iterations before algorithm termination is controlled with the `max_iterations` parameter.

The output labels may be saved with the `output` output parameter and the centroids of each cluster may be saved with the `centroid` output parameter.

### Example
For example, to run mean shift clustering on the dataset `'data'` and store the centroids to `'centroids'`, the following command may be used: 

```python
>>> output = mean_shift(input_=data)
>>> centroids = output['centroid']
```

### See also

 - [kmeans()](#kmeans)
 - [dbscan()](#dbscan)
 - [Mean shift on Wikipedia](https://en.wikipedia.org/wiki/Mean_shift)
 - [Mean Shift, Mode Seeking, and Clustering (pdf)](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=1c168275c59ba382588350ee1443537f59978183)
 - [mlpack::mean_shift::MeanShift C++ class documentation](../../user/methods/mean_shift.md)

## nbc()
{: #nbc }

#### Parametric Naive Bayes Classifier
{: #nbc_descr }

```python
>>> from mlpack import nbc
>>> d = nbc(check_input_matrices=False, copy_all_inputs=False,
        incremental_variance=False, input_model=None, labels=np.empty([0],
        dtype=np.uint64), test=np.empty([0, 0]), training=np.empty([0, 0]),
        verbose=False)
>>> output_model = d['output_model']
>>> predictions = d['predictions']
>>> probabilities = d['probabilities']
```

An implementation of the Naive Bayes Classifier, used for classification. Given labeled data, an NBC model can be trained and saved, or, a pre-trained model can be used for classification. [Detailed documentation](#nbc_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `incremental_variance` | [`bool`](#doc_bool) | The variance of each class will be calculated incrementally. | `False` |
| `input_model` | [`NBCModelType`](#doc_model) | Input Naive Bayes model. | `None` |
| `labels` | [`int vector`](#doc_int_vector) | A file containing labels for the training set. | `np.empty([0], dtype=np.uint64)` |
| `test` | [`matrix`](#doc_matrix) | A matrix containing the test set. | `np.empty([0, 0])` |
| `training` | [`matrix`](#doc_matrix) | A matrix containing the training set. | `np.empty([0, 0])` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output_model` | [`NBCModelType`](#doc_model) | File to save trained Naive Bayes model to. | 
| `predictions` | [`int vector`](#doc_int_vector) | The matrix in which the predicted labels for the test set will be written. | 
| `probabilities` | [`matrix`](#doc_matrix) | The matrix in which the predicted probability of labels for the test set will be written. | 

### Detailed documentation
{: #nbc_detailed-documentation }

This program trains the Naive Bayes classifier on the given labeled training set, or loads a model from the given model file, and then may use that trained model to classify the points in a given test set.

The training set is specified with the `training` parameter.  Labels may be either the last row of the training set, or alternately the `labels` parameter may be specified to pass a separate matrix of labels.

If training is not desired, a pre-existing model may be loaded with the `input_model` parameter.



The `incremental_variance` parameter can be used to force the training to use an incremental algorithm for calculating variance.  This is slower, but can help avoid loss of precision in some cases.

If classifying a test set is desired, the test set may be specified with the `test` parameter, and the classifications may be saved with the `predictions`predictions  parameter.  If saving the trained model is desired, this may be done with the `output_model` output parameter.

### Example
For example, to train a Naive Bayes classifier on the dataset `'data'` with labels `'labels'` and save the model to `'nbc_model'`, the following command may be used:

```python
>>> output = nbc(training=data, labels=labels)
>>> nbc_model = output['output_model']
```

Then, to use `'nbc_model'` to predict the classes of the dataset `'test_set'` and save the predicted classes to `'predictions'`, the following command may be used:

```python
>>> output = nbc(input_model=nbc_model, test=test_set)
>>> predictions = output['predictions']
```

### See also

 - [softmax_regression()](#softmax_regression)
 - [random_forest()](#random_forest)
 - [Naive Bayes classifier on Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
 - [NaiveBayesClassifier C++ class documentation](../../user/methods/naive_bayes_classifier.md)

## nca()
{: #nca }

#### Neighborhood Components Analysis (NCA)
{: #nca_descr }

```python
>>> from mlpack import nca
>>> d = nca(armijo_constant=0.0001, batch_size=50,
        check_input_matrices=False, copy_all_inputs=False, input_=np.empty([0,
        0]), labels=np.empty([0], dtype=np.uint64), linear_scan=False,
        max_iterations=500000, max_line_search_trials=50, max_step=1e+20,
        min_step=1e-20, normalize=False, num_basis=5, optimizer='sgd', seed=0,
        step_size=0.01, tolerance=1e-07, verbose=False, wolfe=0.9)
>>> output = d['output']
```

An implementation of neighborhood components analysis, a distance learning technique that can be used for preprocessing.  Given a labeled dataset, this uses NCA, which seeks to improve the k-nearest-neighbor classification, and returns the learned distance metric. [Detailed documentation](#nca_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `armijo_constant` | [`float`](#doc_float) | Armijo constant for L-BFGS. | `0.0001` |
| `batch_size` | [`int`](#doc_int) | Batch size for mini-batch SGD. | `50` |
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `input_` | [`matrix`](#doc_matrix) | Input dataset to run NCA on. | `**--**` |
| `labels` | [`int vector`](#doc_int_vector) | Labels for input dataset. | `np.empty([0], dtype=np.uint64)` |
| `linear_scan` | [`bool`](#doc_bool) | Don't shuffle the order in which data points are visited for SGD or mini-batch SGD. | `False` |
| `max_iterations` | [`int`](#doc_int) | Maximum number of iterations for SGD or L-BFGS (0 indicates no limit). | `500000` |
| `max_line_search_trials` | [`int`](#doc_int) | Maximum number of line search trials for L-BFGS. | `50` |
| `max_step` | [`float`](#doc_float) | Maximum step of line search for L-BFGS. | `1e+20` |
| `min_step` | [`float`](#doc_float) | Minimum step of line search for L-BFGS. | `1e-20` |
| `normalize` | [`bool`](#doc_bool) | Use a normalized starting point for optimization. This is useful for when points are far apart, or when SGD is returning NaN. | `False` |
| `num_basis` | [`int`](#doc_int) | Number of memory points to be stored for L-BFGS. | `5` |
| `optimizer` | [`str`](#doc_str) | Optimizer to use; 'sgd' or 'lbfgs'. | `'sgd'` |
| `seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `step_size` | [`float`](#doc_float) | Step size for stochastic gradient descent (alpha). | `0.01` |
| `tolerance` | [`float`](#doc_float) | Maximum tolerance for termination of SGD or L-BFGS. | `1e-07` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |
| `wolfe` | [`float`](#doc_float) | Wolfe condition parameter for L-BFGS. | `0.9` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output` | [`matrix`](#doc_matrix) | Output matrix for learned distance matrix. | 

### Detailed documentation
{: #nca_detailed-documentation }

This program implements Neighborhood Components Analysis, both a linear dimensionality reduction technique and a distance learning technique.  The method seeks to improve k-nearest-neighbor classification on a dataset by scaling the dimensions.  The method is nonparametric, and does not require a value of k.  It works by using stochastic ("soft") neighbor assignments and using optimization techniques over the gradient of the accuracy of the neighbor assignments.

To work, this algorithm needs labeled data.  It can be given as the last row of the input dataset (specified with `input_`), or alternatively as a separate matrix (specified with `labels`).

This implementation of NCA uses stochastic gradient descent, mini-batch stochastic gradient descent, or the L_BFGS optimizer.  These optimizers do not guarantee global convergence for a nonconvex objective function (NCA's objective function is nonconvex), so the final results could depend on the random seed or other optimizer parameters.

Stochastic gradient descent, specified by the value 'sgd' for the parameter `optimizer`, depends primarily on three parameters: the step size (specified with `step_size`), the batch size (specified with `batch_size`), and the maximum number of iterations (specified with `max_iterations`).  In addition, a normalized starting point can be used by specifying the `normalize` parameter, which is necessary if many warnings of the form 'Denominator of p_i is 0!' are given.  Tuning the step size can be a tedious affair.  In general, the step size is too large if the objective is not mostly uniformly decreasing, or if zero-valued denominator warnings are being issued.  The step size is too small if the objective is changing very slowly.  Setting the termination condition can be done easily once a good step size parameter is found; either increase the maximum iterations to a large number and allow SGD to find a minimum, or set the maximum iterations to 0 (allowing infinite iterations) and set the tolerance (specified by `tolerance`) to define the maximum allowed difference between objectives for SGD to terminate.  Be careful---setting the tolerance instead of the maximum iterations can take a very long time and may actually never converge due to the properties of the SGD optimizer. Note that a single iteration of SGD refers to a single point, so to take a single pass over the dataset, set the value of the `max_iterations` parameter equal to the number of points in the dataset.

The L-BFGS optimizer, specified by the value 'lbfgs' for the parameter `optimizer`, uses a back-tracking line search algorithm to minimize a function.  The following parameters are used by L-BFGS: `num_basis` (specifies the number of memory points used by L-BFGS), `max_iterations`, `armijo_constant`, `wolfe`, `tolerance` (the optimization is terminated when the gradient norm is below this value), `max_line_search_trials`, `min_step`, and `max_step` (which both refer to the line search routine).  For more details on the L-BFGS optimizer, consult either the mlpack L-BFGS documentation (in lbfgs.hpp) or the vast set of published literature on L-BFGS.

By default, the SGD optimizer is used.

### See also

 - [lmnn()](#lmnn)
 - [Neighbourhood components analysis on Wikipedia](https://en.wikipedia.org/wiki/Neighbourhood_components_analysis)
 - [Neighbourhood components analysis (pdf)](https://proceedings.neurips.cc/paper_files/paper/2004/file/42fe880812925e520249e808937738d2-Paper.pdf)
 - [NCA C++ class documentation](../../user/methods/nca.md)

## knn()
{: #knn }

#### k-Nearest-Neighbors Search
{: #knn_descr }

```python
>>> from mlpack import knn
>>> d = knn(algorithm='dual_tree', check_input_matrices=False,
        copy_all_inputs=False, epsilon=0, input_model=None, k=0, leaf_size=20,
        query=np.empty([0, 0]), random_basis=False, reference=np.empty([0, 0]),
        rho=0.7, seed=0, tau=0, tree_type='kd', true_distances=np.empty([0, 0]),
        true_neighbors=np.empty([0, 0], dtype=np.uint64), verbose=False)
>>> distances = d['distances']
>>> neighbors = d['neighbors']
>>> output_model = d['output_model']
```

An implementation of k-nearest-neighbor search using single-tree and dual-tree algorithms.  Given a set of reference points and query points, this can find the k nearest neighbors in the reference set of each query point using trees; trees that are built can be saved for future use. [Detailed documentation](#knn_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `algorithm` | [`str`](#doc_str) | Type of neighbor search: 'naive', 'single_tree', 'dual_tree', 'greedy'. | `'dual_tree'` |
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `epsilon` | [`float`](#doc_float) | If specified, will do approximate nearest neighbor search with given relative error. | `0` |
| `input_model` | [`KNNModelType`](#doc_model) | Pre-trained kNN model. | `None` |
| `k` | [`int`](#doc_int) | Number of nearest neighbors to find. | `0` |
| `leaf_size` | [`int`](#doc_int) | Leaf size for tree building (used for kd-trees, vp trees, random projection trees, UB trees, R trees, R* trees, X trees, Hilbert R trees, R+ trees, R++ trees, spill trees, and octrees). | `20` |
| `query` | [`matrix`](#doc_matrix) | Matrix containing query points (optional). | `np.empty([0, 0])` |
| `random_basis` | [`bool`](#doc_bool) | Before tree-building, project the data onto a random orthogonal basis. | `False` |
| `reference` | [`matrix`](#doc_matrix) | Matrix containing the reference dataset. | `np.empty([0, 0])` |
| `rho` | [`float`](#doc_float) | Balance threshold (only valid for spill trees). | `0.7` |
| `seed` | [`int`](#doc_int) | Random seed (if 0, std::time(NULL) is used). | `0` |
| `tau` | [`float`](#doc_float) | Overlapping size (only valid for spill trees). | `0` |
| `tree_type` | [`str`](#doc_str) | Type of tree to use: 'kd', 'vp', 'rp', 'max-rp', 'ub', 'cover', 'r', 'r-star', 'x', 'ball', 'hilbert-r', 'r-plus', 'r-plus-plus', 'spill', 'oct'. | `'kd'` |
| `true_distances` | [`matrix`](#doc_matrix) | Matrix of true distances to compute the effective error (average relative error) (it is printed when -v is specified). | `np.empty([0, 0])` |
| `true_neighbors` | [`int matrix`](#doc_int_matrix) | Matrix of true neighbors to compute the recall (it is printed when -v is specified). | `np.empty([0, 0], dtype=np.uint64)` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `distances` | [`matrix`](#doc_matrix) | Matrix to output distances into. | 
| `neighbors` | [`int matrix`](#doc_int_matrix) | Matrix to output neighbors into. | 
| `output_model` | [`KNNModelType`](#doc_model) | If specified, the kNN model will be output here. | 

### Detailed documentation
{: #knn_detailed-documentation }

This program will calculate the k-nearest-neighbors of a set of points using kd-trees or cover trees (cover tree support is experimental and may be slow). You may specify a separate set of reference points and query points, or just a reference set which will be used as both the reference and query set.

### Example
For example, the following command will calculate the 5 nearest neighbors of each point in `'input'` and store the distances in `'distances'` and the neighbors in `'neighbors'`: 

```python
>>> output = knn(k=5, reference=input)
>>> neighbors = output['neighbors']
>>> distances = output['distances']
```

The output is organized such that row i and column j in the neighbors output matrix corresponds to the index of the point in the reference set which is the j'th nearest neighbor from the point in the query set with index i.  Row j and column i in the distances output matrix corresponds to the distance between those two points.

### See also

 - [lsh()](#lsh)
 - [krann()](#krann)
 - [kfn()](#kfn)
 - [Tree-independent dual-tree algorithms (pdf)](http://proceedings.mlr.press/v28/curtin13.pdf)
 - [NeighborSearch C++ class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/neighbor_search/neighbor_search.hpp)

## kfn()
{: #kfn }

#### k-Furthest-Neighbors Search
{: #kfn_descr }

```python
>>> from mlpack import kfn
>>> d = kfn(algorithm='dual_tree', check_input_matrices=False,
        copy_all_inputs=False, epsilon=0, input_model=None, k=0, leaf_size=20,
        percentage=1, query=np.empty([0, 0]), random_basis=False,
        reference=np.empty([0, 0]), seed=0, tree_type='kd',
        true_distances=np.empty([0, 0]), true_neighbors=np.empty([0, 0],
        dtype=np.uint64), verbose=False)
>>> distances = d['distances']
>>> neighbors = d['neighbors']
>>> output_model = d['output_model']
```

An implementation of k-furthest-neighbor search using single-tree and dual-tree algorithms.  Given a set of reference points and query points, this can find the k furthest neighbors in the reference set of each query point using trees; trees that are built can be saved for future use. [Detailed documentation](#kfn_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `algorithm` | [`str`](#doc_str) | Type of neighbor search: 'naive', 'single_tree', 'dual_tree', 'greedy'. | `'dual_tree'` |
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `epsilon` | [`float`](#doc_float) | If specified, will do approximate furthest neighbor search with given relative error. Must be in the range [0,1). | `0` |
| `input_model` | [`KFNModelType`](#doc_model) | Pre-trained kFN model. | `None` |
| `k` | [`int`](#doc_int) | Number of furthest neighbors to find. | `0` |
| `leaf_size` | [`int`](#doc_int) | Leaf size for tree building (used for kd-trees, vp trees, random projection trees, UB trees, R trees, R* trees, X trees, Hilbert R trees, R+ trees, R++ trees, and octrees). | `20` |
| `percentage` | [`float`](#doc_float) | If specified, will do approximate furthest neighbor search. Must be in the range (0,1] (decimal form). Resultant neighbors will be at least (p*100) % of the distance as the true furthest neighbor. | `1` |
| `query` | [`matrix`](#doc_matrix) | Matrix containing query points (optional). | `np.empty([0, 0])` |
| `random_basis` | [`bool`](#doc_bool) | Before tree-building, project the data onto a random orthogonal basis. | `False` |
| `reference` | [`matrix`](#doc_matrix) | Matrix containing the reference dataset. | `np.empty([0, 0])` |
| `seed` | [`int`](#doc_int) | Random seed (if 0, std::time(NULL) is used). | `0` |
| `tree_type` | [`str`](#doc_str) | Type of tree to use: 'kd', 'vp', 'rp', 'max-rp', 'ub', 'cover', 'r', 'r-star', 'x', 'ball', 'hilbert-r', 'r-plus', 'r-plus-plus', 'oct'. | `'kd'` |
| `true_distances` | [`matrix`](#doc_matrix) | Matrix of true distances to compute the effective error (average relative error) (it is printed when -v is specified). | `np.empty([0, 0])` |
| `true_neighbors` | [`int matrix`](#doc_int_matrix) | Matrix of true neighbors to compute the recall (it is printed when -v is specified). | `np.empty([0, 0], dtype=np.uint64)` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `distances` | [`matrix`](#doc_matrix) | Matrix to output distances into. | 
| `neighbors` | [`int matrix`](#doc_int_matrix) | Matrix to output neighbors into. | 
| `output_model` | [`KFNModelType`](#doc_model) | If specified, the kFN model will be output here. | 

### Detailed documentation
{: #kfn_detailed-documentation }

This program will calculate the k-furthest-neighbors of a set of points. You may specify a separate set of reference points and query points, or just a reference set which will be used as both the reference and query set.

### Example
For example, the following will calculate the 5 furthest neighbors of eachpoint in `'input'` and store the distances in `'distances'` and the neighbors in `'neighbors'`: 

```python
>>> output = kfn(k=5, reference=input)
>>> distances = output['distances']
>>> neighbors = output['neighbors']
```

The output files are organized such that row i and column j in the neighbors output matrix corresponds to the index of the point in the reference set which is the j'th furthest neighbor from the point in the query set with index i.  Row i and column j in the distances output file corresponds to the distance between those two points.

### See also

 - [approx_kfn()](#approx_kfn)
 - [knn()](#knn)
 - [Tree-independent dual-tree algorithms (pdf)](http://proceedings.mlr.press/v28/curtin13.pdf)
 - [NeighborSearch C++ class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/neighbor_search/neighbor_search.hpp)

## nmf()
{: #nmf }

#### Non-negative Matrix Factorization
{: #nmf_descr }

```python
>>> from mlpack import nmf
>>> d = nmf(check_input_matrices=False, copy_all_inputs=False,
        initial_h=np.empty([0, 0]), initial_w=np.empty([0, 0]),
        input_=np.empty([0, 0]), max_iterations=10000, min_residue=1e-05,
        rank=0, seed=0, update_rules='multdist', verbose=False)
>>> h = d['h']
>>> w = d['w']
```

An implementation of non-negative matrix factorization.  This can be used to decompose an input dataset into two low-rank non-negative components. [Detailed documentation](#nmf_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `initial_h` | [`matrix`](#doc_matrix) | Initial H matrix. | `np.empty([0, 0])` |
| `initial_w` | [`matrix`](#doc_matrix) | Initial W matrix. | `np.empty([0, 0])` |
| `input_` | [`matrix`](#doc_matrix) | Input dataset to perform NMF on. | `**--**` |
| `max_iterations` | [`int`](#doc_int) | Number of iterations before NMF terminates (0 runs until convergence. | `10000` |
| `min_residue` | [`float`](#doc_float) | The minimum root mean square residue allowed for each iteration, below which the program terminates. | `1e-05` |
| `rank` | [`int`](#doc_int) | Rank of the factorization. | `**--**` |
| `seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `update_rules` | [`str`](#doc_str) | Update rules for each iteration; ( multdist \| multdiv \| als ). | `'multdist'` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `h` | [`matrix`](#doc_matrix) | Matrix to save the calculated H to. | 
| `w` | [`matrix`](#doc_matrix) | Matrix to save the calculated W to. | 

### Detailed documentation
{: #nmf_detailed-documentation }

This program performs non-negative matrix factorization on the given dataset, storing the resulting decomposed matrices in the specified files.  For an input dataset V, NMF decomposes V into two matrices W and H such that 

V = W * H

where all elements in W and H are non-negative.  If V is of size (n x m), then W will be of size (n x r) and H will be of size (r x m), where r is the rank of the factorization (specified by the `rank` parameter).

Optionally, the desired update rules for each NMF iteration can be chosen from the following list:

 - multdist: multiplicative distance-based update rules (Lee and Seung 1999)
 - multdiv: multiplicative divergence-based update rules (Lee and Seung 1999)
 - als: alternating least squares update rules (Paatero and Tapper 1994)

The maximum number of iterations is specified with `max_iterations`, and the minimum residue required for algorithm termination is specified with the `min_residue` parameter.

### Example
For example, to run NMF on the input matrix `'V'` using the 'multdist' update rules with a rank-10 decomposition and storing the decomposed matrices into `'W'` and `'H'`, the following command could be used: 

```python
>>> output = nmf(input_=V, rank=10, update_rules='multdist')
>>> W = output['w']
>>> H = output['h']
```

### See also

 - [cf()](#cf)
 - [Non-negative matrix factorization on Wikipedia](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization)
 - [Algorithms for non-negative matrix factorization (pdf)](https://proceedings.neurips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf)
 - [NMF C++ class documentation](../../user/methods/nmf.md)
 - [AMF C++ class documentation](../../user/methods/amf.md)

## pca()
{: #pca }

#### Principal Components Analysis
{: #pca_descr }

```python
>>> from mlpack import pca
>>> d = pca(check_input_matrices=False, copy_all_inputs=False,
        decomposition_method='exact', input_=np.empty([0, 0]),
        new_dimensionality=0, scale=False, var_to_retain=0, verbose=False)
>>> output = d['output']
```

An implementation of several strategies for principal components analysis (PCA), a common preprocessing step.  Given a dataset and a desired new dimensionality, this can reduce the dimensionality of the data using the linear transformation determined by PCA. [Detailed documentation](#pca_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `decomposition_method` | [`str`](#doc_str) | Method used for the principal components analysis: 'exact', 'randomized', 'randomized-block-krylov', 'quic'. | `'exact'` |
| `input_` | [`matrix`](#doc_matrix) | Input dataset to perform PCA on. | `**--**` |
| `new_dimensionality` | [`int`](#doc_int) | Desired dimensionality of output dataset. If 0, no dimensionality reduction is performed. | `0` |
| `scale` | [`bool`](#doc_bool) | If set, the data will be scaled before running PCA, such that the variance of each feature is 1. | `False` |
| `var_to_retain` | [`float`](#doc_float) | Amount of variance to retain; should be between 0 and 1.  If 1, all variance is retained.  Overrides -d. | `0` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output` | [`matrix`](#doc_matrix) | Matrix to save modified dataset to. | 

### Detailed documentation
{: #pca_detailed-documentation }

This program performs principal components analysis on the given dataset using the exact, randomized, randomized block Krylov, or QUIC SVD method. It will transform the data onto its principal components, optionally performing dimensionality reduction by ignoring the principal components with the smallest eigenvalues.

Use the `input_` parameter to specify the dataset to perform PCA on.  A desired new dimensionality can be specified with the `new_dimensionality` parameter, or the desired variance to retain can be specified with the `var_to_retain` parameter.  If desired, the dataset can be scaled before running PCA with the `scale` parameter.

Multiple different decomposition techniques can be used.  The method to use can be specified with the `decomposition_method` parameter, and it may take the values 'exact', 'randomized', or 'quic'.

### Example
For example, to reduce the dimensionality of the matrix `'data'` to 5 dimensions using randomized SVD for the decomposition, storing the output matrix to `'data_mod'`, the following command can be used:

```python
>>> output = pca(input_=data, new_dimensionality=5,
  decomposition_method='randomized')
>>> data_mod = output['output']
```

### See also

 - [Principal component analysis on Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)
 - [PCA C++ class documentation](../../user/methods/pca.md)

## perceptron()
{: #perceptron }

#### Perceptron
{: #perceptron_descr }

```python
>>> from mlpack import perceptron
>>> d = perceptron(check_input_matrices=False, copy_all_inputs=False,
        input_model=None, labels=np.empty([0], dtype=np.uint64),
        max_iterations=1000, test=np.empty([0, 0]), training=np.empty([0, 0]),
        verbose=False)
>>> output_model = d['output_model']
>>> predictions = d['predictions']
```

An implementation of a perceptron---a single level neural network--=for classification.  Given labeled data, a perceptron can be trained and saved for future use; or, a pre-trained perceptron can be used for classification on new points. [Detailed documentation](#perceptron_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `input_model` | [`PerceptronModelType`](#doc_model) | Input perceptron model. | `None` |
| `labels` | [`int vector`](#doc_int_vector) | A matrix containing labels for the training set. | `np.empty([0], dtype=np.uint64)` |
| `max_iterations` | [`int`](#doc_int) | The maximum number of iterations the perceptron is to be run | `1000` |
| `test` | [`matrix`](#doc_matrix) | A matrix containing the test set. | `np.empty([0, 0])` |
| `training` | [`matrix`](#doc_matrix) | A matrix containing the training set. | `np.empty([0, 0])` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output_model` | [`PerceptronModelType`](#doc_model) | Output for trained perceptron model. | 
| `predictions` | [`int vector`](#doc_int_vector) | The matrix in which the predicted labels for the test set will be written. | 

### Detailed documentation
{: #perceptron_detailed-documentation }

This program implements a perceptron, which is a single level neural network. The perceptron makes its predictions based on a linear predictor function combining a set of weights with the feature vector.  The perceptron learning rule is able to converge, given enough iterations (specified using the `max_iterations` parameter), if the data supplied is linearly separable.  The perceptron is parameterized by a matrix of weight vectors that denote the numerical weights of the neural network.

This program allows loading a perceptron from a model (via the `input_model` parameter) or training a perceptron given training data (via the `training` parameter), or both those things at once.  In addition, this program allows classification on a test dataset (via the `test` parameter) and the classification results on the test set may be saved with the `predictions` output parameter.  The perceptron model may be saved with the `output_model` output parameter.

### Example
The training data given with the `training` option may have class labels as its last dimension (so, if the training data is in CSV format, labels should be the last column).  Alternately, the `labels` parameter may be used to specify a separate matrix of labels.

All these options make it easy to train a perceptron, and then re-use that perceptron for later classification.  The invocation below trains a perceptron on `'training_data'` with labels `'training_labels'`, and saves the model to `'perceptron_model'`.

```python
>>> output = perceptron(training=training_data, labels=training_labels)
>>> perceptron_model = output['output_model']
```

Then, this model can be re-used for classification on the test data `'test_data'`.  The example below does precisely that, saving the predicted classes to `'predictions'`.

```python
>>> output = perceptron(input_model=perceptron_model, test=test_data)
>>> predictions = output['predictions']
```

Note that all of the options may be specified at once: predictions may be calculated right after training a model, and model training can occur even if an existing perceptron model is passed with the `input_model` parameter.  However, note that the number of classes and the dimensionality of all data must match.  So you cannot pass a perceptron model trained on 2 classes and then re-train with a 4-class dataset.  Similarly, attempting classification on a 3-dimensional dataset with a perceptron that has been trained on 8 dimensions will cause an error.

### See also

 - [adaboost()](#adaboost)
 - [Perceptron on Wikipedia](https://en.wikipedia.org/wiki/Perceptron)
 - [Perceptron C++ class documentation](../../user/methods/perceptron.md)

## preprocess_split()
{: #preprocess_split }

#### Split Data
{: #preprocess_split_descr }

```python
>>> from mlpack import preprocess_split
>>> d = preprocess_split(check_input_matrices=False,
        copy_all_inputs=False, input_=np.empty([0, 0]),
        input_labels=np.empty([0, 0], dtype=np.uint64), no_shuffle=False,
        seed=0, stratify_data=False, test_ratio=0.2, verbose=False)
>>> test = d['test']
>>> test_labels = d['test_labels']
>>> training = d['training']
>>> training_labels = d['training_labels']
```

A utility to split data into a training and testing dataset.  This can also split labels according to the same split. [Detailed documentation](#preprocess_split_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `input_` | [`matrix`](#doc_matrix) | Matrix containing data. | `**--**` |
| `input_labels` | [`int matrix`](#doc_int_matrix) | Matrix containing labels. | `np.empty([0, 0], dtype=np.uint64)` |
| `no_shuffle` | [`bool`](#doc_bool) | Avoid shuffling the data before splitting. | `False` |
| `seed` | [`int`](#doc_int) | Random seed (0 for std::time(NULL)). | `0` |
| `stratify_data` | [`bool`](#doc_bool) | Stratify the data according to labels | `False` |
| `test_ratio` | [`float`](#doc_float) | Ratio of test set; if not set,the ratio defaults to 0.2 | `0.2` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `test` | [`matrix`](#doc_matrix) | Matrix to save test data to. | 
| `test_labels` | [`int matrix`](#doc_int_matrix) | Matrix to save test labels to. | 
| `training` | [`matrix`](#doc_matrix) | Matrix to save training data to. | 
| `training_labels` | [`int matrix`](#doc_int_matrix) | Matrix to save train labels to. | 

### Detailed documentation
{: #preprocess_split_detailed-documentation }

This utility takes a dataset and optionally labels and splits them into a training set and a test set. Before the split, the points in the dataset are randomly reordered. The percentage of the dataset to be used as the test set can be specified with the `test_ratio` parameter; the default is 0.2 (20%).

The output training and test matrices may be saved with the `training` and `test` output parameters.

Optionally, labels can also be split along with the data by specifying the `input_labels` parameter.  Splitting labels works the same way as splitting the data. The output training and test labels may be saved with the `training_labels` and `test_labels` output parameters, respectively.

### Example
So, a simple example where we want to split the dataset `'X'` into `'X_train'` and `'X_test'` with 60% of the data in the training set and 40% of the dataset in the test set, we could run 

```python
>>> output = preprocess_split(input_=X, test_ratio=0.4)
>>> X_train = output['training']
>>> X_test = output['test']
```

Also by default the dataset is shuffled and split; you can provide the `no_shuffle` option to avoid shuffling the data; an example to avoid shuffling of data is:

```python
>>> output = preprocess_split(input_=X, test_ratio=0.4, no_shuffle=True)
>>> X_train = output['training']
>>> X_test = output['test']
```

If we had a dataset `'X'` and associated labels `'y'`, and we wanted to split these into `'X_train'`, `'y_train'`, `'X_test'`, and `'y_test'`, with 30% of the data in the test set, we could run

```python
>>> output = preprocess_split(input_=X, input_labels=y, test_ratio=0.3)
>>> X_train = output['training']
>>> y_train = output['training_labels']
>>> X_test = output['test']
>>> y_test = output['test_labels']
```

To maintain the ratio of each class in the train and test sets, the`stratify_data` option can be used.

```python
>>> output = preprocess_split(input_=X, test_ratio=0.4, stratify_data=True)
>>> X_train = output['training']
>>> X_test = output['test']
```

### See also

 - [preprocess_binarize()](#preprocess_binarize)
 - [preprocess_describe()](#preprocess_describe)

## preprocess_binarize()
{: #preprocess_binarize }

#### Binarize Data
{: #preprocess_binarize_descr }

```python
>>> from mlpack import preprocess_binarize
>>> d = preprocess_binarize(check_input_matrices=False,
        copy_all_inputs=False, dimension=0, input_=np.empty([0, 0]),
        threshold=0, verbose=False)
>>> output = d['output']
```

A utility to binarize a dataset.  Given a dataset, this utility converts each value in the desired dimension(s) to 0 or 1; this can be a useful preprocessing step. [Detailed documentation](#preprocess_binarize_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `dimension` | [`int`](#doc_int) | Dimension to apply the binarization. If not set, the program will binarize every dimension by default. | `0` |
| `input_` | [`matrix`](#doc_matrix) | Input data matrix. | `**--**` |
| `threshold` | [`float`](#doc_float) | Threshold to be applied for binarization. If not set, the threshold defaults to 0.0. | `0` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output` | [`matrix`](#doc_matrix) | Matrix in which to save the output. | 

### Detailed documentation
{: #preprocess_binarize_detailed-documentation }

This utility takes a dataset and binarizes the variables into either 0 or 1 given threshold. User can apply binarization on a dimension or the whole dataset.  The dimension to apply binarization to can be specified using the `dimension` parameter; if left unspecified, every dimension will be binarized.  The threshold for binarization can also be specified with the `threshold` parameter; the default threshold is 0.0.

The binarized matrix may be saved with the `output` output parameter.

### Example
For example, if we want to set all variables greater than 5 in the dataset `'X'` to 1 and variables less than or equal to 5.0 to 0, and save the result to `'Y'`, we could run

```python
>>> output = preprocess_binarize(input_=X, threshold=5)
>>> Y = output['output']
```

But if we want to apply this to only the first (0th) dimension of `'X'`,  we could instead run

```python
>>> output = preprocess_binarize(input_=X, threshold=5, dimension=0)
>>> Y = output['output']
```

### See also

 - [preprocess_describe()](#preprocess_describe)
 - [preprocess_split()](#preprocess_split)

## preprocess_describe()
{: #preprocess_describe }

#### Descriptive Statistics
{: #preprocess_describe_descr }

```python
>>> from mlpack import preprocess_describe
>>> preprocess_describe(check_input_matrices=False,
        copy_all_inputs=False, dimension=0, input_=np.empty([0, 0]),
        population=False, precision=4, row_major=False, verbose=False, width=8)
```

A utility for printing descriptive statistics about a dataset.  This prints a number of details about a dataset in a tabular format. [Detailed documentation](#preprocess_describe_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `dimension` | [`int`](#doc_int) | Dimension of the data. Use this to specify a dimension | `0` |
| `input_` | [`matrix`](#doc_matrix) | Matrix containing data, | `**--**` |
| `population` | [`bool`](#doc_bool) | If specified, the program will calculate statistics assuming the dataset is the population. By default, the program will assume the dataset as a sample. | `False` |
| `precision` | [`int`](#doc_int) | Precision of the output statistics. | `4` |
| `row_major` | [`bool`](#doc_bool) | If specified, the program will calculate statistics across rows, not across columns.  (Remember that in mlpack, a column represents a point, so this option is generally not necessary.) | `False` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |
| `width` | [`int`](#doc_int) | Width of the output table. | `8` |


### Detailed documentation
{: #preprocess_describe_detailed-documentation }

This utility takes a dataset and prints out the descriptive statistics of the data. Descriptive statistics is the discipline of quantitatively describing the main features of a collection of information, or the quantitative description itself. The program does not modify the original file, but instead prints out the statistics to the console. The printed result will look like a table.

Optionally, width and precision of the output can be adjusted by a user using the `width` and `precision` parameters. A user can also select a specific dimension to analyze if there are too many dimensions. The `population` parameter can be specified when the dataset should be considered as a population.  Otherwise, the dataset will be considered as a sample.

### Example
So, a simple example where we want to print out statistical facts about the dataset `'X'` using the default settings, we could run 

```python
>>> preprocess_describe(input_=X, verbose=True)
```

If we want to customize the width to 10 and precision to 5 and consider the dataset as a population, we could run

```python
>>> preprocess_describe(input_=X, width=10, precision=5, verbose=True)
```

### See also

 - [preprocess_binarize()](#preprocess_binarize)
 - [preprocess_split()](#preprocess_split)

## preprocess_scale()
{: #preprocess_scale }

#### Scale Data
{: #preprocess_scale_descr }

```python
>>> from mlpack import preprocess_scale
>>> d = preprocess_scale(check_input_matrices=False,
        copy_all_inputs=False, epsilon=1e-06, input_=np.empty([0, 0]),
        input_model=None, inverse_scaling=False, max_value=1, min_value=0,
        scaler_method='standard_scaler', seed=0, verbose=False)
>>> output = d['output']
>>> output_model = d['output_model']
```

A utility to perform feature scaling on datasets using one of sixtechniques.  Both scaling and inverse scaling are supported, andscalers can be saved and then applied to other datasets. [Detailed documentation](#preprocess_scale_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `epsilon` | [`float`](#doc_float) | regularization Parameter for pcawhitening, or zcawhitening, should be between -1 to 1. | `1e-06` |
| `input_` | [`matrix`](#doc_matrix) | Matrix containing data. | `**--**` |
| `input_model` | [`ScalingModelType`](#doc_model) | Input Scaling model. | `None` |
| `inverse_scaling` | [`bool`](#doc_bool) | Inverse Scaling to get original dataset | `False` |
| `max_value` | [`int`](#doc_int) | Ending value of range for min_max_scaler. | `1` |
| `min_value` | [`int`](#doc_int) | Starting value of range for min_max_scaler. | `0` |
| `scaler_method` | [`str`](#doc_str) | method to use for scaling, the default is standard_scaler. | `'standard_scaler'` |
| `seed` | [`int`](#doc_int) | Random seed (0 for std::time(NULL)). | `0` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output` | [`matrix`](#doc_matrix) | Matrix to save scaled data to. | 
| `output_model` | [`ScalingModelType`](#doc_model) | Output scaling model. | 

### Detailed documentation
{: #preprocess_scale_detailed-documentation }

This utility takes a dataset and performs feature scaling using one of the six scaler methods namely: 'max_abs_scaler', 'mean_normalization', 'min_max_scaler' ,'standard_scaler', 'pca_whitening' and 'zca_whitening'. The function takes a matrix as `input_` and a scaling method type which you can specify using `scaler_method` parameter; the default is standard scaler, and outputs a matrix with scaled feature.

The output scaled feature matrix may be saved with the `output` output parameters.

The model to scale features can be saved using `output_model` and later can be loaded back using`input_model`.

### Example
So, a simple example where we want to scale the dataset `'X'` into `'X_scaled'` with  standard_scaler as scaler_method, we could run 

```python
>>> output = preprocess_scale(input_=X, scaler_method='standard_scaler')
>>> X_scaled = output['output']
```

A simple example where we want to whiten the dataset `'X'` into `'X_whitened'` with  PCA as whitening_method and use 0.01 as regularization parameter, we could run 

```python
>>> output = preprocess_scale(input_=X, scaler_method='pca_whitening',
  epsilon=0.01)
>>> X_scaled = output['output']
```

You can also retransform the scaled dataset back using`inverse_scaling`. An example to rescale : `'X_scaled'` into `'X'`using the saved model `input_model` is:

```python
>>> output = preprocess_scale(input_=X_scaled, inverse_scaling=True,
  input_model=saved)
>>> X = output['output']
```

Another simple example where we want to scale the dataset `'X'` into `'X_scaled'` with  min_max_scaler as scaler method, where scaling range is 1 to 3 instead of default 0 to 1. We could run 

```python
>>> output = preprocess_scale(input_=X, scaler_method='min_max_scaler',
  min_value=1, max_value=3)
>>> X_scaled = output['output']
```

### See also

 - [preprocess_binarize()](#preprocess_binarize)
 - [preprocess_describe()](#preprocess_describe)

## preprocess_one_hot_encoding()
{: #preprocess_one_hot_encoding }

#### One Hot Encoding
{: #preprocess_one_hot_encoding_descr }

```python
>>> from mlpack import preprocess_one_hot_encoding
>>> d = preprocess_one_hot_encoding(check_input_matrices=False,
        copy_all_inputs=False, dimensions=[], input_=np.empty([0, 0]),
        verbose=False)
>>> output = d['output']
```

A utility to do one-hot encoding on features of dataset. [Detailed documentation](#preprocess_one_hot_encoding_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `dimensions` | [`list of ints`](#doc_list_of_ints) | Index of dimensions that need to be one-hot encoded (if unspecified, all categorical dimensions are one-hot encoded). | `[]` |
| `input_` | [`categorical matrix`](#doc_categorical_matrix) | Matrix containing data. | `**--**` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output` | [`matrix`](#doc_matrix) | Matrix to save one-hot encoded features data to. | 

### Detailed documentation
{: #preprocess_one_hot_encoding_detailed-documentation }

This utility takes a dataset and a vector of indices and does one-hot encoding of the respective features at those indices. Indices represent the IDs of the dimensions to be one-hot encoded.

If no dimensions are specified with `dimensions`, then all categorical-type dimensions will be one-hot encoded. Otherwise, only the dimensions given in `dimensions` will be one-hot encoded.

The output matrix with encoded features may be saved with the `output` parameters.

### Example
So, a simple example where we want to encode 1st and 3rd feature from dataset `'X'` into `'X_output'` would be

```python
>>> output = preprocess_one_hot_encoding(input_=X, dimensions=1,
  dimensions=3)
>>> X_ouput = output['output']
```

### See also

 - [preprocess_binarize()](#preprocess_binarize)
 - [preprocess_describe()](#preprocess_describe)
 - [One-hot encoding on Wikipedia](https://en.m.wikipedia.org/wiki/One-hot)

## radical()
{: #radical }

#### RADICAL
{: #radical_descr }

```python
>>> from mlpack import radical
>>> d = radical(angles=150, check_input_matrices=False,
        copy_all_inputs=False, input_=np.empty([0, 0]), noise_std_dev=0.175,
        objective=False, replicates=30, seed=0, sweeps=0, verbose=False)
>>> output_ic = d['output_ic']
>>> output_unmixing = d['output_unmixing']
```

An implementation of RADICAL, a method for independent component analysis (ICA).  Given a dataset, this can decompose the dataset into an unmixing matrix and an independent component matrix; this can be useful for preprocessing. [Detailed documentation](#radical_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `angles` | [`int`](#doc_int) | Number of angles to consider in brute-force search during Radical2D. | `150` |
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `input_` | [`matrix`](#doc_matrix) | Input dataset for ICA. | `**--**` |
| `noise_std_dev` | [`float`](#doc_float) | Standard deviation of Gaussian noise. | `0.175` |
| `objective` | [`bool`](#doc_bool) | If set, an estimate of the final objective function is printed. | `False` |
| `replicates` | [`int`](#doc_int) | Number of Gaussian-perturbed replicates to use (per point) in Radical2D. | `30` |
| `seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `sweeps` | [`int`](#doc_int) | Number of sweeps; each sweep calls Radical2D once for each pair of dimensions. | `0` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output_ic` | [`matrix`](#doc_matrix) | Matrix to save independent components to. | 
| `output_unmixing` | [`matrix`](#doc_matrix) | Matrix to save unmixing matrix to. | 

### Detailed documentation
{: #radical_detailed-documentation }

An implementation of RADICAL, a method for independent component analysis (ICA).  Assuming that we have an input matrix X, the goal is to find a square unmixing matrix W such that Y = W * X and the dimensions of Y are independent components.  If the algorithm is running particularly slowly, try reducing the number of replicates.

The input matrix to perform ICA on should be specified with the `input_` parameter.  The output matrix Y may be saved with the `output_ic` output parameter, and the output unmixing matrix W may be saved with the `output_unmixing` output parameter.

### Example
For example, to perform ICA on the matrix `'X'` with 40 replicates, saving the independent components to `'ic'`, the following command may be used: 

```python
>>> output = radical(input_=X, replicates=40)
>>> ic = output['output_ic']
```

### See also

 - [Independent component analysis on Wikipedia](https://en.wikipedia.org/wiki/Independent_component_analysis)
 - [ICA using spacings estimates of entropy (pdf)](https://www.jmlr.org/papers/volume4/learned-miller03a/learned-miller03a.pdf)
 - [Radical C++ class documentation](../../user/methods/radical.md)

## random_forest()
{: #random_forest }

#### Random forests
{: #random_forest_descr }

```python
>>> from mlpack import random_forest
>>> d = random_forest(check_input_matrices=False, copy_all_inputs=False,
        input_model=None, labels=np.empty([0], dtype=np.uint64),
        maximum_depth=0, minimum_gain_split=0, minimum_leaf_size=1,
        num_trees=10, print_training_accuracy=False, seed=0, subspace_dim=0,
        test=np.empty([0, 0]), test_labels=np.empty([0], dtype=np.uint64),
        training=np.empty([0, 0]), verbose=False, warm_start=False)
>>> output_model = d['output_model']
>>> predictions = d['predictions']
>>> probabilities = d['probabilities']
```

An implementation of the standard random forest algorithm by Leo Breiman for classification.  Given labeled data, a random forest can be trained and saved for future use; or, a pre-trained random forest can be used for classification. [Detailed documentation](#random_forest_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `input_model` | [`RandomForestModelType`](#doc_model) | Pre-trained random forest to use for classification. | `None` |
| `labels` | [`int vector`](#doc_int_vector) | Labels for training dataset. | `np.empty([0], dtype=np.uint64)` |
| `maximum_depth` | [`int`](#doc_int) | Maximum depth of the tree (0 means no limit). | `0` |
| `minimum_gain_split` | [`float`](#doc_float) | Minimum gain needed to make a split when building a tree. | `0` |
| `minimum_leaf_size` | [`int`](#doc_int) | Minimum number of points in each leaf node. | `1` |
| `num_trees` | [`int`](#doc_int) | Number of trees in the random forest. | `10` |
| `print_training_accuracy` | [`bool`](#doc_bool) | If set, then the accuracy of the model on the training set will be predicted (verbose must also be specified). | `False` |
| `seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `subspace_dim` | [`int`](#doc_int) | Dimensionality of random subspace to use for each split.  '0' will autoselect the square root of data dimensionality. | `0` |
| `test` | [`matrix`](#doc_matrix) | Test dataset to produce predictions for. | `np.empty([0, 0])` |
| `test_labels` | [`int vector`](#doc_int_vector) | Test dataset labels, if accuracy calculation is desired. | `np.empty([0], dtype=np.uint64)` |
| `training` | [`matrix`](#doc_matrix) | Training dataset. | `np.empty([0, 0])` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |
| `warm_start` | [`bool`](#doc_bool) | If true and passed along with `training` and `input_model` then trains more trees on top of existing model. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output_model` | [`RandomForestModelType`](#doc_model) | Model to save trained random forest to. | 
| `predictions` | [`int vector`](#doc_int_vector) | Predicted classes for each point in the test set. | 
| `probabilities` | [`matrix`](#doc_matrix) | Predicted class probabilities for each point in the test set. | 

### Detailed documentation
{: #random_forest_detailed-documentation }

This program is an implementation of the standard random forest classification algorithm by Leo Breiman.  A random forest can be trained and saved for later use, or a random forest may be loaded and predictions or class probabilities for points may be generated.

The training set and associated labels are specified with the `training` and `labels` parameters, respectively.  The labels should be in the range `[0, num_classes - 1]`. Optionally, if `labels` is not specified, the labels are assumed to be the last dimension of the training dataset.

When a model is trained, the `output_model` output parameter may be used to save the trained model.  A model may be loaded for predictions with the `input_model`parameter. The `input_model` parameter may not be specified when the `training` parameter is specified.  The `minimum_leaf_size` parameter specifies the minimum number of training points that must fall into each leaf for it to be split.  The `num_trees` controls the number of trees in the random forest.  The `minimum_gain_split` parameter controls the minimum required gain for a decision tree node to split.  Larger values will force higher-confidence splits.  The `maximum_depth` parameter specifies the maximum depth of the tree.  The `subspace_dim` parameter is used to control the number of random dimensions chosen for an individual node's split.  If `print_training_accuracy` is specified, the calculated accuracy on the training set will be printed.

Test data may be specified with the `test` parameter, and if performance measures are desired for that test set, labels for the test points may be specified with the `test_labels` parameter.  Predictions for each test point may be saved via the `predictions`output parameter.  Class probabilities for each prediction may be saved with the `probabilities` output parameter.

### Example
For example, to train a random forest with a minimum leaf size of 20 using 10 trees on the dataset contained in `'data'`with labels `'labels'`, saving the output random forest to `'rf_model'` and printing the training error, one could call

```python
>>> output = random_forest(training=data, labels=labels, minimum_leaf_size=20,
  num_trees=10, print_training_accuracy=True)
>>> rf_model = output['output_model']
```

Then, to use that model to classify points in `'test_set'` and print the test error given the labels `'test_labels'` using that model, while saving the predictions for each point to `'predictions'`, one could call 

```python
>>> output = random_forest(input_model=rf_model, test=test_set,
  test_labels=test_labels)
>>> predictions = output['predictions']
```

### See also

 - [decision_tree()](#decision_tree)
 - [hoeffding_tree()](#hoeffding_tree)
 - [softmax_regression()](#softmax_regression)
 - [Random forest on Wikipedia](https://en.wikipedia.org/wiki/Random_forest)
 - [Random forests (pdf)](https://www.eecis.udel.edu/~shatkay/Course/papers/BreimanRandomForests2001.pdf)
 - [RandomForest C++ class documentation](../../user/methods/random_forest.md)

## krann()
{: #krann }

#### K-Rank-Approximate-Nearest-Neighbors (kRANN)
{: #krann_descr }

```python
>>> from mlpack import krann
>>> d = krann(alpha=0.95, check_input_matrices=False,
        copy_all_inputs=False, first_leaf_exact=False, input_model=None, k=0,
        leaf_size=20, naive=False, query=np.empty([0, 0]), random_basis=False,
        reference=np.empty([0, 0]), sample_at_leaves=False, seed=0,
        single_mode=False, single_sample_limit=20, tau=5, tree_type='kd',
        verbose=False)
>>> distances = d['distances']
>>> neighbors = d['neighbors']
>>> output_model = d['output_model']
```

An implementation of rank-approximate k-nearest-neighbor search (kRANN)  using single-tree and dual-tree algorithms.  Given a set of reference points and query points, this can find the k nearest neighbors in the reference set of each query point using trees; trees that are built can be saved for future use. [Detailed documentation](#krann_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `alpha` | [`float`](#doc_float) | The desired success probability. | `0.95` |
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `first_leaf_exact` | [`bool`](#doc_bool) | The flag to trigger sampling only after exactly exploring the first leaf. | `False` |
| `input_model` | [`RAModelType`](#doc_model) | Pre-trained kNN model. | `None` |
| `k` | [`int`](#doc_int) | Number of nearest neighbors to find. | `0` |
| `leaf_size` | [`int`](#doc_int) | Leaf size for tree building (used for kd-trees, UB trees, R trees, R* trees, X trees, Hilbert R trees, R+ trees, R++ trees, and octrees). | `20` |
| `naive` | [`bool`](#doc_bool) | If true, sampling will be done without using a tree. | `False` |
| `query` | [`matrix`](#doc_matrix) | Matrix containing query points (optional). | `np.empty([0, 0])` |
| `random_basis` | [`bool`](#doc_bool) | Before tree-building, project the data onto a random orthogonal basis. | `False` |
| `reference` | [`matrix`](#doc_matrix) | Matrix containing the reference dataset. | `np.empty([0, 0])` |
| `sample_at_leaves` | [`bool`](#doc_bool) | The flag to trigger sampling at leaves. | `False` |
| `seed` | [`int`](#doc_int) | Random seed (if 0, std::time(NULL) is used). | `0` |
| `single_mode` | [`bool`](#doc_bool) | If true, single-tree search is used (as opposed to dual-tree search. | `False` |
| `single_sample_limit` | [`int`](#doc_int) | The limit on the maximum number of samples (and hence the largest node you can approximate). | `20` |
| `tau` | [`float`](#doc_float) | The allowed rank-error in terms of the percentile of the data. | `5` |
| `tree_type` | [`str`](#doc_str) | Type of tree to use: 'kd', 'ub', 'cover', 'r', 'x', 'r-star', 'hilbert-r', 'r-plus', 'r-plus-plus', 'oct'. | `'kd'` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `distances` | [`matrix`](#doc_matrix) | Matrix to output distances into. | 
| `neighbors` | [`int matrix`](#doc_int_matrix) | Matrix to output neighbors into. | 
| `output_model` | [`RAModelType`](#doc_model) | If specified, the kNN model will be output here. | 

### Detailed documentation
{: #krann_detailed-documentation }

This program will calculate the k rank-approximate-nearest-neighbors of a set of points. You may specify a separate set of reference points and query points, or just a reference set which will be used as both the reference and query set. You must specify the rank approximation (in %) (and optionally the success probability).

### Example
For example, the following will return 5 neighbors from the top 0.1% of the data (with probability 0.95) for each point in `'input'` and store the distances in `'distances'` and the neighbors in `'neighbors.csv'`:

```python
>>> output = krann(reference=input, k=5, tau=0.1)
>>> distances = output['distances']
>>> neighbors = output['neighbors']
```

Note that tau must be set such that the number of points in the corresponding percentile of the data is greater than k.  Thus, if we choose tau = 0.1 with a dataset of 1000 points and k = 5, then we are attempting to choose 5 nearest neighbors out of the closest 1 point -- this is invalid and the program will terminate with an error message.

The output matrices are organized such that row i and column j in the neighbors output file corresponds to the index of the point in the reference set which is the i'th nearest neighbor from the point in the query set with index j.  Row i and column j in the distances output file corresponds to the distance between those two points.

### See also

 - [knn()](#knn)
 - [lsh()](#lsh)
 - [Rank-approximate nearest neighbor search: Retaining meaning and speed in high dimensions (pdf)](https://proceedings.neurips.cc/paper_files/paper/2009/file/ddb30680a691d157187ee1cf9e896d03-Paper.pdf)
 - [RASearch C++ class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/rann/ra_search.hpp)

## softmax_regression()
{: #softmax_regression }

#### Softmax Regression
{: #softmax_regression_descr }

```python
>>> from mlpack import softmax_regression
>>> d = softmax_regression(check_input_matrices=False,
        copy_all_inputs=False, input_model=None, labels=np.empty([0],
        dtype=np.uint64), lambda_=0.0001, max_iterations=400,
        no_intercept=False, number_of_classes=0, test=np.empty([0, 0]),
        test_labels=np.empty([0], dtype=np.uint64), training=np.empty([0, 0]),
        verbose=False)
>>> output_model = d['output_model']
>>> predictions = d['predictions']
>>> probabilities = d['probabilities']
```

An implementation of softmax regression for classification, which is a multiclass generalization of logistic regression.  Given labeled data, a softmax regression model can be trained and saved for future use, or, a pre-trained softmax regression model can be used for classification of new points. [Detailed documentation](#softmax_regression_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `input_model` | [`SoftmaxRegression<>Type`](#doc_model) | File containing existing model (parameters). | `None` |
| `labels` | [`int vector`](#doc_int_vector) | A matrix containing labels (0 or 1) for the points in the training set (y). The labels must order as a row. | `np.empty([0], dtype=np.uint64)` |
| `lambda_` | [`float`](#doc_float) | L2-regularization constant | `0.0001` |
| `max_iterations` | [`int`](#doc_int) | Maximum number of iterations before termination. | `400` |
| `no_intercept` | [`bool`](#doc_bool) | Do not add the intercept term to the model. | `False` |
| `number_of_classes` | [`int`](#doc_int) | Number of classes for classification; if unspecified (or 0), the number of classes found in the labels will be used. | `0` |
| `test` | [`matrix`](#doc_matrix) | Matrix containing test dataset. | `np.empty([0, 0])` |
| `test_labels` | [`int vector`](#doc_int_vector) | Matrix containing test labels. | `np.empty([0], dtype=np.uint64)` |
| `training` | [`matrix`](#doc_matrix) | A matrix containing the training set (the matrix of predictors, X). | `np.empty([0, 0])` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output_model` | [`SoftmaxRegression<>Type`](#doc_model) | File to save trained softmax regression model to. | 
| `predictions` | [`int vector`](#doc_int_vector) | Matrix to save predictions for test dataset into. | 
| `probabilities` | [`matrix`](#doc_matrix) | Matrix to save class probabilities for test dataset into. | 

### Detailed documentation
{: #softmax_regression_detailed-documentation }

This program performs softmax regression, a generalization of logistic regression to the multiclass case, and has support for L2 regularization.  The program is able to train a model, load  an existing model, and give predictions (and optionally their accuracy) for test data.

Training a softmax regression model is done by giving a file of training points with the `training` parameter and their corresponding labels with the `labels` parameter. The number of classes can be manually specified with the `number_of_classes` parameter, and the maximum number of iterations of the L-BFGS optimizer can be specified with the `max_iterations` parameter.  The L2 regularization constant can be specified with the `lambda_` parameter and if an intercept term is not desired in the model, the `no_intercept` parameter can be specified.

The trained model can be saved with the `output_model` output parameter. If training is not desired, but only testing is, a model can be loaded with the `input_model` parameter.  At the current time, a loaded model cannot be trained further, so specifying both `input_model` and `training` is not allowed.

The program is also able to evaluate a model on test data.  A test dataset can be specified with the `test` parameter. Class predictions can be saved with the `predictions` output parameter.  If labels are specified for the test data with the `test_labels` parameter, then the program will print the accuracy of the predictions on the given test set and its corresponding labels.

### Example
For example, to train a softmax regression model on the data `'dataset'` with labels `'labels'` with a maximum of 1000 iterations for training, saving the trained model to `'sr_model'`, the following command can be used: 

```python
>>> output = softmax_regression(training=dataset, labels=labels)
>>> sr_model = output['output_model']
```

Then, to use `'sr_model'` to classify the test points in `'test_points'`, saving the output predictions to `'predictions'`, the following command can be used:

```python
>>> output = softmax_regression(input_model=sr_model, test=test_points)
>>> predictions = output['predictions']
```

### See also

 - [logistic_regression()](#logistic_regression)
 - [random_forest()](#random_forest)
 - [Multinomial logistic regression (softmax regression) on Wikipedia](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)
 - [SoftmaxRegression C++ class documentation](../../user/methods/softmax_regression.md)

## sparse_coding()
{: #sparse_coding }

#### Sparse Coding
{: #sparse_coding_descr }

```python
>>> from mlpack import sparse_coding
>>> d = sparse_coding(atoms=15, check_input_matrices=False,
        copy_all_inputs=False, initial_dictionary=np.empty([0, 0]),
        input_model=None, lambda1=0, lambda2=0, max_iterations=0,
        newton_tolerance=1e-06, normalize=False, objective_tolerance=0.01,
        seed=0, test=np.empty([0, 0]), training=np.empty([0, 0]),
        verbose=False)
>>> codes = d['codes']
>>> dictionary = d['dictionary']
>>> output_model = d['output_model']
```

An implementation of Sparse Coding with Dictionary Learning.  Given a dataset, this will decompose the dataset into a sparse combination of a few dictionary elements, where the dictionary is learned during computation; a dictionary can be reused for future sparse coding of new points. [Detailed documentation](#sparse_coding_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `atoms` | [`int`](#doc_int) | Number of atoms in the dictionary. | `15` |
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `initial_dictionary` | [`matrix`](#doc_matrix) | Optional initial dictionary matrix. | `np.empty([0, 0])` |
| `input_model` | [`SparseCoding<>Type`](#doc_model) | File containing input sparse coding model. | `None` |
| `lambda1` | [`float`](#doc_float) | Sparse coding l1-norm regularization parameter. | `0` |
| `lambda2` | [`float`](#doc_float) | Sparse coding l2-norm regularization parameter. | `0` |
| `max_iterations` | [`int`](#doc_int) | Maximum number of iterations for sparse coding (0 indicates no limit). | `0` |
| `newton_tolerance` | [`float`](#doc_float) | Tolerance for convergence of Newton method. | `1e-06` |
| `normalize` | [`bool`](#doc_bool) | If set, the input data matrix will be normalized before coding. | `False` |
| `objective_tolerance` | [`float`](#doc_float) | Tolerance for convergence of the objective function. | `0.01` |
| `seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `test` | [`matrix`](#doc_matrix) | Optional matrix to be encoded by trained model. | `np.empty([0, 0])` |
| `training` | [`matrix`](#doc_matrix) | Matrix of training data (X). | `np.empty([0, 0])` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `codes` | [`matrix`](#doc_matrix) | Matrix to save the output sparse codes of the test matrix (--test_file) to. | 
| `dictionary` | [`matrix`](#doc_matrix) | Matrix to save the output dictionary to. | 
| `output_model` | [`SparseCoding<>Type`](#doc_model) | File to save trained sparse coding model to. | 

### Detailed documentation
{: #sparse_coding_detailed-documentation }

An implementation of Sparse Coding with Dictionary Learning, which achieves sparsity via an l1-norm regularizer on the codes (LASSO) or an (l1+l2)-norm regularizer on the codes (the Elastic Net).  Given a dense data matrix X with d dimensions and n points, sparse coding seeks to find a dense dictionary matrix D with k atoms in d dimensions, and a sparse coding matrix Z with n points in k dimensions.

The original data matrix X can then be reconstructed as Z * D.  Therefore, this program finds a representation of each point in X as a sparse linear combination of atoms in the dictionary D.

The sparse coding is found with an algorithm which alternates between a dictionary step, which updates the dictionary D, and a sparse coding step, which updates the sparse coding matrix.

Once a dictionary D is found, the sparse coding model may be used to encode other matrices, and saved for future usage.

To run this program, either an input matrix or an already-saved sparse coding model must be specified.  An input matrix may be specified with the `training` option, along with the number of atoms in the dictionary (specified with the `atoms` parameter).  It is also possible to specify an initial dictionary for the optimization, with the `initial_dictionary` parameter.  An input model may be specified with the `input_model` parameter.

### Example
As an example, to build a sparse coding model on the dataset `'data'` using 200 atoms and an l1-regularization parameter of 0.1, saving the model into `'model'`, use 

```python
>>> output = sparse_coding(training=data, atoms=200, lambda1=0.1)
>>> model = output['output_model']
```

Then, this model could be used to encode a new matrix, `'otherdata'`, and save the output codes to `'codes'`: 

```python
>>> output = sparse_coding(input_model=model, test=otherdata)
>>> codes = output['codes']
```

### See also

 - [local_coordinate_coding()](#local_coordinate_coding)
 - [Sparse dictionary learning on Wikipedia](https://en.wikipedia.org/wiki/Sparse_dictionary_learning)
 - [Efficient sparse coding algorithms (pdf)](https://proceedings.neurips.cc/paper_files/paper/2006/file/2d71b2ae158c7c5912cc0bbde2bb9d95-Paper.pdf)
 - [Regularization and variable selection via the elastic net](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=46217f372a75dddc2254fdbc6b9418ba3554e453)
 - [SparseCoding C++ class documentation](../../user/methods/sparse_coding.md)

## class Adaboost
{: #adaboost }

#### AdaBoost
{: #adaboost_descr }


This program implements the AdaBoost (or Adaptive Boosting) algorithm. The variant of AdaBoost implemented here is AdaBoost.MH. It uses a weak learner, either decision stumps or perceptrons, and over many iterations, creates a strong learner that is a weighted ensemble of weak learners. It runs these iterations until a tolerance value is crossed for change in the value of the weighted training error.

For more information about the algorithm, see the paper "Improved Boosting Algorithms Using Confidence-Rated Predictions", by R.E. Schapire and Y. Singer.
### Parameters

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `iterations` | [`int`](#doc_int) | The maximum number of boosting iterations to be run (0 will run until convergence.) | `1000` |
| `tolerance` | [`float`](#doc_float) | The tolerance for change in values of the weighted error during training. | `1e-10` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |
| `weak_learner` | [`str`](#doc_str) | The type of weak learner to use: 'decision_stump', or 'perceptron'. | `'decision_stump'` |

### Example

```python
>>> import pandas as pd
>>> from mlpack import preprocess_split
>>> from mlpack import Adaboost
>>> X = pd.read_csv('https://example.com')
>>> y = pd.read_csv('https://example.com')
>>> d = preprocess_split(input_=X, input_labels=y, test_ratio=0.2)
>>> X_train = d['training']
>>> y_train = d['training_labels']
>>> X_test = d['test']
>>> y_test = d['test_labels']
>>> model = Adaboost(check_input_matrices=False, copy_all_inputs=False,
  iterations=1000, tolerance=1e-10, verbose=False,
  weak_learner='decision_stump')
>>> output_model = model.fit(training=X_train, labels=y_train)
>>> predictions = model.predict(test=X_test)
>>> probabilities = model.predict_proba(test=X_test)
```

### Methods

| **name** | **description** |
|----------|-----------------|
| fit | Training AdaBoost model. |
| predict | Class predictions from model. |
| predict_proba | Class probabilities from model. |

### 1. fit

Training AdaBoost model.

#### Input Parameters:

| **name** | **type** | **description** |
|----------|----------|-----------------|
| `labels` | [`int vector`](#doc_int_vector) | Labels for the training set. | 
| `training` | [`matrix`](#doc_matrix) | Dataset for training AdaBoost. | 

#### Returns: 

| **type** | **description** |
|----------|-----------------|
| [`AdaBoostModelType`](#doc_model) | Output trained AdaBoost model. | 

### 2. predict

Class predictions from model.

#### Input Parameters:

| **name** | **type** | **description** |
|----------|----------|-----------------|
| `test` | [`matrix`](#doc_matrix) | Test dataset. | 

#### Returns: 

| **type** | **description** |
|----------|-----------------|
| [`int vector`](#doc_int_vector) | Predicted labels for the test set. | 

### 3. predict_proba

Class probabilities from model.

#### Input Parameters:

| **name** | **type** | **description** |
|----------|----------|-----------------|
| `test` | [`matrix`](#doc_matrix) | Test dataset. | 

#### Returns: 

| **type** | **description** |
|----------|-----------------|
| [`matrix`](#doc_matrix) | Predicted class probabilities for each point in the test set. | 

## class LinearRegression
{: #linear_regression }

#### Simple Linear Regression
{: #linear_regression_descr }


An implementation of simple linear regression and simple ridge regression using ordinary least squares. This solves the problem

  y = X * b + e
### Parameters

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `lambda_` | [`float`](#doc_float) | Tikhonov regularization for ridge regression.  If 0, the method reduces to linear regression. | `0` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |

### Example

```python
>>> import pandas as pd
>>> from mlpack import preprocess_split
>>> from mlpack import LinearRegression
>>> X = pd.read_csv('https://example.com')
>>> y = pd.read_csv('https://example.com')
>>> d = preprocess_split(input_=X, input_labels=y, test_ratio=0.2)
>>> X_train = d['training']
>>> y_train = d['training_labels']
>>> X_test = d['test']
>>> y_test = d['test_labels']
>>> lr = LinearRegression(check_input_matrices=False, copy_all_inputs=False,
  lambda_=0, verbose=False)
>>> output_model = lr.fit(training=X_train, training_responses=y_train)
>>> output_predictions = model.predict(test=X_test)
```

### Methods

| **name** | **description** |
|----------|-----------------|
| fit | Train a linear regression model. |
| predict | Predictions from model. |

### 1. fit

Train a linear regression model.

#### Input Parameters:

| **name** | **type** | **description** |
|----------|----------|-----------------|
| `training` | [`matrix`](#doc_matrix) | Matrix containing training set X (regressors). | 
| `training_responses` | [`vector`](#doc_vector) | Optional vector containing y (responses). If not given, the responses are assumed to be the last row of the input file. | 

#### Returns: 

| **type** | **description** |
|----------|-----------------|
| [`LinearRegression<>Type`](#doc_model) | Output LinearRegression model. | 

### 2. predict

Predictions from model.

#### Input Parameters:

| **name** | **type** | **description** |
|----------|----------|-----------------|
| `test` | [`matrix`](#doc_matrix) | Matrix containing X' (test regressors). | 

#### Returns: 

| **type** | **description** |
|----------|-----------------|
| [`vector`](#doc_vector) | If --test_file is specified, this matrix is where the predicted responses will be saved. | 

## image_converter()
{: #image_converter }

#### Image Converter
{: #image_converter_descr }

```python
>>> from mlpack import image_converter
>>> d = image_converter(channels=0, check_input_matrices=False,
        copy_all_inputs=False, dataset=np.empty([0, 0]), height=0, input_=[],
        quality=90, save=False, verbose=False, width=0)
>>> output = d['output']
```

A utility to load an image or set of images into a single dataset that can then be used by other mlpack methods and utilities. This can also unpack an image dataset into individual files, for instance after mlpack methods have been used. [Detailed documentation](#image_converter_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `channels` | [`int`](#doc_int) | Number of channels in the image. | `0` |
| `check_input_matrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `False` |
| `copy_all_inputs` | [`bool`](#doc_bool) | If specified, all input parameters will be deep copied before the method is run.  This is useful for debugging problems where the input parameters are being modified by the algorithm, but can slow down the code.  <span class="special">Only exists in Python binding.</span> | `False` |
| `dataset` | [`matrix`](#doc_matrix) | Input matrix to save as images. | `np.empty([0, 0])` |
| `height` | [`int`](#doc_int) | Height of the images. | `0` |
| `input_` | [`list of strs`](#doc_list_of_strs) | Image filenames which have to be loaded/saved. | `**--**` |
| `quality` | [`int`](#doc_int) | Compression of the image if saved as jpg (0-100). | `90` |
| `save` | [`bool`](#doc_bool) | Save a dataset as images. | `False` |
| `verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `False` |
| `width` | [`int`](#doc_int) | Width of the image. | `0` |

### Output options

Results are returned in a Python dictionary.  The keys of the dictionary are the names of the output parameters.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output` | [`matrix`](#doc_matrix) | Matrix to save images data to, Onlyneeded if you are specifying 'save' option. | 

### Detailed documentation
{: #image_converter_detailed-documentation }

This utility takes an image or an array of images and loads them to a matrix. You can optionally specify the height `height` width `width` and channel `channels` of the images that needs to be loaded; otherwise, these parameters will be automatically detected from the image.
There are other options too, that can be specified such as `quality`.

You can also provide a dataset and save them as images using `dataset` and `save` as an parameter.

### Example
 An example to load an image : 

```python
>>> output = image_converter(input_=X, height=256, width=256, channels=3)
>>> Y = output['output']
```

 An example to save an image is :

```python
>>> image_converter(input_=X, height=256, width=256, channels=3, dataset=Y,
  save=True)
```

### See also

 - [preprocess_binarize()](#preprocess_binarize)
 - [preprocess_describe()](#preprocess_describe)

