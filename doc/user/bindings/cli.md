# mlpack CLI binding documentation

## mlpack overview

mlpack is an intuitive, fast, and flexible header-only C++ machine learning library with bindings to other languages.  It aims to provide fast, lightweight implementations of both common and cutting-edge machine learning algorithms.

This reference page details mlpack's bindings to the command-line.

Further useful mlpack documentation links are given below.

 - [mlpack homepage](https://www.mlpack.org/)
 - [mlpack on Github](https://github.com/mlpack/mlpack)
 - [mlpack main documentation page](https://www.mlpack.org/doc/index.html)

See also the quickstart guide for CLI:

 - [CLI Quickstart](../../quickstart/cli.md)

## Data Formats

<div id="data-formats-div" markdown="1">
mlpack bindings for CLI take and return a restricted set of types, for simplicity.  These include primitive types, matrix/vector types, categorical matrix types, and model types. Each type is detailed below.

 - `int`{: #doc_int}: An integer (i.e., "1").
 - `double`{: #doc_double }: A floating-point number (i.e., "0.5").
 - `flag`{: #doc_flag }: A boolean flag option.  If not specified, it is false; if specified, it is true.
 - `string`{: #doc_string }: A character string (i.e., "hello").
 - `int vector`{: #doc_int_vector }: A vector of integers, separated by commas (i.e., "1,2,3").
 - `string vector`{: #doc_string_vector }: A vector of strings, separated by commas (i.e., "hello","goodbye").
 - `2-d matrix file`{: #doc_a_2_d_matrix_file }: A data matrix filename.  The file can be CSV (.csv), TSV (.csv), ASCII (space-separated values, .txt), Armadillo ASCII (.txt), PGM (.pgm), PPM (.ppm), Armadillo binary (.bin), or HDF5 (.h5, .hdf, .hdf5, or .he5), if mlpack was compiled with HDF5 support.  The type of the data is detected by the extension of the filename.  The storage should be such that one row corresponds to one point, and one column corresponds to one dimension (this is the typical storage format for on-disk data).  CSV files will be checked for a header; if no header is found, the first row will be loaded as a data point.  All values of the matrix will be loaded as double-precision floating point data.
 - `2-d index matrix file`{: #doc_a_2_d_index_matrix_file }: A data matrix filename, where the matrix holds only non-negative integer values.  This type is often used for labels or indices.  The file can be CSV (.csv), TSV (.csv), ASCII (space-separated values, .txt), Armadillo ASCII (.txt), PGM (.pgm), PPM (.ppm), Armadillo binary (.bin), or HDF5 (.h5, .hdf, .hdf5, or .he5), if mlpack was compiled with HDF5 support.  The type of the data is detected by the extension of the filename.  The storage should be such that one row corresponds to one point, and one column corresponds to one dimension (this is the typical storage format for on-disk data).  CSV files will be checked for a header; if no header is found, the first row will be loaded as a data point.  All values of the matrix will be loaded as unsigned integers.
 - `1-d matrix file`{: #doc_a_1_d_matrix_file }: A one-dimensional vector filename.  This file can take the same formats as the data matrix filenames; however, it must either contain one row and many columns, or one column and many rows.
 - `1-d index matrix file`{: #doc_a_1_d_index_matrix_file }: A one-dimensional vector filename, where the matrix holds only non-negative integer values.  This type is typically used for labels or predictions or other indices.  This file can take the same formats as the data matrix filenames; however, it must either contain one row and many columns, or one column and many rows.
 - `2-d categorical matrix file`{: #doc_a_2_d_categorical_matrix_file }: A filename for a data matrix that can contain categorical (non-numeric) data.  If the file contains only numeric data, then the same formats for regular data matrices can be used.  If the file contains strings or other values that can't be parsed as numbers, then the type to be loaded must be CSV (.csv) or ARFF (.arff).  Any non-numeric data will be converted to an unsigned integer value, and dimensions where the data is converted will be treated as categorical dimensions.  When using this format, there is no need for one-hot encoding of categorical data.
 - `mlpackModel file`{: #doc_model }: A filename containing an mlpack model.  These can have one of three formats: binary (.bin), text (.txt), and XML (.xml).  The XML format produces the largest (but most human-readable) files, while the binary format can be significantly more compact and quicker to load and save.
</div>


## mlpack_approx_kfn
{: #approx_kfn }

#### Approximate furthest neighbor search
{: #approx_kfn_descr }

```bash
$ mlpack_approx_kfn [--algorithm 'ds'] [--calculate_error]
        [--exact_distances_file <string>] [--help] [--info <string>]
        [--input_model_file <string>] [--k 0] [--num_projections 5]
        [--num_tables 5] [--query_file <string>] [--reference_file <string>]
        [--verbose] [--version] [--distances_file <string>] [--neighbors_file
        <string>] [--output_model_file <string>]
```

An implementation of two strategies for furthest neighbor search.  This can be used to compute the furthest neighbor of query point(s) from a set of points; furthest neighbor models can be saved and reused with future query point(s). [Detailed documentation](#approx_kfn_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--algorithm (-a)` | [`string`](#doc_string) | Algorithm to use: 'ds' or 'qdafn'. | `'ds'` |
| `--calculate_error (-e)` | [`flag`](#doc_flag) | If set, calculate the average distance error for the first furthest neighbor only. |  |
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--exact_distances_file (-x)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix containing exact distances to furthest neighbors; this can be used to avoid explicit calculation when --calculate_error is set. | `''` |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_model_file (-m)` | [`ApproxKFNModel file`](#doc_model) | File containing input model. | `''` |
| `--k (-k)` | [`int`](#doc_int) | Number of furthest neighbors to search for. | `0` |
| `--num_projections (-p)` | [`int`](#doc_int) | Number of projections to use in each hash table. | `5` |
| `--num_tables (-t)` | [`int`](#doc_int) | Number of hash tables to use. | `5` |
| `--query_file (-q)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix containing query points. | `''` |
| `--reference_file (-r)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix containing the reference dataset. | `''` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--distances_file (-d)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to save furthest neighbor distances to. | 
| `--neighbors_file (-n)` | [`2-d index matrix file`](#doc_a_2_d_index_matrix_file) | Matrix to save neighbor indices to. | 
| `--output_model_file (-M)` | [`ApproxKFNModel file`](#doc_model) | File to save output model to. | 

### Detailed documentation
{: #approx_kfn_detailed-documentation }

This program implements two strategies for furthest neighbor search. These strategies are:

 - The 'qdafn' algorithm from "Approximate Furthest Neighbor in High Dimensions" by R. Pagh, F. Silvestri, J. Sivertsen, and M. Skala, in Similarity Search and Applications 2015 (SISAP).
 - The 'DrusillaSelect' algorithm from "Fast approximate furthest neighbors with data-dependent candidate selection", by R.R. Curtin and A.B. Gardner, in Similarity Search and Applications 2016 (SISAP).

These two strategies give approximate results for the furthest neighbor search problem and can be used as fast replacements for other furthest neighbor techniques such as those found in the mlpack_kfn program.  Note that typically, the 'ds' algorithm requires far fewer tables and projections than the 'qdafn' algorithm.

Specify a reference set (set to search in) with `--reference_file (-r)`, specify a query set with `--query_file (-q)`, and specify algorithm parameters with `--num_tables (-t)` and `--num_projections (-p)` (or don't and defaults will be used).  The algorithm to be used (either 'ds'---the default---or 'qdafn')  may be specified with `--algorithm (-a)`.  Also specify the number of neighbors to search for with `--k (-k)`.

Note that for 'qdafn' in lower dimensions, `--num_projections (-p)` may need to be set to a high value in order to return results for each query point.

If no query set is specified, the reference set will be used as the query set.  The `--output_model_file (-M)` output parameter may be used to store the built model, and an input model may be loaded instead of specifying a reference set with the `--input_model_file (-m)` option.

Results for each query point can be stored with the `--neighbors_file (-n)` and `--distances_file (-d)` output parameters.  Each row of these output matrices holds the k distances or neighbor indices for each query point.

### Example
For example, to find the 5 approximate furthest neighbors with `'reference_set.csv'` as the reference set and `'query_set.csv'` as the query set using DrusillaSelect, storing the furthest neighbor indices to `'neighbors.csv'` and the furthest neighbor distances to `'distances.csv'`, one could call

```bash
$ mlpack_approx_kfn --query_file query_set.csv --reference_file
  reference_set.csv --k 5 --algorithm ds --neighbors_file neighbors.csv
  --distances_file distances.csv
```

and to perform approximate all-furthest-neighbors search with k=1 on the set `'data.csv'` storing only the furthest neighbor distances to `'distances.csv'`, one could call

```bash
$ mlpack_approx_kfn --reference_file reference_set.csv --k 1 --distances_file
  distances.csv
```

A trained model can be re-used.  If a model has been previously saved to `'model.bin'`, then we may find 3 approximate furthest neighbors on a query set `'new_query_set.csv'` using that model and store the furthest neighbor indices into `'neighbors.csv'` by calling

```bash
$ mlpack_approx_kfn --input_model_file model.bin --query_file
  new_query_set.csv --k 3 --neighbors_file neighbors.csv
```

### See also

 - [k-furthest-neighbor search](#kfn)
 - [k-nearest-neighbor search](#knn)
 - [Fast approximate furthest neighbors with data-dependent candidate selection (pdf)](http://ratml.org/pub/pdf/2016fast.pdf)
 - [Approximate furthest neighbor in high dimensions (pdf)](https://www.rasmuspagh.net/papers/approx-furthest-neighbor-SISAP15.pdf)
 - [QDAFN class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/approx_kfn/qdafn.hpp)
 - [DrusillaSelect class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/approx_kfn/drusilla_select.hpp)

## mlpack_bayesian_linear_regression
{: #bayesian_linear_regression }

#### BayesianLinearRegression
{: #bayesian_linear_regression_descr }

```bash
$ mlpack_bayesian_linear_regression [--center] [--help] [--info
        <string>] [--input_file <string>] [--input_model_file <string>]
        [--responses_file <string>] [--scale] [--test_file <string>] [--verbose]
        [--version] [--output_model_file <string>] [--predictions_file <string>]
        [--stds_file <string>]
```

An implementation of the bayesian linear regression. [Detailed documentation](#bayesian_linear_regression_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--center (-c)` | [`flag`](#doc_flag) | Center the data and fit the intercept if enabled. |  |
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_file (-i)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix of covariates (X). | `''` |
| `--input_model_file (-m)` | [`BayesianLinearRegression<> file`](#doc_model) | Trained BayesianLinearRegression model to use. | `''` |
| `--responses_file (-r)` | [`1-d matrix file`](#doc_a_1_d_matrix_file) | Matrix of responses/observations (y). | `''` |
| `--scale (-s)` | [`flag`](#doc_flag) | Scale each feature by their standard deviations if enabled. |  |
| `--test_file (-t)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix containing points to regress on (test points). | `''` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_model_file (-M)` | [`BayesianLinearRegression<> file`](#doc_model) | Output BayesianLinearRegression model. | 
| `--predictions_file (-o)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | If --test_file is specified, this file is where the predicted responses will be saved. | 
| `--stds_file (-u)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | If specified, this is where the standard deviations of the predictive distribution will be saved. | 

### Detailed documentation
{: #bayesian_linear_regression_detailed-documentation }

An implementation of the bayesian linear regression.
This model is a probabilistic view and implementation of the linear regression. The final solution is obtained by computing a posterior distribution from gaussian likelihood and a zero mean gaussian isotropic  prior distribution on the solution. 
Optimization is AUTOMATIC and does not require cross validation. The optimization is performed by maximization of the evidence function. Parameters are tuned during the maximization of the marginal likelihood. This procedure includes the Ockham's razor that penalizes over complex solutions. 

This program is able to train a Bayesian linear regression model or load a model from file, output regression predictions for a test set, and save the trained model to a file.

To train a BayesianLinearRegression model, the `--input_file (-i)` and `--responses_file (-r)`parameters must be given. The `--center (-c)`and `--scale (-s)` parameters control the centering and the normalizing options. A trained model can be saved with the `--output_model_file (-M)`. If no training is desired at all, a model can be passed via the `--input_model_file (-m)` parameter.

The program can also provide predictions for test data using either the trained model or the given input model.  Test points can be specified with the `--test_file (-t)` parameter.  Predicted responses to the test points can be saved with the `--predictions_file (-o)` output parameter. The corresponding standard deviation can be save by precising the `--stds_file (-u)` parameter.

### Example
For example, the following command trains a model on the data `'data.csv'` and responses `'responses.csv'`with center set to true and scale set to false (so, Bayesian linear regression is being solved, and then the model is saved to `'blr_model.bin'`:

```bash
$ mlpack_bayesian_linear_regression --input_file data.csv --responses_file
  responses.csv --center --scale --output_model_file blr_model.bin
```

The following command uses the `'blr_model.bin'` to provide predicted  responses for the data `'test.csv'` and save those  responses to `'test_predictions.csv'`: 

```bash
$ mlpack_bayesian_linear_regression --input_model_file blr_model.bin
  --test_file test.csv --predictions_file test_predictions.csv
```

Because the estimator computes a predictive distribution instead of a simple point estimate, the `--stds_file (-u)` parameter allows one to save the prediction uncertainties: 

```bash
$ mlpack_bayesian_linear_regression --input_model_file blr_model.bin
  --test_file test.csv --predictions_file test_predictions.csv --stds_file
  stds.csv
```

### See also

 - [Bayesian Interpolation](https://cs.uwaterloo.ca/~mannr/cs886-w10/mackay-bayesian.pdf)
 - [Bayesian Linear Regression, Section 3.3](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
 - [BayesianLinearRegression C++ class documentation](../../user/methods/bayesian_linear_regression.md)

## mlpack_cf
{: #cf }

#### Collaborative Filtering
{: #cf_descr }

```bash
$ mlpack_cf [--algorithm 'NMF'] [--all_user_recommendations] [--help]
        [--info <string>] [--input_model_file <string>] [--interpolation
        'average'] [--iteration_only_termination] [--max_iterations 1000]
        [--min_residue 1e-05] [--neighbor_search 'euclidean'] [--neighborhood 5]
        [--normalization 'none'] [--query_file <string>] [--rank 0]
        [--recommendations 5] [--seed 0] [--test_file <string>] [--training_file
        <string>] [--verbose] [--version] [--output_file <string>]
        [--output_model_file <string>]
```

An implementation of several collaborative filtering (CF) techniques for recommender systems.  This can be used to train a new CF model, or use an existing CF model to compute recommendations. [Detailed documentation](#cf_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--algorithm (-a)` | [`string`](#doc_string) | Algorithm used for matrix factorization. | `'NMF'` |
| `--all_user_recommendations (-A)` | [`flag`](#doc_flag) | Generate recommendations for all users. |  |
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_model_file (-m)` | [`CFModel file`](#doc_model) | Trained CF model to load. | `''` |
| `--interpolation (-i)` | [`string`](#doc_string) | Algorithm used for weight interpolation. | `'average'` |
| `--iteration_only_termination (-I)` | [`flag`](#doc_flag) | Terminate only when the maximum number of iterations is reached. |  |
| `--max_iterations (-N)` | [`int`](#doc_int) | Maximum number of iterations. If set to zero, there is no limit on the number of iterations. | `1000` |
| `--min_residue (-r)` | [`double`](#doc_double) | Residue required to terminate the factorization (lower values generally mean better fits). | `1e-05` |
| `--neighbor_search (-S)` | [`string`](#doc_string) | Algorithm used for neighbor search. | `'euclidean'` |
| `--neighborhood (-n)` | [`int`](#doc_int) | Size of the neighborhood of similar users to consider for each query user. | `5` |
| `--normalization (-z)` | [`string`](#doc_string) | Normalization performed on the ratings. | `'none'` |
| `--query_file (-q)` | [`2-d index matrix file`](#doc_a_2_d_index_matrix_file) | List of query users for which recommendations should be generated. | `''` |
| `--rank (-R)` | [`int`](#doc_int) | Rank of decomposed matrices (if 0, a heuristic is used to estimate the rank). | `0` |
| `--recommendations (-c)` | [`int`](#doc_int) | Number of recommendations to generate for each query user. | `5` |
| `--seed (-s)` | [`int`](#doc_int) | Set the random seed (0 uses std::time(NULL)). | `0` |
| `--test_file (-T)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Test set to calculate RMSE on. | `''` |
| `--training_file (-t)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Input dataset to perform CF on. | `''` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_file (-o)` | [`2-d index matrix file`](#doc_a_2_d_index_matrix_file) | Matrix that will store output recommendations. | 
| `--output_model_file (-M)` | [`CFModel file`](#doc_model) | Output for trained CF model. | 

### Detailed documentation
{: #cf_detailed-documentation }

This program performs collaborative filtering (CF) on the given dataset. Given a list of user, item and preferences (the `--training_file (-t)` parameter), the program will perform a matrix decomposition and then can perform a series of actions related to collaborative filtering.  Alternately, the program can load an existing saved CF model with the `--input_model_file (-m)` parameter and then use that model to provide recommendations or predict values.

The input matrix should be a 3-dimensional matrix of ratings, where the first dimension is the user, the second dimension is the item, and the third dimension is that user's rating of that item.  Both the users and items should be numeric indices, not names. The indices are assumed to start from 0.

A set of query users for which recommendations can be generated may be specified with the `--query_file (-q)` parameter; alternately, recommendations may be generated for every user in the dataset by specifying the `--all_user_recommendations (-A)` parameter.  In addition, the number of recommendations per user to generate can be specified with the `--recommendations (-c)` parameter, and the number of similar users (the size of the neighborhood) to be considered when generating recommendations can be specified with the `--neighborhood (-n)` parameter.

For performing the matrix decomposition, the following optimization algorithms can be specified via the `--algorithm (-a)` parameter:

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


The following neighbor search algorithms can be specified via the `--neighbor_search (-S)` parameter:

 - 'cosine'  -- Cosine Search Algorithm
 - 'euclidean'  -- Euclidean Search Algorithm
 - 'pearson'  -- Pearson Search Algorithm


The following weight interpolation algorithms can be specified via the `--interpolation (-i)` parameter:

 - 'average'  -- Average Interpolation Algorithm
 - 'regression'  -- Regression Interpolation Algorithm
 - 'similarity'  -- Similarity Interpolation Algorithm


The following ranking normalization algorithms can be specified via the `--normalization (-z)` parameter:

 - 'none'  -- No Normalization
 - 'item_mean'  -- Item Mean Normalization
 - 'overall_mean'  -- Overall Mean Normalization
 - 'user_mean'  -- User Mean Normalization
 - 'z_score'  -- Z-Score Normalization

A trained model may be saved to with the `--output_model_file (-M)` output parameter.

### Example
To train a CF model on a dataset `'training_set.csv'` using NMF for decomposition and saving the trained model to `'model.bin'`, one could call: 

```bash
$ mlpack_cf --training_file training_set.csv --algorithm NMF
  --output_model_file model.bin
```

Then, to use this model to generate recommendations for the list of users in the query set `'users.csv'`, storing 5 recommendations in `'recommendations.csv'`, one could call 

```bash
$ mlpack_cf --input_model_file model.bin --query_file users.csv
  --recommendations 5 --output_file recommendations.csv
```

### See also

 - [Collaborative Filtering on Wikipedia](https://en.wikipedia.org/wiki/Collaborative_filtering)
 - [Matrix factorization on Wikipedia](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems))
 - [Matrix factorization techniques for recommender systems (pdf)](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=cf17f85a0a7991fa01dbfb3e5878fbf71ea4bdc5)
 - [CFType class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/cf/cf.hpp)

## mlpack_dbscan
{: #dbscan }

#### DBSCAN clustering
{: #dbscan_descr }

```bash
$ mlpack_dbscan [--epsilon 1] [--help] [--info <string>] --input_file
        <string> [--min_size 5] [--naive] [--selection_type 'ordered']
        [--single_mode] [--tree_type 'kd'] [--verbose] [--version]
        [--assignments_file <string>] [--centroids_file <string>]
```

An implementation of DBSCAN clustering.  Given a dataset, this can compute and return a clustering of that dataset. [Detailed documentation](#dbscan_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--epsilon (-e)` | [`double`](#doc_double) | Radius of each range search. | `1` |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_file (-i)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Input dataset to cluster. | `**--**` |
| `--min_size (-m)` | [`int`](#doc_int) | Minimum number of points for a cluster. | `5` |
| `--naive (-N)` | [`flag`](#doc_flag) | If set, brute-force range search (not tree-based) will be used. |  |
| `--selection_type (-s)` | [`string`](#doc_string) | If using point selection policy, the type of selection to use ('ordered', 'random'). | `'ordered'` |
| `--single_mode (-S)` | [`flag`](#doc_flag) | If set, single-tree range search (not dual-tree) will be used. |  |
| `--tree_type (-t)` | [`string`](#doc_string) | If using single-tree or dual-tree search, the type of tree to use ('kd', 'r', 'r-star', 'x', 'hilbert-r', 'r-plus', 'r-plus-plus', 'cover', 'ball'). | `'kd'` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--assignments_file (-a)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | Output matrix for assignments of each point. | 
| `--centroids_file (-C)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to save output centroids to. | 

### Detailed documentation
{: #dbscan_detailed-documentation }

This program implements the DBSCAN algorithm for clustering using accelerated tree-based range search.  The type of tree that is used may be parameterized, or brute-force range search may also be used.

The input dataset to be clustered may be specified with the `--input_file (-i)` parameter; the radius of each range search may be specified with the `--epsilon (-e)` parameters, and the minimum number of points in a cluster may be specified with the `--min_size (-m)` parameter.

The `--assignments_file (-a)` and `--centroids_file (-C)` output parameters may be used to save the output of the clustering. `--assignments_file (-a)` contains the cluster assignments of each point, and `--centroids_file (-C)` contains the centroids of each cluster.

The range search may be controlled with the `--tree_type (-t)`, `--single_mode (-S)`, and `--naive (-N)` parameters.  `--tree_type (-t)` can control the type of tree used for range search; this can take a variety of values: 'kd', 'r', 'r-star', 'x', 'hilbert-r', 'r-plus', 'r-plus-plus', 'cover', 'ball'. The `--single_mode (-S)` parameter will force single-tree search (as opposed to the default dual-tree search), and '`--naive (-N)` will force brute-force range search.

### Example
An example usage to run DBSCAN on the dataset in `'input.csv'` with a radius of 0.5 and a minimum cluster size of 5 is given below:

```bash
$ mlpack_dbscan --input_file input.csv --epsilon 0.5 --min_size 5
```

### See also

 - [DBSCAN on Wikipedia](https://en.wikipedia.org/wiki/DBSCAN)
 - [A density-based algorithm for discovering clusters in large spatial databases with noise (pdf)](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf)
 - [DBSCAN class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/dbscan/dbscan.hpp)

## mlpack_decision_tree
{: #decision_tree }

#### Decision tree
{: #decision_tree_descr }

```bash
$ mlpack_decision_tree [--help] [--info <string>] [--input_model_file
        <string>] [--labels_file <string>] [--maximum_depth 0]
        [--minimum_gain_split 1e-07] [--minimum_leaf_size 20]
        [--print_training_accuracy] [--test_file <string>] [--test_labels_file
        <string>] [--training_file <string>] [--verbose] [--version]
        [--weights_file <string>] [--output_model_file <string>]
        [--predictions_file <string>] [--probabilities_file <string>]
```

An implementation of an ID3-style decision tree for classification, which supports categorical data.  Given labeled data with numeric or categorical features, a decision tree can be trained and saved; or, an existing decision tree can be used for classification on new points. [Detailed documentation](#decision_tree_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_model_file (-m)` | [`DecisionTreeModel file`](#doc_model) | Pre-trained decision tree, to be used with test points. | `''` |
| `--labels_file (-l)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | Training labels. | `''` |
| `--maximum_depth (-D)` | [`int`](#doc_int) | Maximum depth of the tree (0 means no limit). | `0` |
| `--minimum_gain_split (-g)` | [`double`](#doc_double) | Minimum gain for node splitting. | `1e-07` |
| `--minimum_leaf_size (-n)` | [`int`](#doc_int) | Minimum number of points in a leaf. | `20` |
| `--print_training_accuracy (-a)` | [`flag`](#doc_flag) | Print the training accuracy. |  |
| `--test_file (-T)` | [`2-d categorical matrix file`](#doc_a_2_d_categorical_matrix_file) | Testing dataset (may be categorical). | `''` |
| `--test_labels_file (-L)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | Test point labels, if accuracy calculation is desired. | `''` |
| `--training_file (-t)` | [`2-d categorical matrix file`](#doc_a_2_d_categorical_matrix_file) | Training dataset (may be categorical). | `''` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |
| `--weights_file (-w)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | The weight of labels | `''` |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_model_file (-M)` | [`DecisionTreeModel file`](#doc_model) | Output for trained decision tree. | 
| `--predictions_file (-p)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | Class predictions for each test point. | 
| `--probabilities_file (-P)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Class probabilities for each test point. | 

### Detailed documentation
{: #decision_tree_detailed-documentation }

Train and evaluate using a decision tree.  Given a dataset containing numeric or categorical features, and associated labels for each point in the dataset, this program can train a decision tree on that data.

The training set and associated labels are specified with the `--training_file (-t)` and `--labels_file (-l)` parameters, respectively.  The labels should be in the range `[0, num_classes - 1]`. Optionally, if `--labels_file (-l)` is not specified, the labels are assumed to be the last dimension of the training dataset.

When a model is trained, the `--output_model_file (-M)` output parameter may be used to save the trained model.  A model may be loaded for predictions with the `--input_model_file (-m)` parameter.  The `--input_model_file (-m)` parameter may not be specified when the `--training_file (-t)` parameter is specified.  The `--minimum_leaf_size (-n)` parameter specifies the minimum number of training points that must fall into each leaf for it to be split.  The `--minimum_gain_split (-g)` parameter specifies the minimum gain that is needed for the node to split.  The `--maximum_depth (-D)` parameter specifies the maximum depth of the tree.  If `--print_training_accuracy (-a)` is specified, the training accuracy will be printed.

Test data may be specified with the `--test_file (-T)` parameter, and if performance numbers are desired for that test set, labels may be specified with the `--test_labels_file (-L)` parameter.  Predictions for each test point may be saved via the `--predictions_file (-p)` output parameter.  Class probabilities for each prediction may be saved with the `--probabilities_file (-P)` output parameter.

### Example
For example, to train a decision tree with a minimum leaf size of 20 on the dataset contained in `'data.csv'` with labels `'labels.csv'`, saving the output model to `'tree.bin'` and printing the training error, one could call

```bash
$ mlpack_decision_tree --training_file data.arff --labels_file labels.csv
  --output_model_file tree.bin --minimum_leaf_size 20 --minimum_gain_split 0.001
  --print_training_accuracy
```

Then, to use that model to classify points in `'test_set.csv'` and print the test error given the labels `'test_labels.csv'` using that model, while saving the predictions for each point to `'predictions.csv'`, one could call 

```bash
$ mlpack_decision_tree --input_model_file tree.bin --test_file test_set.arff
  --test_labels_file test_labels.csv --predictions_file predictions.csv
```

### See also

 - [Random forest](#random_forest)
 - [Decision trees on Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning)
 - [Induction of Decision Trees (pdf)](https://www.hunch.net/~coms-4771/quinlan.pdf)
 - [DecisionTree C++ class documentation](../../user/methods/decision_tree.md)

## mlpack_det
{: #det }

#### Density Estimation With Density Estimation Trees
{: #det_descr }

```bash
$ mlpack_det [--folds 10] [--help] [--info <string>] [--input_model_file
        <string>] [--max_leaf_size 10] [--min_leaf_size 5] [--path_format 'lr']
        [--skip_pruning] [--test_file <string>] [--training_file <string>]
        [--verbose] [--version] [--output_model_file <string>]
        [--tag_counters_file <string>] [--tag_file <string>]
        [--test_set_estimates_file <string>] [--training_set_estimates_file
        <string>] [--vi_file <string>]
```

An implementation of density estimation trees for the density estimation task.  Density estimation trees can be trained or used to predict the density at locations given by query points. [Detailed documentation](#det_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--folds (-f)` | [`int`](#doc_int) | The number of folds of cross-validation to perform for the estimation (0 is LOOCV) | `10` |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_model_file (-m)` | [`DTree<> file`](#doc_model) | Trained density estimation tree to load. | `''` |
| `--max_leaf_size (-L)` | [`int`](#doc_int) | The maximum size of a leaf in the unpruned, fully grown DET. | `10` |
| `--min_leaf_size (-l)` | [`int`](#doc_int) | The minimum size of a leaf in the unpruned, fully grown DET. | `5` |
| `--path_format (-p)` | [`string`](#doc_string) | The format of path printing: 'lr', 'id-lr', or 'lr-id'. | `'lr'` |
| `--skip_pruning (-s)` | [`flag`](#doc_flag) | Whether to bypass the pruning process and output the unpruned tree only. |  |
| `--test_file (-T)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | A set of test points to estimate the density of. | `''` |
| `--training_file (-t)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | The data set on which to build a density estimation tree. | `''` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_model_file (-M)` | [`DTree<> file`](#doc_model) | Output to save trained density estimation tree to. | 
| `--tag_counters_file (-c)` | [`string`](#doc_string) | The file to output the number of points that went to each leaf. | 
| `--tag_file (-g)` | [`string`](#doc_string) | The file to output the tags (and possibly paths) for each sample in the test set. | 
| `--test_set_estimates_file (-E)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | The output estimates on the test set from the final optimally pruned tree. | 
| `--training_set_estimates_file (-e)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | The output density estimates on the training set from the final optimally pruned tree. | 
| `--vi_file (-i)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | The output variable importance values for each feature. | 

### Detailed documentation
{: #det_detailed-documentation }

This program performs a number of functions related to Density Estimation Trees.  The optimal Density Estimation Tree (DET) can be trained on a set of data (specified by `--training_file (-t)`) using cross-validation (with number of folds specified with the `--folds (-f)` parameter).  This trained density estimation tree may then be saved with the `--output_model_file (-M)` output parameter.

The variable importances (that is, the feature importance values for each dimension) may be saved with the `--vi_file (-i)` output parameter, and the density estimates for each training point may be saved with the `--training_set_estimates_file (-e)` output parameter.

Enabling path printing for each node outputs the path from the root node to a leaf for each entry in the test set, or training set (if a test set is not provided).  Strings like 'LRLRLR' (indicating that traversal went to the left child, then the right child, then the left child, and so forth) will be output. If 'lr-id' or 'id-lr' are given as the `--path_format (-p)` parameter, then the ID (tag) of every node along the path will be printed after or before the L or R character indicating the direction of traversal, respectively.

This program also can provide density estimates for a set of test points, specified in the `--test_file (-T)` parameter.  The density estimation tree used for this task will be the tree that was trained on the given training points, or a tree given as the parameter `--input_model_file (-m)`.  The density estimates for the test points may be saved using the `--test_set_estimates_file (-E)` output parameter.

### See also

 - [Density estimation on Wikipedia](https://en.wikipedia.org/wiki/Density_estimation)
 - [Density estimation trees (pdf)](https://www.mlpack.org/papers/det.pdf)
 - [DTree class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/det/dtree.hpp)

## mlpack_emst
{: #emst }

#### Fast Euclidean Minimum Spanning Tree
{: #emst_descr }

```bash
$ mlpack_emst [--help] [--info <string>] --input_file <string>
        [--leaf_size 1] [--naive] [--verbose] [--version] [--output_file
        <string>]
```

An implementation of the Dual-Tree Boruvka algorithm for computing the Euclidean minimum spanning tree of a set of input points. [Detailed documentation](#emst_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_file (-i)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Input data matrix. | `**--**` |
| `--leaf_size (-l)` | [`int`](#doc_int) | Leaf size in the kd-tree.  One-element leaves give the empirically best performance, but at the cost of greater memory requirements. | `1` |
| `--naive (-n)` | [`flag`](#doc_flag) | Compute the MST using O(n^2) naive algorithm. |  |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_file (-o)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Output data.  Stored as an edge list. | 

### Detailed documentation
{: #emst_detailed-documentation }

This program can compute the Euclidean minimum spanning tree of a set of input points using the dual-tree Boruvka algorithm.

The set to calculate the minimum spanning tree of is specified with the `--input_file (-i)` parameter, and the output may be saved with the `--output_file (-o)` output parameter.

The `--leaf_size (-l)` parameter controls the leaf size of the kd-tree that is used to calculate the minimum spanning tree, and if the `--naive (-n)` option is given, then brute-force search is used (this is typically much slower in low dimensions).  The leaf size does not affect the results, but it may have some effect on the runtime of the algorithm.

### Example
For example, the minimum spanning tree of the input dataset `'data.csv'` can be calculated with a leaf size of 20 and stored as `'spanning_tree.csv'` using the following command:

```bash
$ mlpack_emst --input_file data.csv --leaf_size 20 --output_file
  spanning_tree.csv
```

The output matrix is a three-dimensional matrix, where each row indicates an edge.  The first dimension corresponds to the lesser index of the edge; the second dimension corresponds to the greater index of the edge; and the third column corresponds to the distance between the two points.

### See also

 - [Minimum spanning tree on Wikipedia](https://en.wikipedia.org/wiki/Minimum_spanning_tree)
 - [Fast Euclidean Minimum Spanning Tree: Algorithm, Analysis, and Applications (pdf)](https://www.mlpack.org/papers/emst.pdf)
 - [DualTreeBoruvka class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/emst/dtb.hpp)

## mlpack_fastmks
{: #fastmks }

#### FastMKS (Fast Max-Kernel Search)
{: #fastmks_descr }

```bash
$ mlpack_fastmks [--bandwidth 1] [--base 2] [--degree 2] [--help]
        [--info <string>] [--input_model_file <string>] [--k 0] [--kernel
        'linear'] [--naive] [--offset 0] [--query_file <string>]
        [--reference_file <string>] [--scale 1] [--single] [--verbose]
        [--version] [--indices_file <string>] [--kernels_file <string>]
        [--output_model_file <string>]
```

An implementation of the single-tree and dual-tree fast max-kernel search (FastMKS) algorithm.  Given a set of reference points and a set of query points, this can find the reference point with maximum kernel value for each query point; trained models can be reused for future queries. [Detailed documentation](#fastmks_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--bandwidth (-w)` | [`double`](#doc_double) | Bandwidth (for Gaussian, Epanechnikov, and triangular kernels). | `1` |
| `--base (-b)` | [`double`](#doc_double) | Base to use during cover tree construction. | `2` |
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--degree (-d)` | [`double`](#doc_double) | Degree of polynomial kernel. | `2` |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_model_file (-m)` | [`FastMKSModel file`](#doc_model) | Input FastMKS model to use. | `''` |
| `--k (-k)` | [`int`](#doc_int) | Number of maximum kernels to find. | `0` |
| `--kernel (-K)` | [`string`](#doc_string) | Kernel type to use: 'linear', 'polynomial', 'cosine', 'gaussian', 'epanechnikov', 'triangular', 'hyptan'. | `'linear'` |
| `--naive (-N)` | [`flag`](#doc_flag) | If true, O(n^2) naive mode is used for computation. |  |
| `--offset (-o)` | [`double`](#doc_double) | Offset of kernel (for polynomial and hyptan kernels). | `0` |
| `--query_file (-q)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | The query dataset. | `''` |
| `--reference_file (-r)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | The reference dataset. | `''` |
| `--scale (-s)` | [`double`](#doc_double) | Scale of kernel (for hyptan kernel). | `1` |
| `--single (-S)` | [`flag`](#doc_flag) | If true, single-tree search is used (as opposed to dual-tree search. |  |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--indices_file (-i)` | [`2-d index matrix file`](#doc_a_2_d_index_matrix_file) | Output matrix of indices. | 
| `--kernels_file (-p)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Output matrix of kernels. | 
| `--output_model_file (-M)` | [`FastMKSModel file`](#doc_model) | Output for FastMKS model. | 

### Detailed documentation
{: #fastmks_detailed-documentation }

This program will find the k maximum kernels of a set of points, using a query set and a reference set (which can optionally be the same set). More specifically, for each point in the query set, the k points in the reference set with maximum kernel evaluations are found.  The kernel function used is specified with the `--kernel (-K)` parameter.

### Example
For example, the following command will calculate, for each point in the query set `'query.csv'`, the five points in the reference set `'reference.csv'` with maximum kernel evaluation using the linear kernel.  The kernel evaluations may be saved with the  `'kernels.csv'` output parameter and the indices may be saved with the `'indices.csv'` output parameter.

```bash
$ mlpack_fastmks --k 5 --reference_file reference.csv --query_file query.csv
  --indices_file indices.csv --kernels_file kernels.csv --kernel linear
```

The output matrices are organized such that row i and column j in the indices matrix corresponds to the index of the point in the reference set that has j'th largest kernel evaluation with the point in the query set with index i.  Row i and column j in the kernels matrix corresponds to the kernel evaluation between those two points.

This program performs FastMKS using a cover tree.  The base used to build the cover tree can be specified with the `--base (-b)` parameter.

### See also

 - [k-nearest-neighbor search](#knn)
 - [Dual-tree Fast Exact Max-Kernel Search (pdf)](https://mlpack.org/papers/fmks.pdf)
 - [FastMKS class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/fastmks/fastmks.hpp)

## mlpack_gmm_train
{: #gmm_train }

#### Gaussian Mixture Model (GMM) Training
{: #gmm_train_descr }

```bash
$ mlpack_gmm_train [--diagonal_covariance] --gaussians 0 [--help]
        [--info <string>] --input_file <string> [--input_model_file <string>]
        [--kmeans_max_iterations 1000] [--max_iterations 250]
        [--no_force_positive] [--noise 0] [--percentage 0.02] [--refined_start]
        [--samplings 100] [--seed 0] [--tolerance 1e-10] [--trials 1]
        [--verbose] [--version] [--output_model_file <string>]
```

An implementation of the EM algorithm for training Gaussian mixture models (GMMs).  Given a dataset, this can train a GMM for future use with other tools. [Detailed documentation](#gmm_train_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--diagonal_covariance (-d)` | [`flag`](#doc_flag) | Force the covariance of the Gaussians to be diagonal.  This can accelerate training time significantly. |  |
| `--gaussians (-g)` | [`int`](#doc_int) | Number of Gaussians in the GMM. | `**--**` |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_file (-i)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | The training data on which the model will be fit. | `**--**` |
| `--input_model_file (-m)` | [`GMM file`](#doc_model) | Initial input GMM model to start training with. | `''` |
| `--kmeans_max_iterations (-k)` | [`int`](#doc_int) | Maximum number of iterations for the k-means algorithm (used to initialize EM). | `1000` |
| `--max_iterations (-n)` | [`int`](#doc_int) | Maximum number of iterations of EM algorithm (passing 0 will run until convergence). | `250` |
| `--no_force_positive (-P)` | [`flag`](#doc_flag) | Do not force the covariance matrices to be positive definite. |  |
| `--noise (-N)` | [`double`](#doc_double) | Variance of zero-mean Gaussian noise to add to data. | `0` |
| `--percentage (-p)` | [`double`](#doc_double) | If using --refined_start, specify the percentage of the dataset used for each sampling (should be between 0.0 and 1.0). | `0.02` |
| `--refined_start (-r)` | [`flag`](#doc_flag) | During the initialization, use refined initial positions for k-means clustering (Bradley and Fayyad, 1998). |  |
| `--samplings (-S)` | [`int`](#doc_int) | If using --refined_start, specify the number of samplings used for initial points. | `100` |
| `--seed (-s)` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `--tolerance (-T)` | [`double`](#doc_double) | Tolerance for convergence of EM. | `1e-10` |
| `--trials (-t)` | [`int`](#doc_int) | Number of trials to perform in training GMM. | `1` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_model_file (-M)` | [`GMM file`](#doc_model) | Output for trained GMM model. | 

### Detailed documentation
{: #gmm_train_detailed-documentation }

This program takes a parametric estimate of a Gaussian mixture model (GMM) using the EM algorithm to find the maximum likelihood estimate.  The model may be saved and reused by other mlpack GMM tools.

The input data to train on must be specified with the `--input_file (-i)` parameter, and the number of Gaussians in the model must be specified with the `--gaussians (-g)` parameter.  Optionally, many trials with different random initializations may be run, and the result with highest log-likelihood on the training data will be taken.  The number of trials to run is specified with the `--trials (-t)` parameter.  By default, only one trial is run.

The tolerance for convergence and maximum number of iterations of the EM algorithm are specified with the `--tolerance (-T)` and `--max_iterations (-n)` parameters, respectively.  The GMM may be initialized for training with another model, specified with the `--input_model_file (-m)` parameter. Otherwise, the model is initialized by running k-means on the data.  The k-means clustering initialization can be controlled with the `--kmeans_max_iterations (-k)`, `--refined_start (-r)`, `--samplings (-S)`, and `--percentage (-p)` parameters.  If `--refined_start (-r)` is specified, then the Bradley-Fayyad refined start initialization will be used.  This can often lead to better clustering results.

The 'diagonal_covariance' flag will cause the learned covariances to be diagonal matrices.  This significantly simplifies the model itself and causes training to be faster, but restricts the ability to fit more complex GMMs.

If GMM training fails with an error indicating that a covariance matrix could not be inverted, make sure that the `--no_force_positive (-P)` parameter is not specified.  Alternately, adding a small amount of Gaussian noise (using the `--noise (-N)` parameter) to the entire dataset may help prevent Gaussians with zero variance in a particular dimension, which is usually the cause of non-invertible covariance matrices.

The `--no_force_positive (-P)` parameter, if set, will avoid the checks after each iteration of the EM algorithm which ensure that the covariance matrices are positive definite.  Specifying the flag can cause faster runtime, but may also cause non-positive definite covariance matrices, which will cause the program to crash.

### Example
As an example, to train a 6-Gaussian GMM on the data in `'data.csv'` with a maximum of 100 iterations of EM and 3 trials, saving the trained GMM to `'gmm.bin'`, the following command can be used:

```bash
$ mlpack_gmm_train --input_file data.csv --gaussians 6 --trials 3
  --output_model_file gmm.bin
```

To re-train that GMM on another set of data `'data2.csv'`, the following command may be used: 

```bash
$ mlpack_gmm_train --input_model_file gmm.bin --input_file data2.csv
  --gaussians 6 --output_model_file new_gmm.bin
```

### See also

 - [mlpack_gmm_generate](#gmm_generate)
 - [mlpack_gmm_probability](#gmm_probability)
 - [Gaussian Mixture Models on Wikipedia](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model)
 - [GMM class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/gmm/gmm.hpp)

## mlpack_gmm_generate
{: #gmm_generate }

#### GMM Sample Generator
{: #gmm_generate_descr }

```bash
$ mlpack_gmm_generate [--help] [--info <string>] --input_model_file
        <string> --samples 0 [--seed 0] [--verbose] [--version] [--output_file
        <string>]
```

A sample generator for pre-trained GMMs.  Given a pre-trained GMM, this can sample new points randomly from that distribution. [Detailed documentation](#gmm_generate_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_model_file (-m)` | [`GMM file`](#doc_model) | Input GMM model to generate samples from. | `**--**` |
| `--samples (-n)` | [`int`](#doc_int) | Number of samples to generate. | `**--**` |
| `--seed (-s)` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_file (-o)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to save output samples in. | 

### Detailed documentation
{: #gmm_generate_detailed-documentation }

This program is able to generate samples from a pre-trained GMM (use gmm_train to train a GMM).  The pre-trained GMM must be specified with the `--input_model_file (-m)` parameter.  The number of samples to generate is specified by the `--samples (-n)` parameter.  Output samples may be saved with the `--output_file (-o)` output parameter.

### Example
The following command can be used to generate 100 samples from the pre-trained GMM `'gmm.bin'` and store those generated samples in `'samples.csv'`:

```bash
$ mlpack_gmm_generate --input_model_file gmm.bin --samples 100 --output_file
  samples.csv
```

### See also

 - [mlpack_gmm_train](#gmm_train)
 - [mlpack_gmm_probability](#gmm_probability)
 - [Gaussian Mixture Models on Wikipedia](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model)
 - [GMM class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/gmm/gmm.hpp)

## mlpack_gmm_probability
{: #gmm_probability }

#### GMM Probability Calculator
{: #gmm_probability_descr }

```bash
$ mlpack_gmm_probability [--help] [--info <string>] --input_file
        <string> --input_model_file <string> [--verbose] [--version]
        [--output_file <string>]
```

A probability calculator for GMMs.  Given a pre-trained GMM and a set of points, this can compute the probability that each point is from the given GMM. [Detailed documentation](#gmm_probability_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_file (-i)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Input matrix to calculate probabilities of. | `**--**` |
| `--input_model_file (-m)` | [`GMM file`](#doc_model) | Input GMM to use as model. | `**--**` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_file (-o)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to store calculated probabilities in. | 

### Detailed documentation
{: #gmm_probability_detailed-documentation }

This program calculates the probability that given points came from a given GMM (that is, P(X \| gmm)).  The GMM is specified with the `--input_model_file (-m)` parameter, and the points are specified with the `--input_file (-i)` parameter.  The output probabilities may be saved via the `--output_file (-o)` output parameter.

### Example
So, for example, to calculate the probabilities of each point in `'points.csv'` coming from the pre-trained GMM `'gmm.bin'`, while storing those probabilities in `'probs.csv'`, the following command could be used:

```bash
$ mlpack_gmm_probability --input_model_file gmm.bin --input_file points.csv
  --output_file probs.csv
```

### See also

 - [mlpack_gmm_train](#gmm_train)
 - [mlpack_gmm_generate](#gmm_generate)
 - [Gaussian Mixture Models on Wikipedia](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model)
 - [GMM class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/gmm/gmm.hpp)

## mlpack_hmm_train
{: #hmm_train }

#### Hidden Markov Model (HMM) Training
{: #hmm_train_descr }

```bash
$ mlpack_hmm_train [--batch] [--gaussians 0] [--help] [--info <string>]
        --input_file <string> [--input_model_file <string>] [--labels_file
        <string>] [--seed 0] [--states 0] [--tolerance 1e-05] [--type
        'gaussian'] [--verbose] [--version] [--output_model_file <string>]
```

An implementation of training algorithms for Hidden Markov Models (HMMs). Given labeled or unlabeled data, an HMM can be trained for further use with other mlpack HMM tools. [Detailed documentation](#hmm_train_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--batch (-b)` | [`flag`](#doc_flag) | If true, input_file (and if passed, labels_file) are expected to contain a list of files to use as input observation sequences (and label sequences). |  |
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--gaussians (-g)` | [`int`](#doc_int) | Number of gaussians in each GMM (necessary when type is 'gmm'). | `0` |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_file (-i)` | [`string`](#doc_string) | File containing input observations. | `**--**` |
| `--input_model_file (-m)` | [`HMMModel file`](#doc_model) | Pre-existing HMM model to initialize training with. | `''` |
| `--labels_file (-l)` | [`string`](#doc_string) | Optional file of hidden states, used for labeled training. | `''` |
| `--seed (-s)` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `--states (-n)` | [`int`](#doc_int) | Number of hidden states in HMM (necessary, unless model_file is specified). | `0` |
| `--tolerance (-T)` | [`double`](#doc_double) | Tolerance of the Baum-Welch algorithm. | `1e-05` |
| `--type (-t)` | [`string`](#doc_string) | Type of HMM: discrete \| gaussian \| diag_gmm \| gmm. | `'gaussian'` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_model_file (-M)` | [`HMMModel file`](#doc_model) | Output for trained HMM. | 

### Detailed documentation
{: #hmm_train_detailed-documentation }

This program allows a Hidden Markov Model to be trained on labeled or unlabeled data.  It supports four types of HMMs: Discrete HMMs, Gaussian HMMs, GMM HMMs, or Diagonal GMM HMMs

Either one input sequence can be specified (with `--input_file (-i)`), or, a file containing files in which input sequences can be found (when `--input_file (-i)`and`--batch (-b)` are used together).  In addition, labels can be provided in the file specified by `--labels_file (-l)`, and if `--batch (-b)` is used, the file given to `--labels_file (-l)` should contain a list of files of labels corresponding to the sequences in the file given to `--input_file (-i)`.

The HMM is trained with the Baum-Welch algorithm if no labels are provided.  The tolerance of the Baum-Welch algorithm can be set with the `--tolerance (-T)`option.  By default, the transition matrix is randomly initialized and the emission distributions are initialized to fit the extent of the data.

Optionally, a pre-created HMM model can be used as a guess for the transition matrix and emission probabilities; this is specifiable with `--output_model_file (-M)`.

### See also

 - [mlpack_hmm_generate](#hmm_generate)
 - [mlpack_hmm_loglik](#hmm_loglik)
 - [mlpack_hmm_viterbi](#hmm_viterbi)
 - [Hidden Mixture Models on Wikipedia](https://en.wikipedia.org/wiki/Hidden_Markov_model)
 - [HMM class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/hmm/hmm.hpp)

## mlpack_hmm_generate
{: #hmm_generate }

#### Hidden Markov Model (HMM) Sequence Generator
{: #hmm_generate_descr }

```bash
$ mlpack_hmm_generate [--help] [--info <string>] --length 0 --model_file
        <string> [--seed 0] [--start_state 0] [--verbose] [--version]
        [--output_file <string>] [--state_file <string>]
```

A utility to generate random sequences from a pre-trained Hidden Markov Model (HMM).  The length of the desired sequence can be specified, and a random sequence of observations is returned. [Detailed documentation](#hmm_generate_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--length (-l)` | [`int`](#doc_int) | Length of sequence to generate. | `**--**` |
| `--model_file (-m)` | [`HMMModel file`](#doc_model) | Trained HMM to generate sequences with. | `**--**` |
| `--seed (-s)` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `--start_state (-t)` | [`int`](#doc_int) | Starting state of sequence. | `0` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_file (-o)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to save observation sequence to. | 
| `--state_file (-S)` | [`2-d index matrix file`](#doc_a_2_d_index_matrix_file) | Matrix to save hidden state sequence to. | 

### Detailed documentation
{: #hmm_generate_detailed-documentation }

This utility takes an already-trained HMM, specified as the `--model_file (-m)` parameter, and generates a random observation sequence and hidden state sequence based on its parameters. The observation sequence may be saved with the `--output_file (-o)` output parameter, and the internal state  sequence may be saved with the `--state_file (-S)` output parameter.

The state to start the sequence in may be specified with the `--start_state (-t)` parameter.

### Example
For example, to generate a sequence of length 150 from the HMM `'hmm.bin'` and save the observation sequence to `'observations.csv'` and the hidden state sequence to `'states.csv'`, the following command may be used: 

```bash
$ mlpack_hmm_generate --model_file hmm.bin --length 150 --output_file
  observations.csv --state_file states.csv
```

### See also

 - [mlpack_hmm_train](#hmm_train)
 - [mlpack_hmm_loglik](#hmm_loglik)
 - [mlpack_hmm_viterbi](#hmm_viterbi)
 - [Hidden Mixture Models on Wikipedia](https://en.wikipedia.org/wiki/Hidden_Markov_model)
 - [HMM class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/hmm/hmm.hpp)

## mlpack_hmm_loglik
{: #hmm_loglik }

#### Hidden Markov Model (HMM) Sequence Log-Likelihood
{: #hmm_loglik_descr }

```bash
$ mlpack_hmm_loglik [--help] [--info <string>] --input_file <string>
        --input_model_file <string> [--verbose] [--version] [--log_likelihood
        0]
```

A utility for computing the log-likelihood of a sequence for Hidden Markov Models (HMMs).  Given a pre-trained HMM and an observation sequence, this computes and returns the log-likelihood of that sequence being observed from that HMM. [Detailed documentation](#hmm_loglik_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_file (-i)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | File containing observations, | `**--**` |
| `--input_model_file (-m)` | [`HMMModel file`](#doc_model) | File containing HMM. | `**--**` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--log_likelihood` | [`double`](#doc_double) | Log-likelihood of the sequence. | 

### Detailed documentation
{: #hmm_loglik_detailed-documentation }

This utility takes an already-trained HMM, specified with the `--input_model_file (-m)` parameter, and evaluates the log-likelihood of a sequence of observations, given with the `--input_file (-i)` parameter.  The computed log-likelihood is given as output.

### Example
For example, to compute the log-likelihood of the sequence `'seq.csv'` with the pre-trained HMM `'hmm.bin'`, the following command may be used: 

```bash
$ mlpack_hmm_loglik --input_file seq.csv --input_model_file hmm.bin
```

### See also

 - [mlpack_hmm_train](#hmm_train)
 - [mlpack_hmm_generate](#hmm_generate)
 - [mlpack_hmm_viterbi](#hmm_viterbi)
 - [Hidden Mixture Models on Wikipedia](https://en.wikipedia.org/wiki/Hidden_Markov_model)
 - [HMM class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/hmm/hmm.hpp)

## mlpack_hmm_viterbi
{: #hmm_viterbi }

#### Hidden Markov Model (HMM) Viterbi State Prediction
{: #hmm_viterbi_descr }

```bash
$ mlpack_hmm_viterbi [--help] [--info <string>] --input_file <string>
        --input_model_file <string> [--verbose] [--version] [--output_file
        <string>]
```

A utility for computing the most probable hidden state sequence for Hidden Markov Models (HMMs).  Given a pre-trained HMM and an observed sequence, this uses the Viterbi algorithm to compute and return the most probable hidden state sequence. [Detailed documentation](#hmm_viterbi_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_file (-i)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix containing observations, | `**--**` |
| `--input_model_file (-m)` | [`HMMModel file`](#doc_model) | Trained HMM to use. | `**--**` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_file (-o)` | [`2-d index matrix file`](#doc_a_2_d_index_matrix_file) | File to save predicted state sequence to. | 

### Detailed documentation
{: #hmm_viterbi_detailed-documentation }

This utility takes an already-trained HMM, specified as `--input_model_file (-m)`, and evaluates the most probable hidden state sequence of a given sequence of observations (specified as '`--input_file (-i)`, using the Viterbi algorithm.  The computed state sequence may be saved using the `--output_file (-o)` output parameter.

### Example
For example, to predict the state sequence of the observations `'obs.csv'` using the HMM `'hmm.bin'`, storing the predicted state sequence to `'states.csv'`, the following command could be used:

```bash
$ mlpack_hmm_viterbi --input_file obs.csv --input_model_file hmm.bin
  --output_file states.csv
```

### See also

 - [mlpack_hmm_train](#hmm_train)
 - [mlpack_hmm_generate](#hmm_generate)
 - [mlpack_hmm_loglik](#hmm_loglik)
 - [Hidden Mixture Models on Wikipedia](https://en.wikipedia.org/wiki/Hidden_Markov_model)
 - [HMM class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/hmm/hmm.hpp)

## mlpack_hoeffding_tree
{: #hoeffding_tree }

#### Hoeffding trees
{: #hoeffding_tree_descr }

```bash
$ mlpack_hoeffding_tree [--batch_mode] [--bins 10] [--confidence 0.95]
        [--help] [--info <string>] [--info_gain] [--input_model_file <string>]
        [--labels_file <string>] [--max_samples 5000] [--min_samples 100]
        [--numeric_split_strategy 'binary'] [--observations_before_binning 100]
        [--passes 1] [--test_file <string>] [--test_labels_file <string>]
        [--training_file <string>] [--verbose] [--version] [--output_model_file
        <string>] [--predictions_file <string>] [--probabilities_file <string>]
```

An implementation of Hoeffding trees, a form of streaming decision tree for classification.  Given labeled data, a Hoeffding tree can be trained and saved for later use, or a pre-trained Hoeffding tree can be used for predicting the classifications of new points. [Detailed documentation](#hoeffding_tree_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--batch_mode (-b)` | [`flag`](#doc_flag) | If true, samples will be considered in batch instead of as a stream.  This generally results in better trees but at the cost of memory usage and runtime. |  |
| `--bins (-B)` | [`int`](#doc_int) | If the 'domingos' split strategy is used, this specifies the number of bins for each numeric split. | `10` |
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--confidence (-c)` | [`double`](#doc_double) | Confidence before splitting (between 0 and 1). | `0.95` |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--info_gain (-i)` | [`flag`](#doc_flag) | If set, information gain is used instead of Gini impurity for calculating Hoeffding bounds. |  |
| `--input_model_file (-m)` | [`HoeffdingTreeModel file`](#doc_model) | Input trained Hoeffding tree model. | `''` |
| `--labels_file (-l)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | Labels for training dataset. | `''` |
| `--max_samples (-n)` | [`int`](#doc_int) | Maximum number of samples before splitting. | `5000` |
| `--min_samples (-I)` | [`int`](#doc_int) | Minimum number of samples before splitting. | `100` |
| `--numeric_split_strategy (-N)` | [`string`](#doc_string) | The splitting strategy to use for numeric features: 'domingos' or 'binary'. | `'binary'` |
| `--observations_before_binning (-o)` | [`int`](#doc_int) | If the 'domingos' split strategy is used, this specifies the number of samples observed before binning is performed. | `100` |
| `--passes (-s)` | [`int`](#doc_int) | Number of passes to take over the dataset. | `1` |
| `--test_file (-T)` | [`2-d categorical matrix file`](#doc_a_2_d_categorical_matrix_file) | Testing dataset (may be categorical). | `''` |
| `--test_labels_file (-L)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | Labels of test data. | `''` |
| `--training_file (-t)` | [`2-d categorical matrix file`](#doc_a_2_d_categorical_matrix_file) | Training dataset (may be categorical). | `''` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_model_file (-M)` | [`HoeffdingTreeModel file`](#doc_model) | Output for trained Hoeffding tree model. | 
| `--predictions_file (-p)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | Matrix to output label predictions for test data into. | 
| `--probabilities_file (-P)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | In addition to predicting labels, provide rediction probabilities in this matrix. | 

### Detailed documentation
{: #hoeffding_tree_detailed-documentation }

This program implements Hoeffding trees, a form of streaming decision tree suited best for large (or streaming) datasets.  This program supports both categorical and numeric data.  Given an input dataset, this program is able to train the tree with numerous training options, and save the model to a file.  The program is also able to use a trained model or a model from file in order to predict classes for a given test set.

The training file and associated labels are specified with the `--training_file (-t)` and `--labels_file (-l)` parameters, respectively. Optionally, if `--labels_file (-l)` is not specified, the labels are assumed to be the last dimension of the training dataset.

The training may be performed in batch mode (like a typical decision tree algorithm) by specifying the `--batch_mode (-b)` option, but this may not be the best option for large datasets.

When a model is trained, it may be saved via the `--output_model_file (-M)` output parameter.  A model may be loaded from file for further training or testing with the `--input_model_file (-m)` parameter.

Test data may be specified with the `--test_file (-T)` parameter, and if performance statistics are desired for that test set, labels may be specified with the `--test_labels_file (-L)` parameter.  Predictions for each test point may be saved with the `--predictions_file (-p)` output parameter, and class probabilities for each prediction may be saved with the `--probabilities_file (-P)` output parameter.

### Example
For example, to train a Hoeffding tree with confidence 0.99 with data `'dataset.csv'`, saving the trained tree to `'tree.bin'`, the following command may be used:

```bash
$ mlpack_hoeffding_tree --training_file dataset.arff --confidence 0.99
  --output_model_file tree.bin
```

Then, this tree may be used to make predictions on the test set `'test_set.csv'`, saving the predictions into `'predictions.csv'` and the class probabilities into `'class_probs.csv'` with the following command: 

```bash
$ mlpack_hoeffding_tree --input_model_file tree.bin --test_file test_set.arff
  --predictions_file predictions.csv --probabilities_file class_probs.csv
```

### See also

 - [mlpack_decision_tree](#decision_tree)
 - [mlpack_random_forest](#random_forest)
 - [Mining High-Speed Data Streams (pdf)](http://dm.cs.washington.edu/papers/vfdt-kdd00.pdf)
 - [HoeffdingTree class documentation](../../user/methods/hoeffding_tree.md)

## mlpack_kde
{: #kde }

#### Kernel Density Estimation
{: #kde_descr }

```bash
$ mlpack_kde [--abs_error 0] [--algorithm 'dual-tree'] [--bandwidth 1]
        [--help] [--info <string>] [--initial_sample_size 100]
        [--input_model_file <string>] [--kernel 'gaussian'] [--mc_break_coef
        0.4] [--mc_entry_coef 3] [--mc_probability 0.95] [--monte_carlo]
        [--query_file <string>] [--reference_file <string>] [--rel_error 0.05]
        [--tree 'kd-tree'] [--verbose] [--version] [--output_model_file
        <string>] [--predictions_file <string>]
```

An implementation of kernel density estimation with dual-tree algorithms. Given a set of reference points and query points and a kernel function, this can estimate the density function at the location of each query point using trees; trees that are built can be saved for later use. [Detailed documentation](#kde_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--abs_error (-E)` | [`double`](#doc_double) | Relative error tolerance for the prediction. | `0` |
| `--algorithm (-a)` | [`string`](#doc_string) | Algorithm to use for the prediction.('dual-tree', 'single-tree'). | `'dual-tree'` |
| `--bandwidth (-b)` | [`double`](#doc_double) | Bandwidth of the kernel. | `1` |
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--initial_sample_size (-s)` | [`int`](#doc_int) | Initial sample size for Monte Carlo estimations. | `100` |
| `--input_model_file (-m)` | [`KDEModel file`](#doc_model) | Contains pre-trained KDE model. | `''` |
| `--kernel (-k)` | [`string`](#doc_string) | Kernel to use for the prediction.('gaussian', 'epanechnikov', 'laplacian', 'spherical', 'triangular'). | `'gaussian'` |
| `--mc_break_coef (-c)` | [`double`](#doc_double) | Controls what fraction of the amount of node's descendants is the limit for the sample size before it recurses. | `0.4` |
| `--mc_entry_coef (-C)` | [`double`](#doc_double) | Controls how much larger does the amount of node descendants has to be compared to the initial sample size in order to be a candidate for Monte Carlo estimations. | `3` |
| `--mc_probability (-P)` | [`double`](#doc_double) | Probability of the estimation being bounded by relative error when using Monte Carlo estimations. | `0.95` |
| `--monte_carlo (-S)` | [`flag`](#doc_flag) | Whether to use Monte Carlo estimations when possible. |  |
| `--query_file (-q)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Query dataset to KDE on. | `''` |
| `--reference_file (-r)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Input reference dataset use for KDE. | `''` |
| `--rel_error (-e)` | [`double`](#doc_double) | Relative error tolerance for the prediction. | `0.05` |
| `--tree (-t)` | [`string`](#doc_string) | Tree to use for the prediction.('kd-tree', 'ball-tree', 'cover-tree', 'octree', 'r-tree'). | `'kd-tree'` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_model_file (-M)` | [`KDEModel file`](#doc_model) | If specified, the KDE model will be saved here. | 
| `--predictions_file (-p)` | [`1-d matrix file`](#doc_a_1_d_matrix_file) | Vector to store density predictions. | 

### Detailed documentation
{: #kde_detailed-documentation }

This program performs a Kernel Density Estimation. KDE is a non-parametric way of estimating probability density function. For each query point the program will estimate its probability density by applying a kernel function to each reference point. The computational complexity of this is O(N^2) where there are N query points and N reference points, but this implementation will typically see better performance as it uses an approximate dual or single tree algorithm for acceleration.

Dual or single tree optimization avoids many barely relevant calculations (as kernel function values decrease with distance), so it is an approximate computation. You can specify the maximum relative error tolerance for each query value with `--rel_error (-e)` as well as the maximum absolute error tolerance with the parameter `--abs_error (-E)`. This program runs using an Euclidean metric. Kernel function can be selected using the `--kernel (-k)` option. You can also choose what which type of tree to use for the dual-tree algorithm with `--tree (-t)`. It is also possible to select whether to use dual-tree algorithm or single-tree algorithm using the `--algorithm (-a)` option.

Monte Carlo estimations can be used to accelerate the KDE estimate when the Gaussian Kernel is used. This provides a probabilistic guarantee on the the error of the resulting KDE instead of an absolute guarantee.To enable Monte Carlo estimations, the `--monte_carlo (-S)` flag can be used, and success probability can be set with the `--mc_probability (-P)` option. It is possible to set the initial sample size for the Monte Carlo estimation using `--initial_sample_size (-s)`. This implementation will only consider a node, as a candidate for the Monte Carlo estimation, if its number of descendant nodes is bigger than the initial sample size. This can be controlled using a coefficient that will multiply the initial sample size and can be set using `--mc_entry_coef (-C)`. To avoid using the same amount of computations an exact approach would take, this program recurses the tree whenever a fraction of the amount of the node's descendant points have already been computed. This fraction is set using `--mc_break_coef (-c)`.

### Example
For example, the following will run KDE using the data in `'ref_data.csv'` for training and the data in `'qu_data.csv'` as query data. It will apply an Epanechnikov kernel with a 0.2 bandwidth to each reference point and use a KD-Tree for the dual-tree optimization. The returned predictions will be within 5% of the real KDE value for each query point.

```bash
$ mlpack_kde --reference_file ref_data.csv --query_file qu_data.csv
  --bandwidth 0.2 --kernel epanechnikov --tree kd-tree --rel_error 0.05
  --predictions_file out_data.csv
```

the predicted density estimations will be stored in `'out_data.csv'`.
If no `--query_file (-q)` is provided, then KDE will be computed on the `--reference_file (-r)` dataset.
It is possible to select either a reference dataset or an input model but not both at the same time. If an input model is selected and parameter values are not set (e.g. `--bandwidth (-b)`) then default parameter values will be used.

In addition to the last program call, it is also possible to activate Monte Carlo estimations if a Gaussian kernel is used. This can provide faster results, but the KDE will only have a probabilistic guarantee of meeting the desired error bound (instead of an absolute guarantee). The following example will run KDE using a Monte Carlo estimation when possible. The results will be within a 5% of the real KDE value with a 95% probability. Initial sample size for the Monte Carlo estimation will be 200 points and a node will be a candidate for the estimation only when it contains 700 (i.e. 3.5*200) points. If a node contains 700 points and 420 (i.e. 0.6*700) have already been sampled, then the algorithm will recurse instead of keep sampling.

```bash
$ mlpack_kde --reference_file ref_data.csv --query_file qu_data.csv
  --bandwidth 0.2 --kernel gaussian --tree kd-tree --rel_error 0.05
  --predictions_file out_data.csv --monte_carlo --mc_probability 0.95
  --initial_sample_size 200 --mc_entry_coef 3.5 --mc_break_coef 0.6
```

### See also

 - [mlpack_knn](#knn)
 - [Kernel density estimation on Wikipedia](https://en.wikipedia.org/wiki/Kernel_density_estimation)
 - [Tree-Independent Dual-Tree Algorithms](https://arxiv.org/pdf/1304.4327)
 - [Fast High-dimensional Kernel Summations Using the Monte Carlo Multipole Method](https://proceedings.neurips.cc/paper_files/paper/2008/file/39059724f73a9969845dfe4146c5660e-Paper.pdf)
 - [KDE C++ class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/kde/kde.hpp)

## mlpack_kernel_pca
{: #kernel_pca }

#### Kernel Principal Components Analysis
{: #kernel_pca_descr }

```bash
$ mlpack_kernel_pca [--bandwidth 1] [--center] [--degree 1] [--help]
        [--info <string>] --input_file <string> --kernel <string>
        [--kernel_scale 1] [--new_dimensionality 0] [--nystroem_method]
        [--offset 0] [--sampling 'kmeans'] [--verbose] [--version]
        [--output_file <string>]
```

An implementation of Kernel Principal Components Analysis (KPCA).  This can be used to perform nonlinear dimensionality reduction or preprocessing on a given dataset. [Detailed documentation](#kernel_pca_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--bandwidth (-b)` | [`double`](#doc_double) | Bandwidth, for 'gaussian' and 'laplacian' kernels. | `1` |
| `--center (-c)` | [`flag`](#doc_flag) | If set, the transformed data will be centered about the origin. |  |
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--degree (-D)` | [`double`](#doc_double) | Degree of polynomial, for 'polynomial' kernel. | `1` |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_file (-i)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Input dataset to perform KPCA on. | `**--**` |
| `--kernel (-k)` | [`string`](#doc_string) | The kernel to use; see the above documentation for the list of usable kernels. | `**--**` |
| `--kernel_scale (-S)` | [`double`](#doc_double) | Scale, for 'hyptan' kernel. | `1` |
| `--new_dimensionality (-d)` | [`int`](#doc_int) | If not 0, reduce the dimensionality of the output dataset by ignoring the dimensions with the smallest eigenvalues. | `0` |
| `--nystroem_method (-n)` | [`flag`](#doc_flag) | If set, the Nystroem method will be used. |  |
| `--offset (-O)` | [`double`](#doc_double) | Offset, for 'hyptan' and 'polynomial' kernels. | `0` |
| `--sampling (-s)` | [`string`](#doc_string) | Sampling scheme to use for the Nystroem method: 'kmeans', 'random', 'ordered' | `'kmeans'` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_file (-o)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to save modified dataset to. | 

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

The parameters for each of the kernels should be specified with the options `--bandwidth (-b)`, `--kernel_scale (-S)`, `--offset (-O)`, or `--degree (-D)` (or a combination of those parameters).

Optionally, the Nystroem method ("Using the Nystroem method to speed up kernel machines", 2001) can be used to calculate the kernel matrix by specifying the `--nystroem_method (-n)` parameter. This approach works by using a subset of the data as basis to reconstruct the kernel matrix; to specify the sampling scheme, the `--sampling (-s)` parameter is used.  The sampling scheme for the Nystroem method can be chosen from the following list: 'kmeans', 'random', 'ordered'.

### Example
For example, the following command will perform KPCA on the dataset `'input.csv'` using the Gaussian kernel, and saving the transformed data to `'transformed.csv'`: 

```bash
$ mlpack_kernel_pca --input_file input.csv --kernel gaussian --output_file
  transformed.csv
```

### See also

 - [Kernel principal component analysis on Wikipedia](https://en.wikipedia.org/wiki/Kernel_principal_component_analysis)
 - [Nonlinear Component Analysis as a Kernel Eigenvalue Problem](https://www.mlpack.org/papers/kpca.pdf)
 - [KernelPCA class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/kernel_pca/kernel_pca.hpp)

## mlpack_kmeans
{: #kmeans }

#### K-Means Clustering
{: #kmeans_descr }

```bash
$ mlpack_kmeans [--algorithm 'naive'] [--allow_empty_clusters]
        --clusters 0 [--help] [--in_place] [--info <string>]
        [--initial_centroids_file <string>] --input_file <string>
        [--kill_empty_clusters] [--kmeans_plus_plus] [--labels_only]
        [--max_iterations 1000] [--percentage 0.02] [--refined_start]
        [--samplings 100] [--seed 0] [--verbose] [--version] [--centroid_file
        <string>] [--output_file <string>]
```

An implementation of several strategies for efficient k-means clustering. Given a dataset and a value of k, this computes and returns a k-means clustering on that data. [Detailed documentation](#kmeans_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--algorithm (-a)` | [`string`](#doc_string) | Algorithm to use for the Lloyd iteration ('naive', 'pelleg-moore', 'elkan', 'hamerly', 'dualtree', or 'dualtree-covertree'). | `'naive'` |
| `--allow_empty_clusters (-e)` | [`flag`](#doc_flag) | Allow empty clusters to be persist. |  |
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--clusters (-c)` | [`int`](#doc_int) | Number of clusters to find (0 autodetects from initial centroids). | `**--**` |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--in_place (-P)` | [`flag`](#doc_flag) | If specified, a column containing the learned cluster assignments will be added to the input dataset file.  In this case, --output_file is overridden. (Do not use in Python.) |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--initial_centroids_file (-I)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Start with the specified initial centroids. | `''` |
| `--input_file (-i)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Input dataset to perform clustering on. | `**--**` |
| `--kill_empty_clusters (-E)` | [`flag`](#doc_flag) | Remove empty clusters when they occur. |  |
| `--kmeans_plus_plus (-K)` | [`flag`](#doc_flag) | Use the k-means++ initialization strategy to choose initial points. |  |
| `--labels_only (-l)` | [`flag`](#doc_flag) | Only output labels into output file. |  |
| `--max_iterations (-m)` | [`int`](#doc_int) | Maximum number of iterations before k-means terminates. | `1000` |
| `--percentage (-p)` | [`double`](#doc_double) | Percentage of dataset to use for each refined start sampling (use when --refined_start is specified). | `0.02` |
| `--refined_start (-r)` | [`flag`](#doc_flag) | Use the refined initial point strategy by Bradley and Fayyad to choose initial points. |  |
| `--samplings (-S)` | [`int`](#doc_int) | Number of samplings to perform for refined start (use when --refined_start is specified). | `100` |
| `--seed (-s)` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--centroid_file (-C)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | If specified, the centroids of each cluster will  be written to the given file. | 
| `--output_file (-o)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to store output labels or labeled data to. | 

### Detailed documentation
{: #kmeans_detailed-documentation }

This program performs K-Means clustering on the given dataset.  It can return the learned cluster assignments, and the centroids of the clusters.  Empty clusters are not allowed by default; when a cluster becomes empty, the point furthest from the centroid of the cluster with maximum variance is taken to fill that cluster.

Optionally, the strategy to choose initial centroids can be specified.  The k-means++ algorithm can be used to choose initial centroids with the `--kmeans_plus_plus (-K)` parameter.  The Bradley and Fayyad approach ("Refining initial points for k-means clustering", 1998) can be used to select initial points by specifying the `--refined_start (-r)` parameter.  This approach works by taking random samplings of the dataset; to specify the number of samplings, the `--samplings (-S)` parameter is used, and to specify the percentage of the dataset to be used in each sample, the `--percentage (-p)` parameter is used (it should be a value between 0.0 and 1.0).

There are several options available for the algorithm used for each Lloyd iteration, specified with the `--algorithm (-a)`  option.  The standard O(kN) approach can be used ('naive').  Other options include the Pelleg-Moore tree-based algorithm ('pelleg-moore'), Elkan's triangle-inequality based algorithm ('elkan'), Hamerly's modification to Elkan's algorithm ('hamerly'), the dual-tree k-means algorithm ('dualtree'), and the dual-tree k-means algorithm using the cover tree ('dualtree-covertree').

The behavior for when an empty cluster is encountered can be modified with the `--allow_empty_clusters (-e)` option.  When this option is specified and there is a cluster owning no points at the end of an iteration, that cluster's centroid will simply remain in its position from the previous iteration. If the `--kill_empty_clusters (-E)` option is specified, then when a cluster owns no points at the end of an iteration, the cluster centroid is simply filled with DBL_MAX, killing it and effectively reducing k for the rest of the computation.  Note that the default option when neither empty cluster option is specified can be time-consuming to calculate; therefore, specifying either of these parameters will often accelerate runtime.

Initial clustering assignments may be specified using the `--initial_centroids_file (-I)` parameter, and the maximum number of iterations may be specified with the `--max_iterations (-m)` parameter.

### Example
As an example, to use Hamerly's algorithm to perform k-means clustering with k=10 on the dataset `'data.csv'`, saving the centroids to `'centroids.csv'` and the assignments for each point to `'assignments.csv'`, the following command could be used:

```bash
$ mlpack_kmeans --input_file data.csv --clusters 10 --output_file
  assignments.csv --centroid_file centroids.csv
```

To run k-means on that same dataset with initial centroids specified in `'initial.csv'` with a maximum of 500 iterations, storing the output centroids in `'final.csv'` the following command may be used:

```bash
$ mlpack_kmeans --input_file data.csv --initial_centroids_file initial.csv
  --clusters 10 --max_iterations 500 --centroid_file final.csv
```

### See also

 - [mlpack_dbscan](#dbscan)
 - [k-means++](https://en.wikipedia.org/wiki/K-means%2B%2B)
 - [Using the triangle inequality to accelerate k-means (pdf)](https://cdn.aaai.org/ICML/2003/ICML03-022.pdf)
 - [Making k-means even faster (pdf)](https://www.cse.iitd.ac.in/~rjaiswal/2015/col870/Project/Faster-k-means/Hamerly.pdf)
 - [Accelerating exact k-means algorithms with geometric reasoning (pdf)](http://reports-archive.adm.cs.cmu.edu/anon/anon/usr/ftp/usr0/ftp/2000/CMU-CS-00-105.pdf)
 - [A dual-tree algorithm for fast k-means clustering with large k (pdf)](http://www.ratml.org/pub/pdf/2017dual.pdf)
 - [KMeans class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/kmeans/kmeans.hpp)

## mlpack_lars
{: #lars }

#### LARS
{: #lars_descr }

```bash
$ mlpack_lars [--help] [--info <string>] [--input_file <string>]
        [--input_model_file <string>] [--lambda1 0] [--lambda2 0]
        [--no_intercept] [--no_normalize] [--responses_file <string>]
        [--test_file <string>] [--use_cholesky] [--verbose] [--version]
        [--output_model_file <string>] [--output_predictions_file <string>]
```

An implementation of Least Angle Regression (Stagewise/laSso), also known as LARS.  This can train a LARS/LASSO/Elastic Net model and use that model or a pre-trained model to output regression predictions for a test set. [Detailed documentation](#lars_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_file (-i)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix of covariates (X). | `''` |
| `--input_model_file (-m)` | [`LARS<> file`](#doc_model) | Trained LARS model to use. | `''` |
| `--lambda1 (-l)` | [`double`](#doc_double) | Regularization parameter for l1-norm penalty. | `0` |
| `--lambda2 (-L)` | [`double`](#doc_double) | Regularization parameter for l2-norm penalty. | `0` |
| `--no_intercept (-n)` | [`flag`](#doc_flag) | Do not fit an intercept in the model. |  |
| `--no_normalize (-N)` | [`flag`](#doc_flag) | Do not normalize data to unit variance before modeling. |  |
| `--responses_file (-r)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix of responses/observations (y). | `''` |
| `--test_file (-t)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix containing points to regress on (test points). | `''` |
| `--use_cholesky (-c)` | [`flag`](#doc_flag) | Use Cholesky decomposition during computation rather than explicitly computing the full Gram matrix. |  |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_model_file (-M)` | [`LARS<> file`](#doc_model) | Output LARS model. | 
| `--output_predictions_file (-o)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | If --test_file is specified, this file is where the predicted responses will be saved. | 

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

For efficiency reasons, it is not recommended to use this algorithm with `--lambda1 (-l)` = 0.  In that case, use the 'linear_regression' program, which implements both unregularized linear regression and ridge regression.

To train a LARS/LASSO/Elastic Net model, the `--input_file (-i)` and `--responses_file (-r)` parameters must be given.  The `--lambda1 (-l)`, `--lambda2 (-L)`, and `--use_cholesky (-c)` parameters control the training options.  A trained model can be saved with the `--output_model_file (-M)`.  If no training is desired at all, a model can be passed via the `--input_model_file (-m)` parameter.

The program can also provide predictions for test data using either the trained model or the given input model.  Test points can be specified with the `--test_file (-t)` parameter.  Predicted responses to the test points can be saved with the `--output_predictions_file (-o)` output parameter.

### Example
For example, the following command trains a model on the data `'data.csv'` and responses `'responses.csv'` with lambda1 set to 0.4 and lambda2 set to 0 (so, LASSO is being solved), and then the model is saved to `'lasso_model.bin'`:

```bash
$ mlpack_lars --input_file data.csv --responses_file responses.csv --lambda1
  0.4 --lambda2 0 --output_model_file lasso_model.bin
```

The following command uses the `'lasso_model.bin'` to provide predicted responses for the data `'test.csv'` and save those responses to `'test_predictions.csv'`: 

```bash
$ mlpack_lars --input_model_file lasso_model.bin --test_file test.csv
  --output_predictions_file test_predictions.csv
```

### See also

 - [mlpack_linear_regression](#linear_regression)
 - [Least angle regression (pdf)](https://mlpack.org/papers/lars.pdf)
 - [LARS C++ class documentation](../../user/methods/lars.md)

## mlpack_linear_svm
{: #linear_svm }

#### Linear SVM is an L2-regularized support vector machine.
{: #linear_svm_descr }

```bash
$ mlpack_linear_svm [--delta 1] [--epochs 50] [--help] [--info <string>]
        [--input_model_file <string>] [--labels_file <string>] [--lambda 0.0001]
        [--max_iterations 10000] [--no_intercept] [--num_classes 0] [--optimizer
        'lbfgs'] [--seed 0] [--shuffle] [--step_size 0.01] [--test_file
        <string>] [--test_labels_file <string>] [--tolerance 1e-10]
        [--training_file <string>] [--verbose] [--version] [--output_model_file
        <string>] [--predictions_file <string>] [--probabilities_file <string>]
```

An implementation of linear SVM for multiclass classification. Given labeled data, a model can be trained and saved for future use; or, a pre-trained model can be used to classify new points. [Detailed documentation](#linear_svm_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--delta (-d)` | [`double`](#doc_double) | Margin of difference between correct class and other classes. | `1` |
| `--epochs (-E)` | [`int`](#doc_int) | Maximum number of full epochs over dataset for psgd | `50` |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_model_file (-m)` | [`LinearSVMModel file`](#doc_model) | Existing model (parameters). | `''` |
| `--labels_file (-l)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | A matrix containing labels (0 or 1) for the points in the training set (y). | `''` |
| `--lambda (-r)` | [`double`](#doc_double) | L2-regularization parameter for training. | `0.0001` |
| `--max_iterations (-n)` | [`int`](#doc_int) | Maximum iterations for optimizer (0 indicates no limit). | `10000` |
| `--no_intercept (-N)` | [`flag`](#doc_flag) | Do not add the intercept term to the model. |  |
| `--num_classes (-c)` | [`int`](#doc_int) | Number of classes for classification; if unspecified (or 0), the number of classes found in the labels will be used. | `0` |
| `--optimizer (-O)` | [`string`](#doc_string) | Optimizer to use for training ('lbfgs' or 'psgd'). | `'lbfgs'` |
| `--seed (-s)` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `--shuffle (-S)` | [`flag`](#doc_flag) | Don't shuffle the order in which data points are visited for parallel SGD. |  |
| `--step_size (-a)` | [`double`](#doc_double) | Step size for parallel SGD optimizer. | `0.01` |
| `--test_file (-T)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix containing test dataset. | `''` |
| `--test_labels_file (-L)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | Matrix containing test labels. | `''` |
| `--tolerance (-e)` | [`double`](#doc_double) | Convergence tolerance for optimizer. | `1e-10` |
| `--training_file (-t)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | A matrix containing the training set (the matrix of predictors, X). | `''` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_model_file (-M)` | [`LinearSVMModel file`](#doc_model) | Output for trained linear svm model. | 
| `--predictions_file (-P)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | If test data is specified, this matrix is where the predictions for the test set will be saved. | 
| `--probabilities_file (-p)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | If test data is specified, this matrix is where the class probabilities for the test set will be saved. | 

### Detailed documentation
{: #linear_svm_detailed-documentation }

An implementation of linear SVMs that uses either L-BFGS or parallel SGD (stochastic gradient descent) to train the model.

This program allows loading a linear SVM model (via the `--input_model_file (-m)` parameter) or training a linear SVM model given training data (specified with the `--training_file (-t)` parameter), or both those things at once.  In addition, this program allows classification on a test dataset (specified with the `--test_file (-T)` parameter) and the classification results may be saved with the `--predictions_file (-P)` output parameter. The trained linear SVM model may be saved using the `--output_model_file (-M)` output parameter.

The training data, if specified, may have class labels as its last dimension.  Alternately, the `--labels_file (-l)` parameter may be used to specify a separate vector of labels.

When a model is being trained, there are many options.  L2 regularization (to prevent overfitting) can be specified with the `--lambda (-r)` option, and the number of classes can be manually specified with the `--num_classes (-c)`and if an intercept term is not desired in the model, the `--no_intercept (-N)` parameter can be specified.Margin of difference between correct class and other classes can be specified with the `--delta (-d)` option.The optimizer used to train the model can be specified with the `--optimizer (-O)` parameter.  Available options are 'psgd' (parallel stochastic gradient descent) and 'lbfgs' (the L-BFGS optimizer).  There are also various parameters for the optimizer; the `--max_iterations (-n)` parameter specifies the maximum number of allowed iterations, and the `--tolerance (-e)` parameter specifies the tolerance for convergence.  For the parallel SGD optimizer, the `--step_size (-a)` parameter controls the step size taken at each iteration by the optimizer and the maximum number of epochs (specified with `--epochs (-E)`). If the objective function for your data is oscillating between Inf and 0, the step size is probably too large.  There are more parameters for the optimizers, but the C++ interface must be used to access these.

Optionally, the model can be used to predict the labels for another matrix of data points, if `--test_file (-T)` is specified.  The `--test_file (-T)` parameter can be specified without the `--training_file (-t)` parameter, so long as an existing linear SVM model is given with the `--input_model_file (-m)` parameter.  The output predictions from the linear SVM model may be saved with the `--predictions_file (-P)` parameter.

### Example
As an example, to train a LinaerSVM on the data '`'data.csv'`' with labels '`'labels.csv'`' with L2 regularization of 0.1, saving the model to '`'lsvm_model.bin'`', the following command may be used:

```bash
$ mlpack_linear_svm --training_file data.csv --labels_file labels.csv --lambda
  0.1 --delta 1 --num_classes 0 --output_model_file lsvm_model.bin
```

Then, to use that model to predict classes for the dataset '`'test.csv'`', storing the output predictions in '`'predictions.csv'`', the following command may be used: 

```bash
$ mlpack_linear_svm --input_model_file lsvm_model.bin --test_file test.csv
  --predictions_file predictions.csv
```

### See also

 - [mlpack_random_forest](#random_forest)
 - [mlpack_logistic_regression](#logistic_regression)
 - [LinearSVM on Wikipedia](https://en.wikipedia.org/wiki/Support-vector_machine)
 - [LinearSVM C++ class documentation](../../user/methods/linear_svm.md)

## mlpack_lmnn
{: #lmnn }

#### Large Margin Nearest Neighbors (LMNN)
{: #lmnn_descr }

```bash
$ mlpack_lmnn [--batch_size 50] [--center] [--distance_file <string>]
        [--help] [--info <string>] --input_file <string> [--k 1] [--labels_file
        <string>] [--linear_scan] [--max_iterations 100000] [--normalize]
        [--optimizer 'amsgrad'] [--passes 50] [--print_accuracy] [--rank 0]
        [--regularization 0.5] [--seed 0] [--step_size 0.01] [--tolerance 1e-07]
        [--update_interval 1] [--verbose] [--version] [--centered_data_file
        <string>] [--output_file <string>] [--transformed_data_file <string>]
```

An implementation of Large Margin Nearest Neighbors (LMNN), a distance learning technique.  Given a labeled dataset, this learns a transformation of the data that improves k-nearest-neighbor performance; this can be useful as a preprocessing step. [Detailed documentation](#lmnn_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--batch_size (-b)` | [`int`](#doc_int) | Batch size for mini-batch SGD. | `50` |
| `--center (-C)` | [`flag`](#doc_flag) | Perform mean-centering on the dataset. It is useful when the centroid of the data is far from the origin. |  |
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--distance_file (-d)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Initial distance matrix to be used as starting point | `''` |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_file (-i)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Input dataset to run LMNN on. | `**--**` |
| `--k (-k)` | [`int`](#doc_int) | Number of target neighbors to use for each datapoint. | `1` |
| `--labels_file (-l)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | Labels for input dataset. | `''` |
| `--linear_scan (-L)` | [`flag`](#doc_flag) | Don't shuffle the order in which data points are visited for SGD or mini-batch SGD. |  |
| `--max_iterations (-n)` | [`int`](#doc_int) | Maximum number of iterations for L-BFGS (0 indicates no limit). | `100000` |
| `--normalize (-N)` | [`flag`](#doc_flag) | Use a normalized starting point for optimization. Itis useful for when points are far apart, or when SGD is returning NaN. |  |
| `--optimizer (-O)` | [`string`](#doc_string) | Optimizer to use; 'amsgrad', 'bbsgd', 'sgd', or 'lbfgs'. | `'amsgrad'` |
| `--passes (-p)` | [`int`](#doc_int) | Maximum number of full passes over dataset for AMSGrad, BB_SGD and SGD. | `50` |
| `--print_accuracy (-P)` | [`flag`](#doc_flag) | Print accuracies on initial and transformed dataset |  |
| `--rank (-A)` | [`int`](#doc_int) | Rank of distance matrix to be optimized.  | `0` |
| `--regularization (-r)` | [`double`](#doc_double) | Regularization for LMNN objective function  | `0.5` |
| `--seed (-s)` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `--step_size (-a)` | [`double`](#doc_double) | Step size for AMSGrad, BB_SGD and SGD (alpha). | `0.01` |
| `--tolerance (-t)` | [`double`](#doc_double) | Maximum tolerance for termination of AMSGrad, BB_SGD, SGD or L-BFGS. | `1e-07` |
| `--update_interval (-R)` | [`int`](#doc_int) | Number of iterations after which impostors need to be recalculated. | `1` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--centered_data_file (-c)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Output matrix for mean-centered dataset. | 
| `--output_file (-o)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Output matrix for learned distance matrix. | 
| `--transformed_data_file (-D)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Output matrix for transformed dataset. | 

### Detailed documentation
{: #lmnn_detailed-documentation }

This program implements Large Margin Nearest Neighbors, a distance learning technique.  The method seeks to improve k-nearest-neighbor classification on a dataset.  The method employes the strategy of reducing distance between similar labeled data points (a.k.a target neighbors) and increasing distance between differently labeled points (a.k.a impostors) using standard optimization techniques over the gradient of the distance between data points.

To work, this algorithm needs labeled data.  It can be given as the last row of the input dataset (specified with `--input_file (-i)`), or alternatively as a separate matrix (specified with `--labels_file (-l)`).  Additionally, a starting point for optimization (specified with `--distance_file (-d)`can be given, having (r x d) dimensionality.  Here r should satisfy 1 <= r <= d, Consequently a Low-Rank matrix will be optimized. Alternatively, Low-Rank distance can be learned by specifying the `--rank (-A)`parameter (A Low-Rank matrix with uniformly distributed values will be used as initial learning point). 

The program also requires number of targets neighbors to work with ( specified with `--k (-k)`), A regularization parameter can also be passed, It acts as a trade of between the pulling and pushing terms (specified with `--regularization (-r)`), In addition, this implementation of LMNN includes a parameter to decide the interval after which impostors must be re-calculated (specified with `--update_interval (-R)`).

Output can either be the learned distance matrix (specified with `--output_file (-o)`), or the transformed dataset  (specified with `--transformed_data_file (-D)`), or both. Additionally mean-centered dataset (specified with `--centered_data_file (-c)`) can be accessed given mean-centering (specified with `--center (-C)`) is performed on the dataset. Accuracy on initial dataset and final transformed dataset can be printed by specifying the `--print_accuracy (-P)`parameter. 

This implementation of LMNN uses AdaGrad, BigBatch_SGD, stochastic gradient descent, mini-batch stochastic gradient descent, or the L_BFGS optimizer. 

AdaGrad, specified by the value 'adagrad' for the parameter `--optimizer (-O)`, uses maximum of past squared gradients. It primarily on six parameters: the step size (specified with `--step_size (-a)`), the batch size (specified with `--batch_size (-b)`), the maximum number of passes (specified with `--passes (-p)`). Inaddition, a normalized starting point can be used by specifying the `--normalize (-N)` parameter. 

BigBatch_SGD, specified by the value 'bbsgd' for the parameter `--optimizer (-O)`, depends primarily on four parameters: the step size (specified with `--step_size (-a)`), the batch size (specified with `--batch_size (-b)`), the maximum number of passes (specified with `--passes (-p)`).  In addition, a normalized starting point can be used by specifying the `--normalize (-N)` parameter. 

Stochastic gradient descent, specified by the value 'sgd' for the parameter `--optimizer (-O)`, depends primarily on three parameters: the step size (specified with `--step_size (-a)`), the batch size (specified with `--batch_size (-b)`), and the maximum number of passes (specified with `--passes (-p)`).  In addition, a normalized starting point can be used by specifying the `--normalize (-N)` parameter. Furthermore, mean-centering can be performed on the dataset by specifying the `--center (-C)`parameter. 

The L-BFGS optimizer, specified by the value 'lbfgs' for the parameter `--optimizer (-O)`, uses a back-tracking line search algorithm to minimize a function.  The following parameters are used by L-BFGS: `--max_iterations (-n)`, `--tolerance (-t)`(the optimization is terminated when the gradient norm is below this value).  For more details on the L-BFGS optimizer, consult either the mlpack L-BFGS documentation (in lbfgs.hpp) or the vast set of published literature on L-BFGS.  In addition, a normalized starting point can be used by specifying the `--normalize (-N)` parameter.

By default, the AMSGrad optimizer is used.

### Example
Example - Let's say we want to learn distance on iris dataset with number of targets as 3 using BigBatch_SGD optimizer. A simple call for the same will look like: 

```bash
$ mlpack_lmnn --input_file iris.csv --labels_file iris_labels.csv --k 3
  --optimizer bbsgd --output_file output.csv
```

Another program call making use of update interval & regularization parameter with dataset having labels as last column can be made as: 

```bash
$ mlpack_lmnn --input_file letter_recognition.csv --k 5 --update_interval 10
  --regularization 0.4 --output_file output.csv
```

### See also

 - [mlpack_nca](#nca)
 - [Large margin nearest neighbor on Wikipedia](https://en.wikipedia.org/wiki/Large_margin_nearest_neighbor)
 - [Distance metric learning for large margin nearest neighbor classification (pdf)](https://proceedings.neurips.cc/paper_files/paper/2005/file/a7f592cef8b130a6967a90617db5681b-Paper.pdf)
 - [LMNN C++ class documentation](../../user/methods/lmnn.md)

## mlpack_local_coordinate_coding
{: #local_coordinate_coding }

#### Local Coordinate Coding
{: #local_coordinate_coding_descr }

```bash
$ mlpack_local_coordinate_coding [--atoms 0] [--help] [--info <string>]
        [--initial_dictionary_file <string>] [--input_model_file <string>]
        [--lambda 0] [--max_iterations 0] [--normalize] [--seed 0] [--test_file
        <string>] [--tolerance 0.01] [--training_file <string>] [--verbose]
        [--version] [--codes_file <string>] [--dictionary_file <string>]
        [--output_model_file <string>]
```

An implementation of Local Coordinate Coding (LCC), a data transformation technique.  Given input data, this transforms each point to be expressed as a linear combination of a few points in the dataset; once an LCC model is trained, it can be used to transform points later also. [Detailed documentation](#local_coordinate_coding_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--atoms (-k)` | [`int`](#doc_int) | Number of atoms in the dictionary. | `0` |
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--initial_dictionary_file (-i)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Optional initial dictionary. | `''` |
| `--input_model_file (-m)` | [`LocalCoordinateCoding<> file`](#doc_model) | Input LCC model. | `''` |
| `--lambda (-l)` | [`double`](#doc_double) | Weighted l1-norm regularization parameter. | `0` |
| `--max_iterations (-n)` | [`int`](#doc_int) | Maximum number of iterations for LCC (0 indicates no limit). | `0` |
| `--normalize (-N)` | [`flag`](#doc_flag) | If set, the input data matrix will be normalized before coding. |  |
| `--seed (-s)` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `--test_file (-T)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Test points to encode. | `''` |
| `--tolerance (-o)` | [`double`](#doc_double) | Tolerance for objective function. | `0.01` |
| `--training_file (-t)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix of training data (X). | `''` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--codes_file (-c)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Output codes matrix. | 
| `--dictionary_file (-d)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Output dictionary matrix. | 
| `--output_model_file (-M)` | [`LocalCoordinateCoding<> file`](#doc_model) | Output for trained LCC model. | 

### Detailed documentation
{: #local_coordinate_coding_detailed-documentation }

An implementation of Local Coordinate Coding (LCC), which codes data that approximately lives on a manifold using a variation of l1-norm regularized sparse coding.  Given a dense data matrix X with n points and d dimensions, LCC seeks to find a dense dictionary matrix D with k atoms in d dimensions, and a coding matrix Z with n points in k dimensions.  Because of the regularization method used, the atoms in D should lie close to the manifold on which the data points lie.

The original data matrix X can then be reconstructed as D * Z.  Therefore, this program finds a representation of each point in X as a sparse linear combination of atoms in the dictionary D.

The coding is found with an algorithm which alternates between a dictionary step, which updates the dictionary D, and a coding step, which updates the coding matrix Z.

To run this program, the input matrix X must be specified (with -i), along with the number of atoms in the dictionary (-k).  An initial dictionary may also be specified with the `--initial_dictionary_file (-i)` parameter.  The l1-norm regularization parameter is specified with the `--lambda (-l)` parameter.

### Example
For example, to run LCC on the dataset `'data.csv'` using 200 atoms and an l1-regularization parameter of 0.1, saving the dictionary `--dictionary_file (-d)` and the codes into `--codes_file (-c)`, use

```bash
$ mlpack_local_coordinate_coding --training_file data.csv --atoms 200 --lambda
  0.1 --dictionary_file dict.csv --codes_file codes.csv
```

The maximum number of iterations may be specified with the `--max_iterations (-n)` parameter. Optionally, the input data matrix X can be normalized before coding with the `--normalize (-N)` parameter.

An LCC model may be saved using the `--output_model_file (-M)` output parameter.  Then, to encode new points from the dataset `'points.csv'` with the previously saved model `'lcc_model.bin'`, saving the new codes to `'new_codes.csv'`, the following command can be used:

```bash
$ mlpack_local_coordinate_coding --input_model_file lcc_model.bin --test_file
  points.csv --codes_file new_codes.csv
```

### See also

 - [mlpack_sparse_coding](#sparse_coding)
 - [Nonlinear learning using local coordinate coding (pdf)](https://proceedings.neurips.cc/paper_files/paper/2009/file/2afe4567e1bf64d32a5527244d104cea-Paper.pdf)
 - [LocalCoordinateCoding C++ class documentation](../../user/methods/local_coordinate_coding.md)

## mlpack_logistic_regression
{: #logistic_regression }

#### L2-regularized Logistic Regression and Prediction
{: #logistic_regression_descr }

```bash
$ mlpack_logistic_regression [--batch_size 64] [--decision_boundary 0.5]
        [--help] [--info <string>] [--input_model_file <string>] [--labels_file
        <string>] [--lambda 0] [--max_iterations 10000] [--optimizer 'lbfgs']
        [--print_training_accuracy] [--step_size 0.01] [--test_file <string>]
        [--tolerance 1e-10] [--training_file <string>] [--verbose] [--version]
        [--output_model_file <string>] [--predictions_file <string>]
        [--probabilities_file <string>]
```

An implementation of L2-regularized logistic regression for two-class classification.  Given labeled data, a model can be trained and saved for future use; or, a pre-trained model can be used to classify new points. [Detailed documentation](#logistic_regression_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--batch_size (-b)` | [`int`](#doc_int) | Batch size for SGD. | `64` |
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--decision_boundary (-d)` | [`double`](#doc_double) | Decision boundary for prediction; if the logistic function for a point is less than the boundary, the class is taken to be 0; otherwise, the class is 1. | `0.5` |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_model_file (-m)` | [`LogisticRegression<> file`](#doc_model) | Existing model (parameters). | `''` |
| `--labels_file (-l)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | A matrix containing labels (0 or 1) for the points in the training set (y). | `''` |
| `--lambda (-L)` | [`double`](#doc_double) | L2-regularization parameter for training. | `0` |
| `--max_iterations (-n)` | [`int`](#doc_int) | Maximum iterations for optimizer (0 indicates no limit). | `10000` |
| `--optimizer (-O)` | [`string`](#doc_string) | Optimizer to use for training ('lbfgs' or 'sgd'). | `'lbfgs'` |
| `--print_training_accuracy (-a)` | [`flag`](#doc_flag) | If set, then the accuracy of the model on the training set will be printed (verbose must also be specified). |  |
| `--step_size (-s)` | [`double`](#doc_double) | Step size for SGD optimizer. | `0.01` |
| `--test_file (-T)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix containing test dataset. | `''` |
| `--tolerance (-e)` | [`double`](#doc_double) | Convergence tolerance for optimizer. | `1e-10` |
| `--training_file (-t)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | A matrix containing the training set (the matrix of predictors, X). | `''` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_model_file (-M)` | [`LogisticRegression<> file`](#doc_model) | Output for trained logistic regression model. | 
| `--predictions_file (-P)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | If test data is specified, this matrix is where the predictions for the test set will be saved. | 
| `--probabilities_file (-p)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | If test data is specified, this matrix is where the class probabilities for the test set will be saved. | 

### Detailed documentation
{: #logistic_regression_detailed-documentation }

An implementation of L2-regularized logistic regression using either the L-BFGS optimizer or SGD (stochastic gradient descent).  This solves the regression problem

  y = (1 / 1 + e^-(X * b)).

In this setting, y corresponds to class labels and X corresponds to data.

This program allows loading a logistic regression model (via the `--input_model_file (-m)` parameter) or training a logistic regression model given training data (specified with the `--training_file (-t)` parameter), or both those things at once.  In addition, this program allows classification on a test dataset (specified with the `--test_file (-T)` parameter) and the classification results may be saved with the `--predictions_file (-P)` output parameter. The trained logistic regression model may be saved using the `--output_model_file (-M)` output parameter.

The training data, if specified, may have class labels as its last dimension.  Alternately, the `--labels_file (-l)` parameter may be used to specify a separate matrix of labels.

When a model is being trained, there are many options.  L2 regularization (to prevent overfitting) can be specified with the `--lambda (-L)` option, and the optimizer used to train the model can be specified with the `--optimizer (-O)` parameter.  Available options are 'sgd' (stochastic gradient descent) and 'lbfgs' (the L-BFGS optimizer).  There are also various parameters for the optimizer; the `--max_iterations (-n)` parameter specifies the maximum number of allowed iterations, and the `--tolerance (-e)` parameter specifies the tolerance for convergence.  For the SGD optimizer, the `--step_size (-s)` parameter controls the step size taken at each iteration by the optimizer.  The batch size for SGD is controlled with the `--batch_size (-b)` parameter. If the objective function for your data is oscillating between Inf and 0, the step size is probably too large.  There are more parameters for the optimizers, but the C++ interface must be used to access these.

For SGD, an iteration refers to a single point. So to take a single pass over the dataset with SGD, `--max_iterations (-n)` should be set to the number of points in the dataset.

Optionally, the model can be used to predict the responses for another matrix of data points, if `--test_file (-T)` is specified.  The `--test_file (-T)` parameter can be specified without the `--training_file (-t)` parameter, so long as an existing logistic regression model is given with the `--input_model_file (-m)` parameter.  The output predictions from the logistic regression model may be saved with the `--predictions_file (-P)` parameter.

This implementation of logistic regression does not support the general multi-class case but instead only the two-class case.  Any labels must be either 0 or 1.  For more classes, see the softmax regression implementation.

### Example
As an example, to train a logistic regression model on the data '`'data.csv'`' with labels '`'labels.csv'`' with L2 regularization of 0.1, saving the model to '`'lr_model.bin'`', the following command may be used:

```bash
$ mlpack_logistic_regression --training_file data.csv --labels_file labels.csv
  --lambda 0.1 --output_model_file lr_model.bin --print_training_accuracy
```

Then, to use that model to predict classes for the dataset '`'test.csv'`', storing the output predictions in '`'predictions.csv'`', the following command may be used: 

```bash
$ mlpack_logistic_regression --input_model_file lr_model.bin --test_file
  test.csv --predictions_file predictions.csv
```

### See also

 - [mlpack_softmax_regression](#softmax_regression)
 - [mlpack_random_forest](#random_forest)
 - [Logistic regression on Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)
 - [:LogisticRegression C++ class documentation](../../user/methods/logistic_regression.md)

## mlpack_lsh
{: #lsh }

#### K-Approximate-Nearest-Neighbor Search with LSH
{: #lsh_descr }

```bash
$ mlpack_lsh [--bucket_size 500] [--hash_width 0] [--help] [--info
        <string>] [--input_model_file <string>] [--k 0] [--num_probes 0]
        [--projections 10] [--query_file <string>] [--reference_file <string>]
        [--second_hash_size 99901] [--seed 0] [--tables 30]
        [--true_neighbors_file <string>] [--verbose] [--version]
        [--distances_file <string>] [--neighbors_file <string>]
        [--output_model_file <string>]
```

An implementation of approximate k-nearest-neighbor search with locality-sensitive hashing (LSH).  Given a set of reference points and a set of query points, this will compute the k approximate nearest neighbors of each query point in the reference set; models can be saved for future use. [Detailed documentation](#lsh_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--bucket_size (-B)` | [`int`](#doc_int) | The size of a bucket in the second level hash. | `500` |
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--hash_width (-H)` | [`double`](#doc_double) | The hash width for the first-level hashing in the LSH preprocessing. By default, the LSH class automatically estimates a hash width for its use. | `0` |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_model_file (-m)` | [`LSHSearch<> file`](#doc_model) | Input LSH model. | `''` |
| `--k (-k)` | [`int`](#doc_int) | Number of nearest neighbors to find. | `0` |
| `--num_probes (-T)` | [`int`](#doc_int) | Number of additional probes for multiprobe LSH; if 0, traditional LSH is used. | `0` |
| `--projections (-K)` | [`int`](#doc_int) | The number of hash functions for each table | `10` |
| `--query_file (-q)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix containing query points (optional). | `''` |
| `--reference_file (-r)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix containing the reference dataset. | `''` |
| `--second_hash_size (-S)` | [`int`](#doc_int) | The size of the second level hash table. | `99901` |
| `--seed (-s)` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `--tables (-L)` | [`int`](#doc_int) | The number of hash tables to be used. | `30` |
| `--true_neighbors_file (-t)` | [`2-d index matrix file`](#doc_a_2_d_index_matrix_file) | Matrix of true neighbors to compute recall with (the recall is printed when -v is specified). | `''` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--distances_file (-d)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to output distances into. | 
| `--neighbors_file (-n)` | [`2-d index matrix file`](#doc_a_2_d_index_matrix_file) | Matrix to output neighbors into. | 
| `--output_model_file (-M)` | [`LSHSearch<> file`](#doc_model) | Output for trained LSH model. | 

### Detailed documentation
{: #lsh_detailed-documentation }

This program will calculate the k approximate-nearest-neighbors of a set of points using locality-sensitive hashing. You may specify a separate set of reference points and query points, or just a reference set which will be used as both the reference and query set. 

### Example
For example, the following will return 5 neighbors from the data for each point in `'input.csv'` and store the distances in `'distances.csv'` and the neighbors in `'neighbors.csv'`:

```bash
$ mlpack_lsh --k 5 --reference_file input.csv --distances_file distances.csv
  --neighbors_file neighbors.csv
```

The output is organized such that row i and column j in the neighbors output corresponds to the index of the point in the reference set which is the j'th nearest neighbor from the point in the query set with index i.  Row j and column i in the distances output file corresponds to the distance between those two points.

Because this is approximate-nearest-neighbors search, results may be different from run to run.  Thus, the `--seed (-s)` parameter can be specified to set the random seed.

This program also has many other parameters to control its functionality; see the parameter-specific documentation for more information.

### See also

 - [mlpack_knn](#knn)
 - [mlpack_krann](#krann)
 - [Locality-sensitive hashing on Wikipedia](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)
 - [Locality-sensitive hashing scheme based on p-stable  distributions(pdf)](https://www.mlpack.org/papers/lsh.pdf)
 - [LSHSearch C++ class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/lsh/lsh.hpp)

## mlpack_mean_shift
{: #mean_shift }

#### Mean Shift Clustering
{: #mean_shift_descr }

```bash
$ mlpack_mean_shift [--force_convergence] [--help] [--in_place] [--info
        <string>] --input_file <string> [--labels_only] [--max_iterations 1000]
        [--radius 0] [--verbose] [--version] [--centroid_file <string>]
        [--output_file <string>]
```

A fast implementation of mean-shift clustering using dual-tree range search.  Given a dataset, this uses the mean shift algorithm to produce and return a clustering of the data. [Detailed documentation](#mean_shift_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--force_convergence (-f)` | [`flag`](#doc_flag) | If specified, the mean shift algorithm will continue running regardless of max_iterations until the clusters converge. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--in_place (-P)` | [`flag`](#doc_flag) | If specified, a column containing the learned cluster assignments will be added to the input dataset file.  In this case, --output_file is overridden.  (Do not use with Python.) |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_file (-i)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Input dataset to perform clustering on. | `**--**` |
| `--labels_only (-l)` | [`flag`](#doc_flag) | If specified, only the output labels will be written to the file specified by --output_file. |  |
| `--max_iterations (-m)` | [`int`](#doc_int) | Maximum number of iterations before mean shift terminates. | `1000` |
| `--radius (-r)` | [`double`](#doc_double) | If the distance between two centroids is less than the given radius, one will be removed.  A radius of 0 or less means an estimate will be calculated and used for the radius. | `0` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--centroid_file (-C)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | If specified, the centroids of each cluster will be written to the given matrix. | 
| `--output_file (-o)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to write output labels or labeled data to. | 

### Detailed documentation
{: #mean_shift_detailed-documentation }

This program performs mean shift clustering on the given dataset, storing the learned cluster assignments either as a column of labels in the input dataset or separately.

The input dataset should be specified with the `--input_file (-i)` parameter, and the radius used for search can be specified with the `--radius (-r)` parameter.  The maximum number of iterations before algorithm termination is controlled with the `--max_iterations (-m)` parameter.

The output labels may be saved with the `--output_file (-o)` output parameter and the centroids of each cluster may be saved with the `--centroid_file (-C)` output parameter.

### Example
For example, to run mean shift clustering on the dataset `'data.csv'` and store the centroids to `'centroids.csv'`, the following command may be used: 

```bash
$ mlpack_mean_shift --input_file data.csv --centroid_file centroids.csv
```

### See also

 - [mlpack_kmeans](#kmeans)
 - [mlpack_dbscan](#dbscan)
 - [Mean shift on Wikipedia](https://en.wikipedia.org/wiki/Mean_shift)
 - [Mean Shift, Mode Seeking, and Clustering (pdf)](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=1c168275c59ba382588350ee1443537f59978183)
 - [mlpack::mean_shift::MeanShift C++ class documentation](../../user/methods/mean_shift.md)

## mlpack_nbc
{: #nbc }

#### Parametric Naive Bayes Classifier
{: #nbc_descr }

```bash
$ mlpack_nbc [--help] [--incremental_variance] [--info <string>]
        [--input_model_file <string>] [--labels_file <string>] [--test_file
        <string>] [--training_file <string>] [--verbose] [--version]
        [--output_model_file <string>] [--predictions_file <string>]
        [--probabilities_file <string>]
```

An implementation of the Naive Bayes Classifier, used for classification. Given labeled data, an NBC model can be trained and saved, or, a pre-trained model can be used for classification. [Detailed documentation](#nbc_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--incremental_variance (-I)` | [`flag`](#doc_flag) | The variance of each class will be calculated incrementally. |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_model_file (-m)` | [`NBCModel file`](#doc_model) | Input Naive Bayes model. | `''` |
| `--labels_file (-l)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | A file containing labels for the training set. | `''` |
| `--test_file (-T)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | A matrix containing the test set. | `''` |
| `--training_file (-t)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | A matrix containing the training set. | `''` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_model_file (-M)` | [`NBCModel file`](#doc_model) | File to save trained Naive Bayes model to. | 
| `--predictions_file (-a)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | The matrix in which the predicted labels for the test set will be written. | 
| `--probabilities_file (-p)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | The matrix in which the predicted probability of labels for the test set will be written. | 

### Detailed documentation
{: #nbc_detailed-documentation }

This program trains the Naive Bayes classifier on the given labeled training set, or loads a model from the given model file, and then may use that trained model to classify the points in a given test set.

The training set is specified with the `--training_file (-t)` parameter.  Labels may be either the last row of the training set, or alternately the `--labels_file (-l)` parameter may be specified to pass a separate matrix of labels.

If training is not desired, a pre-existing model may be loaded with the `--input_model_file (-m)` parameter.



The `--incremental_variance (-I)` parameter can be used to force the training to use an incremental algorithm for calculating variance.  This is slower, but can help avoid loss of precision in some cases.

If classifying a test set is desired, the test set may be specified with the `--test_file (-T)` parameter, and the classifications may be saved with the `--predictions_file (-a)`predictions  parameter.  If saving the trained model is desired, this may be done with the `--output_model_file (-M)` output parameter.

### Example
For example, to train a Naive Bayes classifier on the dataset `'data.csv'` with labels `'labels.csv'` and save the model to `'nbc_model.bin'`, the following command may be used:

```bash
$ mlpack_nbc --training_file data.csv --labels_file labels.csv
  --output_model_file nbc_model.bin
```

Then, to use `'nbc_model.bin'` to predict the classes of the dataset `'test_set.csv'` and save the predicted classes to `'predictions.csv'`, the following command may be used:

```bash
$ mlpack_nbc --input_model_file nbc_model.bin --test_file test_set.csv
  --predictions_file predictions.csv
```

### See also

 - [mlpack_softmax_regression](#softmax_regression)
 - [mlpack_random_forest](#random_forest)
 - [Naive Bayes classifier on Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
 - [NaiveBayesClassifier C++ class documentation](../../user/methods/naive_bayes_classifier.md)

## mlpack_nca
{: #nca }

#### Neighborhood Components Analysis (NCA)
{: #nca_descr }

```bash
$ mlpack_nca [--armijo_constant 0.0001] [--batch_size 50] [--help]
        [--info <string>] --input_file <string> [--labels_file <string>]
        [--linear_scan] [--max_iterations 500000] [--max_line_search_trials 50]
        [--max_step 1e+20] [--min_step 1e-20] [--normalize] [--num_basis 5]
        [--optimizer 'sgd'] [--seed 0] [--step_size 0.01] [--tolerance 1e-07]
        [--verbose] [--version] [--wolfe 0.9] [--output_file <string>]
```

An implementation of neighborhood components analysis, a distance learning technique that can be used for preprocessing.  Given a labeled dataset, this uses NCA, which seeks to improve the k-nearest-neighbor classification, and returns the learned distance metric. [Detailed documentation](#nca_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--armijo_constant (-A)` | [`double`](#doc_double) | Armijo constant for L-BFGS. | `0.0001` |
| `--batch_size (-b)` | [`int`](#doc_int) | Batch size for mini-batch SGD. | `50` |
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_file (-i)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Input dataset to run NCA on. | `**--**` |
| `--labels_file (-l)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | Labels for input dataset. | `''` |
| `--linear_scan (-L)` | [`flag`](#doc_flag) | Don't shuffle the order in which data points are visited for SGD or mini-batch SGD. |  |
| `--max_iterations (-n)` | [`int`](#doc_int) | Maximum number of iterations for SGD or L-BFGS (0 indicates no limit). | `500000` |
| `--max_line_search_trials (-T)` | [`int`](#doc_int) | Maximum number of line search trials for L-BFGS. | `50` |
| `--max_step (-M)` | [`double`](#doc_double) | Maximum step of line search for L-BFGS. | `1e+20` |
| `--min_step (-m)` | [`double`](#doc_double) | Minimum step of line search for L-BFGS. | `1e-20` |
| `--normalize (-N)` | [`flag`](#doc_flag) | Use a normalized starting point for optimization. This is useful for when points are far apart, or when SGD is returning NaN. |  |
| `--num_basis (-B)` | [`int`](#doc_int) | Number of memory points to be stored for L-BFGS. | `5` |
| `--optimizer (-O)` | [`string`](#doc_string) | Optimizer to use; 'sgd' or 'lbfgs'. | `'sgd'` |
| `--seed (-s)` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `--step_size (-a)` | [`double`](#doc_double) | Step size for stochastic gradient descent (alpha). | `0.01` |
| `--tolerance (-t)` | [`double`](#doc_double) | Maximum tolerance for termination of SGD or L-BFGS. | `1e-07` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |
| `--wolfe (-w)` | [`double`](#doc_double) | Wolfe condition parameter for L-BFGS. | `0.9` |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_file (-o)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Output matrix for learned distance matrix. | 

### Detailed documentation
{: #nca_detailed-documentation }

This program implements Neighborhood Components Analysis, both a linear dimensionality reduction technique and a distance learning technique.  The method seeks to improve k-nearest-neighbor classification on a dataset by scaling the dimensions.  The method is nonparametric, and does not require a value of k.  It works by using stochastic ("soft") neighbor assignments and using optimization techniques over the gradient of the accuracy of the neighbor assignments.

To work, this algorithm needs labeled data.  It can be given as the last row of the input dataset (specified with `--input_file (-i)`), or alternatively as a separate matrix (specified with `--labels_file (-l)`).

This implementation of NCA uses stochastic gradient descent, mini-batch stochastic gradient descent, or the L_BFGS optimizer.  These optimizers do not guarantee global convergence for a nonconvex objective function (NCA's objective function is nonconvex), so the final results could depend on the random seed or other optimizer parameters.

Stochastic gradient descent, specified by the value 'sgd' for the parameter `--optimizer (-O)`, depends primarily on three parameters: the step size (specified with `--step_size (-a)`), the batch size (specified with `--batch_size (-b)`), and the maximum number of iterations (specified with `--max_iterations (-n)`).  In addition, a normalized starting point can be used by specifying the `--normalize (-N)` parameter, which is necessary if many warnings of the form 'Denominator of p_i is 0!' are given.  Tuning the step size can be a tedious affair.  In general, the step size is too large if the objective is not mostly uniformly decreasing, or if zero-valued denominator warnings are being issued.  The step size is too small if the objective is changing very slowly.  Setting the termination condition can be done easily once a good step size parameter is found; either increase the maximum iterations to a large number and allow SGD to find a minimum, or set the maximum iterations to 0 (allowing infinite iterations) and set the tolerance (specified by `--tolerance (-t)`) to define the maximum allowed difference between objectives for SGD to terminate.  Be careful---setting the tolerance instead of the maximum iterations can take a very long time and may actually never converge due to the properties of the SGD optimizer. Note that a single iteration of SGD refers to a single point, so to take a single pass over the dataset, set the value of the `--max_iterations (-n)` parameter equal to the number of points in the dataset.

The L-BFGS optimizer, specified by the value 'lbfgs' for the parameter `--optimizer (-O)`, uses a back-tracking line search algorithm to minimize a function.  The following parameters are used by L-BFGS: `--num_basis (-B)` (specifies the number of memory points used by L-BFGS), `--max_iterations (-n)`, `--armijo_constant (-A)`, `--wolfe (-w)`, `--tolerance (-t)` (the optimization is terminated when the gradient norm is below this value), `--max_line_search_trials (-T)`, `--min_step (-m)`, and `--max_step (-M)` (which both refer to the line search routine).  For more details on the L-BFGS optimizer, consult either the mlpack L-BFGS documentation (in lbfgs.hpp) or the vast set of published literature on L-BFGS.

By default, the SGD optimizer is used.

### See also

 - [mlpack_lmnn](#lmnn)
 - [Neighbourhood components analysis on Wikipedia](https://en.wikipedia.org/wiki/Neighbourhood_components_analysis)
 - [Neighbourhood components analysis (pdf)](https://proceedings.neurips.cc/paper_files/paper/2004/file/42fe880812925e520249e808937738d2-Paper.pdf)
 - [NCA C++ class documentation](../../user/methods/nca.md)

## mlpack_knn
{: #knn }

#### k-Nearest-Neighbors Search
{: #knn_descr }

```bash
$ mlpack_knn [--algorithm 'dual_tree'] [--epsilon 0] [--help] [--info
        <string>] [--input_model_file <string>] [--k 0] [--leaf_size 20]
        [--query_file <string>] [--random_basis] [--reference_file <string>]
        [--rho 0.7] [--seed 0] [--tau 0] [--tree_type 'kd']
        [--true_distances_file <string>] [--true_neighbors_file <string>]
        [--verbose] [--version] [--distances_file <string>] [--neighbors_file
        <string>] [--output_model_file <string>]
```

An implementation of k-nearest-neighbor search using single-tree and dual-tree algorithms.  Given a set of reference points and query points, this can find the k nearest neighbors in the reference set of each query point using trees; trees that are built can be saved for future use. [Detailed documentation](#knn_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--algorithm (-a)` | [`string`](#doc_string) | Type of neighbor search: 'naive', 'single_tree', 'dual_tree', 'greedy'. | `'dual_tree'` |
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--epsilon (-e)` | [`double`](#doc_double) | If specified, will do approximate nearest neighbor search with given relative error. | `0` |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_model_file (-m)` | [`KNNModel file`](#doc_model) | Pre-trained kNN model. | `''` |
| `--k (-k)` | [`int`](#doc_int) | Number of nearest neighbors to find. | `0` |
| `--leaf_size (-l)` | [`int`](#doc_int) | Leaf size for tree building (used for kd-trees, vp trees, random projection trees, UB trees, R trees, R* trees, X trees, Hilbert R trees, R+ trees, R++ trees, spill trees, and octrees). | `20` |
| `--query_file (-q)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix containing query points (optional). | `''` |
| `--random_basis (-R)` | [`flag`](#doc_flag) | Before tree-building, project the data onto a random orthogonal basis. |  |
| `--reference_file (-r)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix containing the reference dataset. | `''` |
| `--rho (-b)` | [`double`](#doc_double) | Balance threshold (only valid for spill trees). | `0.7` |
| `--seed (-s)` | [`int`](#doc_int) | Random seed (if 0, std::time(NULL) is used). | `0` |
| `--tau (-u)` | [`double`](#doc_double) | Overlapping size (only valid for spill trees). | `0` |
| `--tree_type (-t)` | [`string`](#doc_string) | Type of tree to use: 'kd', 'vp', 'rp', 'max-rp', 'ub', 'cover', 'r', 'r-star', 'x', 'ball', 'hilbert-r', 'r-plus', 'r-plus-plus', 'spill', 'oct'. | `'kd'` |
| `--true_distances_file (-D)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix of true distances to compute the effective error (average relative error) (it is printed when -v is specified). | `''` |
| `--true_neighbors_file (-T)` | [`2-d index matrix file`](#doc_a_2_d_index_matrix_file) | Matrix of true neighbors to compute the recall (it is printed when -v is specified). | `''` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--distances_file (-d)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to output distances into. | 
| `--neighbors_file (-n)` | [`2-d index matrix file`](#doc_a_2_d_index_matrix_file) | Matrix to output neighbors into. | 
| `--output_model_file (-M)` | [`KNNModel file`](#doc_model) | If specified, the kNN model will be output here. | 

### Detailed documentation
{: #knn_detailed-documentation }

This program will calculate the k-nearest-neighbors of a set of points using kd-trees or cover trees (cover tree support is experimental and may be slow). You may specify a separate set of reference points and query points, or just a reference set which will be used as both the reference and query set.

### Example
For example, the following command will calculate the 5 nearest neighbors of each point in `'input.csv'` and store the distances in `'distances.csv'` and the neighbors in `'neighbors.csv'`: 

```bash
$ mlpack_knn --k 5 --reference_file input.csv --neighbors_file neighbors.csv
  --distances_file distances.csv
```

The output is organized such that row i and column j in the neighbors output matrix corresponds to the index of the point in the reference set which is the j'th nearest neighbor from the point in the query set with index i.  Row j and column i in the distances output matrix corresponds to the distance between those two points.

### See also

 - [mlpack_lsh](#lsh)
 - [mlpack_krann](#krann)
 - [mlpack_kfn](#kfn)
 - [Tree-independent dual-tree algorithms (pdf)](http://proceedings.mlr.press/v28/curtin13.pdf)
 - [NeighborSearch C++ class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/neighbor_search/neighbor_search.hpp)

## mlpack_kfn
{: #kfn }

#### k-Furthest-Neighbors Search
{: #kfn_descr }

```bash
$ mlpack_kfn [--algorithm 'dual_tree'] [--epsilon 0] [--help] [--info
        <string>] [--input_model_file <string>] [--k 0] [--leaf_size 20]
        [--percentage 1] [--query_file <string>] [--random_basis]
        [--reference_file <string>] [--seed 0] [--tree_type 'kd']
        [--true_distances_file <string>] [--true_neighbors_file <string>]
        [--verbose] [--version] [--distances_file <string>] [--neighbors_file
        <string>] [--output_model_file <string>]
```

An implementation of k-furthest-neighbor search using single-tree and dual-tree algorithms.  Given a set of reference points and query points, this can find the k furthest neighbors in the reference set of each query point using trees; trees that are built can be saved for future use. [Detailed documentation](#kfn_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--algorithm (-a)` | [`string`](#doc_string) | Type of neighbor search: 'naive', 'single_tree', 'dual_tree', 'greedy'. | `'dual_tree'` |
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--epsilon (-e)` | [`double`](#doc_double) | If specified, will do approximate furthest neighbor search with given relative error. Must be in the range [0,1). | `0` |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_model_file (-m)` | [`KFNModel file`](#doc_model) | Pre-trained kFN model. | `''` |
| `--k (-k)` | [`int`](#doc_int) | Number of furthest neighbors to find. | `0` |
| `--leaf_size (-l)` | [`int`](#doc_int) | Leaf size for tree building (used for kd-trees, vp trees, random projection trees, UB trees, R trees, R* trees, X trees, Hilbert R trees, R+ trees, R++ trees, and octrees). | `20` |
| `--percentage (-p)` | [`double`](#doc_double) | If specified, will do approximate furthest neighbor search. Must be in the range (0,1] (decimal form). Resultant neighbors will be at least (p*100) % of the distance as the true furthest neighbor. | `1` |
| `--query_file (-q)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix containing query points (optional). | `''` |
| `--random_basis (-R)` | [`flag`](#doc_flag) | Before tree-building, project the data onto a random orthogonal basis. |  |
| `--reference_file (-r)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix containing the reference dataset. | `''` |
| `--seed (-s)` | [`int`](#doc_int) | Random seed (if 0, std::time(NULL) is used). | `0` |
| `--tree_type (-t)` | [`string`](#doc_string) | Type of tree to use: 'kd', 'vp', 'rp', 'max-rp', 'ub', 'cover', 'r', 'r-star', 'x', 'ball', 'hilbert-r', 'r-plus', 'r-plus-plus', 'oct'. | `'kd'` |
| `--true_distances_file (-D)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix of true distances to compute the effective error (average relative error) (it is printed when -v is specified). | `''` |
| `--true_neighbors_file (-T)` | [`2-d index matrix file`](#doc_a_2_d_index_matrix_file) | Matrix of true neighbors to compute the recall (it is printed when -v is specified). | `''` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--distances_file (-d)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to output distances into. | 
| `--neighbors_file (-n)` | [`2-d index matrix file`](#doc_a_2_d_index_matrix_file) | Matrix to output neighbors into. | 
| `--output_model_file (-M)` | [`KFNModel file`](#doc_model) | If specified, the kFN model will be output here. | 

### Detailed documentation
{: #kfn_detailed-documentation }

This program will calculate the k-furthest-neighbors of a set of points. You may specify a separate set of reference points and query points, or just a reference set which will be used as both the reference and query set.

### Example
For example, the following will calculate the 5 furthest neighbors of eachpoint in `'input.csv'` and store the distances in `'distances.csv'` and the neighbors in `'neighbors.csv'`: 

```bash
$ mlpack_kfn --k 5 --reference_file input.csv --distances_file distances.csv
  --neighbors_file neighbors.csv
```

The output files are organized such that row i and column j in the neighbors output matrix corresponds to the index of the point in the reference set which is the j'th furthest neighbor from the point in the query set with index i.  Row i and column j in the distances output file corresponds to the distance between those two points.

### See also

 - [mlpack_approx_kfn](#approx_kfn)
 - [mlpack_knn](#knn)
 - [Tree-independent dual-tree algorithms (pdf)](http://proceedings.mlr.press/v28/curtin13.pdf)
 - [NeighborSearch C++ class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/neighbor_search/neighbor_search.hpp)

## mlpack_nmf
{: #nmf }

#### Non-negative Matrix Factorization
{: #nmf_descr }

```bash
$ mlpack_nmf [--help] [--info <string>] [--initial_h_file <string>]
        [--initial_w_file <string>] --input_file <string> [--max_iterations
        10000] [--min_residue 1e-05] --rank 0 [--seed 0] [--update_rules
        'multdist'] [--verbose] [--version] [--h_file <string>] [--w_file
        <string>]
```

An implementation of non-negative matrix factorization.  This can be used to decompose an input dataset into two low-rank non-negative components. [Detailed documentation](#nmf_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--initial_h_file (-q)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Initial H matrix. | `''` |
| `--initial_w_file (-p)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Initial W matrix. | `''` |
| `--input_file (-i)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Input dataset to perform NMF on. | `**--**` |
| `--max_iterations (-m)` | [`int`](#doc_int) | Number of iterations before NMF terminates (0 runs until convergence. | `10000` |
| `--min_residue (-e)` | [`double`](#doc_double) | The minimum root mean square residue allowed for each iteration, below which the program terminates. | `1e-05` |
| `--rank (-r)` | [`int`](#doc_int) | Rank of the factorization. | `**--**` |
| `--seed (-s)` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `--update_rules (-u)` | [`string`](#doc_string) | Update rules for each iteration; ( multdist \| multdiv \| als ). | `'multdist'` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--h_file (-H)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to save the calculated H to. | 
| `--w_file (-W)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to save the calculated W to. | 

### Detailed documentation
{: #nmf_detailed-documentation }

This program performs non-negative matrix factorization on the given dataset, storing the resulting decomposed matrices in the specified files.  For an input dataset V, NMF decomposes V into two matrices W and H such that 

V = W * H

where all elements in W and H are non-negative.  If V is of size (n x m), then W will be of size (n x r) and H will be of size (r x m), where r is the rank of the factorization (specified by the `--rank (-r)` parameter).

Optionally, the desired update rules for each NMF iteration can be chosen from the following list:

 - multdist: multiplicative distance-based update rules (Lee and Seung 1999)
 - multdiv: multiplicative divergence-based update rules (Lee and Seung 1999)
 - als: alternating least squares update rules (Paatero and Tapper 1994)

The maximum number of iterations is specified with `--max_iterations (-m)`, and the minimum residue required for algorithm termination is specified with the `--min_residue (-e)` parameter.

### Example
For example, to run NMF on the input matrix `'V.csv'` using the 'multdist' update rules with a rank-10 decomposition and storing the decomposed matrices into `'W.csv'` and `'H.csv'`, the following command could be used: 

```bash
$ mlpack_nmf --input_file V.csv --w_file W.csv --h_file H.csv --rank 10
  --update_rules multdist
```

### See also

 - [mlpack_cf](#cf)
 - [Non-negative matrix factorization on Wikipedia](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization)
 - [Algorithms for non-negative matrix factorization (pdf)](https://proceedings.neurips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf)
 - [NMF C++ class documentation](../../user/methods/nmf.md)
 - [AMF C++ class documentation](../../user/methods/amf.md)

## mlpack_pca
{: #pca }

#### Principal Components Analysis
{: #pca_descr }

```bash
$ mlpack_pca [--decomposition_method 'exact'] [--help] [--info <string>]
        --input_file <string> [--new_dimensionality 0] [--scale]
        [--var_to_retain 0] [--verbose] [--version] [--output_file <string>]
```

An implementation of several strategies for principal components analysis (PCA), a common preprocessing step.  Given a dataset and a desired new dimensionality, this can reduce the dimensionality of the data using the linear transformation determined by PCA. [Detailed documentation](#pca_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--decomposition_method (-c)` | [`string`](#doc_string) | Method used for the principal components analysis: 'exact', 'randomized', 'randomized-block-krylov', 'quic'. | `'exact'` |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_file (-i)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Input dataset to perform PCA on. | `**--**` |
| `--new_dimensionality (-d)` | [`int`](#doc_int) | Desired dimensionality of output dataset. If 0, no dimensionality reduction is performed. | `0` |
| `--scale (-s)` | [`flag`](#doc_flag) | If set, the data will be scaled before running PCA, such that the variance of each feature is 1. |  |
| `--var_to_retain (-r)` | [`double`](#doc_double) | Amount of variance to retain; should be between 0 and 1.  If 1, all variance is retained.  Overrides -d. | `0` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_file (-o)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to save modified dataset to. | 

### Detailed documentation
{: #pca_detailed-documentation }

This program performs principal components analysis on the given dataset using the exact, randomized, randomized block Krylov, or QUIC SVD method. It will transform the data onto its principal components, optionally performing dimensionality reduction by ignoring the principal components with the smallest eigenvalues.

Use the `--input_file (-i)` parameter to specify the dataset to perform PCA on.  A desired new dimensionality can be specified with the `--new_dimensionality (-d)` parameter, or the desired variance to retain can be specified with the `--var_to_retain (-r)` parameter.  If desired, the dataset can be scaled before running PCA with the `--scale (-s)` parameter.

Multiple different decomposition techniques can be used.  The method to use can be specified with the `--decomposition_method (-c)` parameter, and it may take the values 'exact', 'randomized', or 'quic'.

### Example
For example, to reduce the dimensionality of the matrix `'data.csv'` to 5 dimensions using randomized SVD for the decomposition, storing the output matrix to `'data_mod.csv'`, the following command can be used:

```bash
$ mlpack_pca --input_file data.csv --new_dimensionality 5
  --decomposition_method randomized --output_file data_mod.csv
```

### See also

 - [Principal component analysis on Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)
 - [PCA C++ class documentation](../../user/methods/pca.md)

## mlpack_perceptron
{: #perceptron }

#### Perceptron
{: #perceptron_descr }

```bash
$ mlpack_perceptron [--help] [--info <string>] [--input_model_file
        <string>] [--labels_file <string>] [--max_iterations 1000] [--test_file
        <string>] [--training_file <string>] [--verbose] [--version]
        [--output_model_file <string>] [--predictions_file <string>]
```

An implementation of a perceptron---a single level neural network--=for classification.  Given labeled data, a perceptron can be trained and saved for future use; or, a pre-trained perceptron can be used for classification on new points. [Detailed documentation](#perceptron_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_model_file (-m)` | [`PerceptronModel file`](#doc_model) | Input perceptron model. | `''` |
| `--labels_file (-l)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | A matrix containing labels for the training set. | `''` |
| `--max_iterations (-n)` | [`int`](#doc_int) | The maximum number of iterations the perceptron is to be run | `1000` |
| `--test_file (-T)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | A matrix containing the test set. | `''` |
| `--training_file (-t)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | A matrix containing the training set. | `''` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_model_file (-M)` | [`PerceptronModel file`](#doc_model) | Output for trained perceptron model. | 
| `--predictions_file (-P)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | The matrix in which the predicted labels for the test set will be written. | 

### Detailed documentation
{: #perceptron_detailed-documentation }

This program implements a perceptron, which is a single level neural network. The perceptron makes its predictions based on a linear predictor function combining a set of weights with the feature vector.  The perceptron learning rule is able to converge, given enough iterations (specified using the `--max_iterations (-n)` parameter), if the data supplied is linearly separable.  The perceptron is parameterized by a matrix of weight vectors that denote the numerical weights of the neural network.

This program allows loading a perceptron from a model (via the `--input_model_file (-m)` parameter) or training a perceptron given training data (via the `--training_file (-t)` parameter), or both those things at once.  In addition, this program allows classification on a test dataset (via the `--test_file (-T)` parameter) and the classification results on the test set may be saved with the `--predictions_file (-P)` output parameter.  The perceptron model may be saved with the `--output_model_file (-M)` output parameter.

### Example
The training data given with the `--training_file (-t)` option may have class labels as its last dimension (so, if the training data is in CSV format, labels should be the last column).  Alternately, the `--labels_file (-l)` parameter may be used to specify a separate matrix of labels.

All these options make it easy to train a perceptron, and then re-use that perceptron for later classification.  The invocation below trains a perceptron on `'training_data.csv'` with labels `'training_labels.csv'`, and saves the model to `'perceptron_model.bin'`.

```bash
$ mlpack_perceptron --training_file training_data.csv --labels_file
  training_labels.csv --output_model_file perceptron_model.bin
```

Then, this model can be re-used for classification on the test data `'test_data.csv'`.  The example below does precisely that, saving the predicted classes to `'predictions.csv'`.

```bash
$ mlpack_perceptron --input_model_file perceptron_model.bin --test_file
  test_data.csv --predictions_file predictions.csv
```

Note that all of the options may be specified at once: predictions may be calculated right after training a model, and model training can occur even if an existing perceptron model is passed with the `--input_model_file (-m)` parameter.  However, note that the number of classes and the dimensionality of all data must match.  So you cannot pass a perceptron model trained on 2 classes and then re-train with a 4-class dataset.  Similarly, attempting classification on a 3-dimensional dataset with a perceptron that has been trained on 8 dimensions will cause an error.

### See also

 - [mlpack_adaboost](#adaboost)
 - [Perceptron on Wikipedia](https://en.wikipedia.org/wiki/Perceptron)
 - [Perceptron C++ class documentation](../../user/methods/perceptron.md)

## mlpack_preprocess_split
{: #preprocess_split }

#### Split Data
{: #preprocess_split_descr }

```bash
$ mlpack_preprocess_split [--help] [--info <string>] --input_file
        <string> [--input_labels_file <string>] [--no_shuffle] [--seed 0]
        [--stratify_data] [--test_ratio 0.2] [--verbose] [--version]
        [--test_file <string>] [--test_labels_file <string>] [--training_file
        <string>] [--training_labels_file <string>]
```

A utility to split data into a training and testing dataset.  This can also split labels according to the same split. [Detailed documentation](#preprocess_split_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_file (-i)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix containing data. | `**--**` |
| `--input_labels_file (-I)` | [`2-d index matrix file`](#doc_a_2_d_index_matrix_file) | Matrix containing labels. | `''` |
| `--no_shuffle (-S)` | [`flag`](#doc_flag) | Avoid shuffling the data before splitting. |  |
| `--seed (-s)` | [`int`](#doc_int) | Random seed (0 for std::time(NULL)). | `0` |
| `--stratify_data (-z)` | [`flag`](#doc_flag) | Stratify the data according to labels |  |
| `--test_ratio (-r)` | [`double`](#doc_double) | Ratio of test set; if not set,the ratio defaults to 0.2 | `0.2` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--test_file (-T)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to save test data to. | 
| `--test_labels_file (-L)` | [`2-d index matrix file`](#doc_a_2_d_index_matrix_file) | Matrix to save test labels to. | 
| `--training_file (-t)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to save training data to. | 
| `--training_labels_file (-l)` | [`2-d index matrix file`](#doc_a_2_d_index_matrix_file) | Matrix to save train labels to. | 

### Detailed documentation
{: #preprocess_split_detailed-documentation }

This utility takes a dataset and optionally labels and splits them into a training set and a test set. Before the split, the points in the dataset are randomly reordered. The percentage of the dataset to be used as the test set can be specified with the `--test_ratio (-r)` parameter; the default is 0.2 (20%).

The output training and test matrices may be saved with the `--training_file (-t)` and `--test_file (-T)` output parameters.

Optionally, labels can also be split along with the data by specifying the `--input_labels_file (-I)` parameter.  Splitting labels works the same way as splitting the data. The output training and test labels may be saved with the `--training_labels_file (-l)` and `--test_labels_file (-L)` output parameters, respectively.

### Example
So, a simple example where we want to split the dataset `'X.csv'` into `'X_train.csv'` and `'X_test.csv'` with 60% of the data in the training set and 40% of the dataset in the test set, we could run 

```bash
$ mlpack_preprocess_split --input_file X.csv --training_file X_train.csv
  --test_file X_test.csv --test_ratio 0.4
```

Also by default the dataset is shuffled and split; you can provide the `--no_shuffle (-S)` option to avoid shuffling the data; an example to avoid shuffling of data is:

```bash
$ mlpack_preprocess_split --input_file X.csv --training_file X_train.csv
  --test_file X_test.csv --test_ratio 0.4 --no_shuffle
```

If we had a dataset `'X.csv'` and associated labels `'y.csv'`, and we wanted to split these into `'X_train.csv'`, `'y_train.csv'`, `'X_test.csv'`, and `'y_test.csv'`, with 30% of the data in the test set, we could run

```bash
$ mlpack_preprocess_split --input_file X.csv --input_labels_file y.csv
  --test_ratio 0.3 --training_file X_train.csv --training_labels_file
  y_train.csv --test_file X_test.csv --test_labels_file y_test.csv
```

To maintain the ratio of each class in the train and test sets, the`--stratify_data (-z)` option can be used.

```bash
$ mlpack_preprocess_split --input_file X.csv --training_file X_train.csv
  --test_file X_test.csv --test_ratio 0.4 --stratify_data
```

### See also

 - [mlpack_preprocess_binarize](#preprocess_binarize)
 - [mlpack_preprocess_describe](#preprocess_describe)

## mlpack_preprocess_binarize
{: #preprocess_binarize }

#### Binarize Data
{: #preprocess_binarize_descr }

```bash
$ mlpack_preprocess_binarize [--dimension 0] [--help] [--info <string>]
        --input_file <string> [--threshold 0] [--verbose] [--version]
        [--output_file <string>]
```

A utility to binarize a dataset.  Given a dataset, this utility converts each value in the desired dimension(s) to 0 or 1; this can be a useful preprocessing step. [Detailed documentation](#preprocess_binarize_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--dimension (-d)` | [`int`](#doc_int) | Dimension to apply the binarization. If not set, the program will binarize every dimension by default. | `0` |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_file (-i)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Input data matrix. | `**--**` |
| `--threshold (-t)` | [`double`](#doc_double) | Threshold to be applied for binarization. If not set, the threshold defaults to 0.0. | `0` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_file (-o)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix in which to save the output. | 

### Detailed documentation
{: #preprocess_binarize_detailed-documentation }

This utility takes a dataset and binarizes the variables into either 0 or 1 given threshold. User can apply binarization on a dimension or the whole dataset.  The dimension to apply binarization to can be specified using the `--dimension (-d)` parameter; if left unspecified, every dimension will be binarized.  The threshold for binarization can also be specified with the `--threshold (-t)` parameter; the default threshold is 0.0.

The binarized matrix may be saved with the `--output_file (-o)` output parameter.

### Example
For example, if we want to set all variables greater than 5 in the dataset `'X.csv'` to 1 and variables less than or equal to 5.0 to 0, and save the result to `'Y.csv'`, we could run

```bash
$ mlpack_preprocess_binarize --input_file X.csv --threshold 5 --output_file
  Y.csv
```

But if we want to apply this to only the first (0th) dimension of `'X.csv'`,  we could instead run

```bash
$ mlpack_preprocess_binarize --input_file X.csv --threshold 5 --dimension 0
  --output_file Y.csv
```

### See also

 - [mlpack_preprocess_describe](#preprocess_describe)
 - [mlpack_preprocess_split](#preprocess_split)

## mlpack_preprocess_describe
{: #preprocess_describe }

#### Descriptive Statistics
{: #preprocess_describe_descr }

```bash
$ mlpack_preprocess_describe [--dimension 0] [--help] [--info <string>]
        --input_file <string> [--population] [--precision 4] [--row_major]
        [--verbose] [--version] [--width 8]
```

A utility for printing descriptive statistics about a dataset.  This prints a number of details about a dataset in a tabular format. [Detailed documentation](#preprocess_describe_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--dimension (-d)` | [`int`](#doc_int) | Dimension of the data. Use this to specify a dimension | `0` |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_file (-i)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix containing data, | `**--**` |
| `--population (-P)` | [`flag`](#doc_flag) | If specified, the program will calculate statistics assuming the dataset is the population. By default, the program will assume the dataset as a sample. |  |
| `--precision (-p)` | [`int`](#doc_int) | Precision of the output statistics. | `4` |
| `--row_major (-r)` | [`flag`](#doc_flag) | If specified, the program will calculate statistics across rows, not across columns.  (Remember that in mlpack, a column represents a point, so this option is generally not necessary.) |  |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |
| `--width (-w)` | [`int`](#doc_int) | Width of the output table. | `8` |


### Detailed documentation
{: #preprocess_describe_detailed-documentation }

This utility takes a dataset and prints out the descriptive statistics of the data. Descriptive statistics is the discipline of quantitatively describing the main features of a collection of information, or the quantitative description itself. The program does not modify the original file, but instead prints out the statistics to the console. The printed result will look like a table.

Optionally, width and precision of the output can be adjusted by a user using the `--width (-w)` and `--precision (-p)` parameters. A user can also select a specific dimension to analyze if there are too many dimensions. The `--population (-P)` parameter can be specified when the dataset should be considered as a population.  Otherwise, the dataset will be considered as a sample.

### Example
So, a simple example where we want to print out statistical facts about the dataset `'X.csv'` using the default settings, we could run 

```bash
$ mlpack_preprocess_describe --input_file X.csv --verbose
```

If we want to customize the width to 10 and precision to 5 and consider the dataset as a population, we could run

```bash
$ mlpack_preprocess_describe --input_file X.csv --width 10 --precision 5
  --verbose
```

### See also

 - [mlpack_preprocess_binarize](#preprocess_binarize)
 - [mlpack_preprocess_split](#preprocess_split)

## mlpack_preprocess_scale
{: #preprocess_scale }

#### Scale Data
{: #preprocess_scale_descr }

```bash
$ mlpack_preprocess_scale [--epsilon 1e-06] [--help] [--info <string>]
        --input_file <string> [--input_model_file <string>] [--inverse_scaling]
        [--max_value 1] [--min_value 0] [--scaler_method 'standard_scaler']
        [--seed 0] [--verbose] [--version] [--output_file <string>]
        [--output_model_file <string>]
```

A utility to perform feature scaling on datasets using one of sixtechniques.  Both scaling and inverse scaling are supported, andscalers can be saved and then applied to other datasets. [Detailed documentation](#preprocess_scale_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--epsilon (-r)` | [`double`](#doc_double) | regularization Parameter for pcawhitening, or zcawhitening, should be between -1 to 1. | `1e-06` |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_file (-i)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix containing data. | `**--**` |
| `--input_model_file (-m)` | [`ScalingModel file`](#doc_model) | Input Scaling model. | `''` |
| `--inverse_scaling (-f)` | [`flag`](#doc_flag) | Inverse Scaling to get original dataset |  |
| `--max_value (-e)` | [`int`](#doc_int) | Ending value of range for min_max_scaler. | `1` |
| `--min_value (-b)` | [`int`](#doc_int) | Starting value of range for min_max_scaler. | `0` |
| `--scaler_method (-a)` | [`string`](#doc_string) | method to use for scaling, the default is standard_scaler. | `'standard_scaler'` |
| `--seed (-s)` | [`int`](#doc_int) | Random seed (0 for std::time(NULL)). | `0` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_file (-o)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to save scaled data to. | 
| `--output_model_file (-M)` | [`ScalingModel file`](#doc_model) | Output scaling model. | 

### Detailed documentation
{: #preprocess_scale_detailed-documentation }

This utility takes a dataset and performs feature scaling using one of the six scaler methods namely: 'max_abs_scaler', 'mean_normalization', 'min_max_scaler' ,'standard_scaler', 'pca_whitening' and 'zca_whitening'. The function takes a matrix as `--input_file (-i)` and a scaling method type which you can specify using `--scaler_method (-a)` parameter; the default is standard scaler, and outputs a matrix with scaled feature.

The output scaled feature matrix may be saved with the `--output_file (-o)` output parameters.

The model to scale features can be saved using `--output_model_file (-M)` and later can be loaded back using`--input_model_file (-m)`.

### Example
So, a simple example where we want to scale the dataset `'X.csv'` into `'X_scaled.csv'` with  standard_scaler as scaler_method, we could run 

```bash
$ mlpack_preprocess_scale --input_file X.csv --output_file X_scaled.csv
  --scaler_method standard_scaler
```

A simple example where we want to whiten the dataset `'X.csv'` into `'X_whitened.csv'` with  PCA as whitening_method and use 0.01 as regularization parameter, we could run 

```bash
$ mlpack_preprocess_scale --input_file X.csv --output_file X_scaled.csv
  --scaler_method pca_whitening --epsilon 0.01
```

You can also retransform the scaled dataset back using`--inverse_scaling (-f)`. An example to rescale : `'X_scaled.csv'` into `'X.csv'`using the saved model `--input_model_file (-m)` is:

```bash
$ mlpack_preprocess_scale --input_file X_scaled.csv --output_file X.csv
  --inverse_scaling --input_model_file saved.bin
```

Another simple example where we want to scale the dataset `'X.csv'` into `'X_scaled.csv'` with  min_max_scaler as scaler method, where scaling range is 1 to 3 instead of default 0 to 1. We could run 

```bash
$ mlpack_preprocess_scale --input_file X.csv --output_file X_scaled.csv
  --scaler_method min_max_scaler --min_value 1 --max_value 3
```

### See also

 - [mlpack_preprocess_binarize](#preprocess_binarize)
 - [mlpack_preprocess_describe](#preprocess_describe)

## mlpack_preprocess_one_hot_encoding
{: #preprocess_one_hot_encoding }

#### One Hot Encoding
{: #preprocess_one_hot_encoding_descr }

```bash
$ mlpack_preprocess_one_hot_encoding [--dimensions []] [--help] [--info
        <string>] --input_file <string> [--verbose] [--version] [--output_file
        <string>]
```

A utility to do one-hot encoding on features of dataset. [Detailed documentation](#preprocess_one_hot_encoding_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--dimensions (-d)` | [`int vector`](#doc_int_vector) | Index of dimensions that need to be one-hot encoded (if unspecified, all categorical dimensions are one-hot encoded). | `[]` |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_file (-i)` | [`2-d categorical matrix file`](#doc_a_2_d_categorical_matrix_file) | Matrix containing data. | `**--**` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_file (-o)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to save one-hot encoded features data to. | 

### Detailed documentation
{: #preprocess_one_hot_encoding_detailed-documentation }

This utility takes a dataset and a vector of indices and does one-hot encoding of the respective features at those indices. Indices represent the IDs of the dimensions to be one-hot encoded.

If no dimensions are specified with `--dimensions (-d)`, then all categorical-type dimensions will be one-hot encoded. Otherwise, only the dimensions given in `--dimensions (-d)` will be one-hot encoded.

The output matrix with encoded features may be saved with the `--output_file (-o)` parameters.

### Example
So, a simple example where we want to encode 1st and 3rd feature from dataset `'X.csv'` into `'X_output.csv'` would be

```bash
$ mlpack_preprocess_one_hot_encoding --input_file X.arff --output_file
  X_ouput.csv --dimensions 1 --dimensions 3
```

### See also

 - [mlpack_preprocess_binarize](#preprocess_binarize)
 - [mlpack_preprocess_describe](#preprocess_describe)
 - [One-hot encoding on Wikipedia](https://en.m.wikipedia.org/wiki/One-hot)

## mlpack_radical
{: #radical }

#### RADICAL
{: #radical_descr }

```bash
$ mlpack_radical [--angles 150] [--help] [--info <string>] --input_file
        <string> [--noise_std_dev 0.175] [--objective] [--replicates 30] [--seed
        0] [--sweeps 0] [--verbose] [--version] [--output_ic_file <string>]
        [--output_unmixing_file <string>]
```

An implementation of RADICAL, a method for independent component analysis (ICA).  Given a dataset, this can decompose the dataset into an unmixing matrix and an independent component matrix; this can be useful for preprocessing. [Detailed documentation](#radical_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--angles (-a)` | [`int`](#doc_int) | Number of angles to consider in brute-force search during Radical2D. | `150` |
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_file (-i)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Input dataset for ICA. | `**--**` |
| `--noise_std_dev (-n)` | [`double`](#doc_double) | Standard deviation of Gaussian noise. | `0.175` |
| `--objective (-O)` | [`flag`](#doc_flag) | If set, an estimate of the final objective function is printed. |  |
| `--replicates (-r)` | [`int`](#doc_int) | Number of Gaussian-perturbed replicates to use (per point) in Radical2D. | `30` |
| `--seed (-s)` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `--sweeps (-S)` | [`int`](#doc_int) | Number of sweeps; each sweep calls Radical2D once for each pair of dimensions. | `0` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_ic_file (-o)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to save independent components to. | 
| `--output_unmixing_file (-u)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to save unmixing matrix to. | 

### Detailed documentation
{: #radical_detailed-documentation }

An implementation of RADICAL, a method for independent component analysis (ICA).  Assuming that we have an input matrix X, the goal is to find a square unmixing matrix W such that Y = W * X and the dimensions of Y are independent components.  If the algorithm is running particularly slowly, try reducing the number of replicates.

The input matrix to perform ICA on should be specified with the `--input_file (-i)` parameter.  The output matrix Y may be saved with the `--output_ic_file (-o)` output parameter, and the output unmixing matrix W may be saved with the `--output_unmixing_file (-u)` output parameter.

### Example
For example, to perform ICA on the matrix `'X.csv'` with 40 replicates, saving the independent components to `'ic.csv'`, the following command may be used: 

```bash
$ mlpack_radical --input_file X.csv --replicates 40 --output_ic_file ic.csv
```

### See also

 - [Independent component analysis on Wikipedia](https://en.wikipedia.org/wiki/Independent_component_analysis)
 - [ICA using spacings estimates of entropy (pdf)](https://www.jmlr.org/papers/volume4/learned-miller03a/learned-miller03a.pdf)
 - [Radical C++ class documentation](../../user/methods/radical.md)

## mlpack_random_forest
{: #random_forest }

#### Random forests
{: #random_forest_descr }

```bash
$ mlpack_random_forest [--help] [--info <string>] [--input_model_file
        <string>] [--labels_file <string>] [--maximum_depth 0]
        [--minimum_gain_split 0] [--minimum_leaf_size 1] [--num_trees 10]
        [--print_training_accuracy] [--seed 0] [--subspace_dim 0] [--test_file
        <string>] [--test_labels_file <string>] [--training_file <string>]
        [--verbose] [--version] [--warm_start] [--output_model_file <string>]
        [--predictions_file <string>] [--probabilities_file <string>]
```

An implementation of the standard random forest algorithm by Leo Breiman for classification.  Given labeled data, a random forest can be trained and saved for future use; or, a pre-trained random forest can be used for classification. [Detailed documentation](#random_forest_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_model_file (-m)` | [`RandomForestModel file`](#doc_model) | Pre-trained random forest to use for classification. | `''` |
| `--labels_file (-l)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | Labels for training dataset. | `''` |
| `--maximum_depth (-D)` | [`int`](#doc_int) | Maximum depth of the tree (0 means no limit). | `0` |
| `--minimum_gain_split (-g)` | [`double`](#doc_double) | Minimum gain needed to make a split when building a tree. | `0` |
| `--minimum_leaf_size (-n)` | [`int`](#doc_int) | Minimum number of points in each leaf node. | `1` |
| `--num_trees (-N)` | [`int`](#doc_int) | Number of trees in the random forest. | `10` |
| `--print_training_accuracy (-a)` | [`flag`](#doc_flag) | If set, then the accuracy of the model on the training set will be predicted (verbose must also be specified). |  |
| `--seed (-s)` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `--subspace_dim (-d)` | [`int`](#doc_int) | Dimensionality of random subspace to use for each split.  '0' will autoselect the square root of data dimensionality. | `0` |
| `--test_file (-T)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Test dataset to produce predictions for. | `''` |
| `--test_labels_file (-L)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | Test dataset labels, if accuracy calculation is desired. | `''` |
| `--training_file (-t)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Training dataset. | `''` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |
| `--warm_start (-w)` | [`flag`](#doc_flag) | If true and passed along with `training` and `input_model` then trains more trees on top of existing model. |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_model_file (-M)` | [`RandomForestModel file`](#doc_model) | Model to save trained random forest to. | 
| `--predictions_file (-p)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | Predicted classes for each point in the test set. | 
| `--probabilities_file (-P)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Predicted class probabilities for each point in the test set. | 

### Detailed documentation
{: #random_forest_detailed-documentation }

This program is an implementation of the standard random forest classification algorithm by Leo Breiman.  A random forest can be trained and saved for later use, or a random forest may be loaded and predictions or class probabilities for points may be generated.

The training set and associated labels are specified with the `--training_file (-t)` and `--labels_file (-l)` parameters, respectively.  The labels should be in the range `[0, num_classes - 1]`. Optionally, if `--labels_file (-l)` is not specified, the labels are assumed to be the last dimension of the training dataset.

When a model is trained, the `--output_model_file (-M)` output parameter may be used to save the trained model.  A model may be loaded for predictions with the `--input_model_file (-m)`parameter. The `--input_model_file (-m)` parameter may not be specified when the `--training_file (-t)` parameter is specified.  The `--minimum_leaf_size (-n)` parameter specifies the minimum number of training points that must fall into each leaf for it to be split.  The `--num_trees (-N)` controls the number of trees in the random forest.  The `--minimum_gain_split (-g)` parameter controls the minimum required gain for a decision tree node to split.  Larger values will force higher-confidence splits.  The `--maximum_depth (-D)` parameter specifies the maximum depth of the tree.  The `--subspace_dim (-d)` parameter is used to control the number of random dimensions chosen for an individual node's split.  If `--print_training_accuracy (-a)` is specified, the calculated accuracy on the training set will be printed.

Test data may be specified with the `--test_file (-T)` parameter, and if performance measures are desired for that test set, labels for the test points may be specified with the `--test_labels_file (-L)` parameter.  Predictions for each test point may be saved via the `--predictions_file (-p)`output parameter.  Class probabilities for each prediction may be saved with the `--probabilities_file (-P)` output parameter.

### Example
For example, to train a random forest with a minimum leaf size of 20 using 10 trees on the dataset contained in `'data.csv'`with labels `'labels.csv'`, saving the output random forest to `'rf_model.bin'` and printing the training error, one could call

```bash
$ mlpack_random_forest --training_file data.csv --labels_file labels.csv
  --minimum_leaf_size 20 --num_trees 10 --output_model_file rf_model.bin
  --print_training_accuracy
```

Then, to use that model to classify points in `'test_set.csv'` and print the test error given the labels `'test_labels.csv'` using that model, while saving the predictions for each point to `'predictions.csv'`, one could call 

```bash
$ mlpack_random_forest --input_model_file rf_model.bin --test_file
  test_set.csv --test_labels_file test_labels.csv --predictions_file
  predictions.csv
```

### See also

 - [mlpack_decision_tree](#decision_tree)
 - [mlpack_hoeffding_tree](#hoeffding_tree)
 - [mlpack_softmax_regression](#softmax_regression)
 - [Random forest on Wikipedia](https://en.wikipedia.org/wiki/Random_forest)
 - [Random forests (pdf)](https://www.eecis.udel.edu/~shatkay/Course/papers/BreimanRandomForests2001.pdf)
 - [RandomForest C++ class documentation](../../user/methods/random_forest.md)

## mlpack_krann
{: #krann }

#### K-Rank-Approximate-Nearest-Neighbors (kRANN)
{: #krann_descr }

```bash
$ mlpack_krann [--alpha 0.95] [--first_leaf_exact] [--help] [--info
        <string>] [--input_model_file <string>] [--k 0] [--leaf_size 20]
        [--naive] [--query_file <string>] [--random_basis] [--reference_file
        <string>] [--sample_at_leaves] [--seed 0] [--single_mode]
        [--single_sample_limit 20] [--tau 5] [--tree_type 'kd'] [--verbose]
        [--version] [--distances_file <string>] [--neighbors_file <string>]
        [--output_model_file <string>]
```

An implementation of rank-approximate k-nearest-neighbor search (kRANN)  using single-tree and dual-tree algorithms.  Given a set of reference points and query points, this can find the k nearest neighbors in the reference set of each query point using trees; trees that are built can be saved for future use. [Detailed documentation](#krann_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--alpha (-a)` | [`double`](#doc_double) | The desired success probability. | `0.95` |
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--first_leaf_exact (-X)` | [`flag`](#doc_flag) | The flag to trigger sampling only after exactly exploring the first leaf. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_model_file (-m)` | [`RAModel file`](#doc_model) | Pre-trained kNN model. | `''` |
| `--k (-k)` | [`int`](#doc_int) | Number of nearest neighbors to find. | `0` |
| `--leaf_size (-l)` | [`int`](#doc_int) | Leaf size for tree building (used for kd-trees, UB trees, R trees, R* trees, X trees, Hilbert R trees, R+ trees, R++ trees, and octrees). | `20` |
| `--naive (-N)` | [`flag`](#doc_flag) | If true, sampling will be done without using a tree. |  |
| `--query_file (-q)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix containing query points (optional). | `''` |
| `--random_basis (-R)` | [`flag`](#doc_flag) | Before tree-building, project the data onto a random orthogonal basis. |  |
| `--reference_file (-r)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix containing the reference dataset. | `''` |
| `--sample_at_leaves (-L)` | [`flag`](#doc_flag) | The flag to trigger sampling at leaves. |  |
| `--seed (-s)` | [`int`](#doc_int) | Random seed (if 0, std::time(NULL) is used). | `0` |
| `--single_mode (-S)` | [`flag`](#doc_flag) | If true, single-tree search is used (as opposed to dual-tree search. |  |
| `--single_sample_limit (-z)` | [`int`](#doc_int) | The limit on the maximum number of samples (and hence the largest node you can approximate). | `20` |
| `--tau (-T)` | [`double`](#doc_double) | The allowed rank-error in terms of the percentile of the data. | `5` |
| `--tree_type (-t)` | [`string`](#doc_string) | Type of tree to use: 'kd', 'ub', 'cover', 'r', 'x', 'r-star', 'hilbert-r', 'r-plus', 'r-plus-plus', 'oct'. | `'kd'` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--distances_file (-d)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to output distances into. | 
| `--neighbors_file (-n)` | [`2-d index matrix file`](#doc_a_2_d_index_matrix_file) | Matrix to output neighbors into. | 
| `--output_model_file (-M)` | [`RAModel file`](#doc_model) | If specified, the kNN model will be output here. | 

### Detailed documentation
{: #krann_detailed-documentation }

This program will calculate the k rank-approximate-nearest-neighbors of a set of points. You may specify a separate set of reference points and query points, or just a reference set which will be used as both the reference and query set. You must specify the rank approximation (in %) (and optionally the success probability).

### Example
For example, the following will return 5 neighbors from the top 0.1% of the data (with probability 0.95) for each point in `'input.csv'` and store the distances in `'distances.csv'` and the neighbors in `'neighbors.csv.csv'`:

```bash
$ mlpack_krann --reference_file input.csv --k 5 --distances_file distances.csv
  --neighbors_file neighbors.csv --tau 0.1
```

Note that tau must be set such that the number of points in the corresponding percentile of the data is greater than k.  Thus, if we choose tau = 0.1 with a dataset of 1000 points and k = 5, then we are attempting to choose 5 nearest neighbors out of the closest 1 point -- this is invalid and the program will terminate with an error message.

The output matrices are organized such that row i and column j in the neighbors output file corresponds to the index of the point in the reference set which is the i'th nearest neighbor from the point in the query set with index j.  Row i and column j in the distances output file corresponds to the distance between those two points.

### See also

 - [mlpack_knn](#knn)
 - [mlpack_lsh](#lsh)
 - [Rank-approximate nearest neighbor search: Retaining meaning and speed in high dimensions (pdf)](https://proceedings.neurips.cc/paper_files/paper/2009/file/ddb30680a691d157187ee1cf9e896d03-Paper.pdf)
 - [RASearch C++ class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/rann/ra_search.hpp)

## mlpack_softmax_regression
{: #softmax_regression }

#### Softmax Regression
{: #softmax_regression_descr }

```bash
$ mlpack_softmax_regression [--help] [--info <string>]
        [--input_model_file <string>] [--labels_file <string>] [--lambda 0.0001]
        [--max_iterations 400] [--no_intercept] [--number_of_classes 0]
        [--test_file <string>] [--test_labels_file <string>] [--training_file
        <string>] [--verbose] [--version] [--output_model_file <string>]
        [--predictions_file <string>] [--probabilities_file <string>]
```

An implementation of softmax regression for classification, which is a multiclass generalization of logistic regression.  Given labeled data, a softmax regression model can be trained and saved for future use, or, a pre-trained softmax regression model can be used for classification of new points. [Detailed documentation](#softmax_regression_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_model_file (-m)` | [`SoftmaxRegression<> file`](#doc_model) | File containing existing model (parameters). | `''` |
| `--labels_file (-l)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | A matrix containing labels (0 or 1) for the points in the training set (y). The labels must order as a row. | `''` |
| `--lambda (-r)` | [`double`](#doc_double) | L2-regularization constant | `0.0001` |
| `--max_iterations (-n)` | [`int`](#doc_int) | Maximum number of iterations before termination. | `400` |
| `--no_intercept (-N)` | [`flag`](#doc_flag) | Do not add the intercept term to the model. |  |
| `--number_of_classes (-c)` | [`int`](#doc_int) | Number of classes for classification; if unspecified (or 0), the number of classes found in the labels will be used. | `0` |
| `--test_file (-T)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix containing test dataset. | `''` |
| `--test_labels_file (-L)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | Matrix containing test labels. | `''` |
| `--training_file (-t)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | A matrix containing the training set (the matrix of predictors, X). | `''` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_model_file (-M)` | [`SoftmaxRegression<> file`](#doc_model) | File to save trained softmax regression model to. | 
| `--predictions_file (-p)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | Matrix to save predictions for test dataset into. | 
| `--probabilities_file (-P)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to save class probabilities for test dataset into. | 

### Detailed documentation
{: #softmax_regression_detailed-documentation }

This program performs softmax regression, a generalization of logistic regression to the multiclass case, and has support for L2 regularization.  The program is able to train a model, load  an existing model, and give predictions (and optionally their accuracy) for test data.

Training a softmax regression model is done by giving a file of training points with the `--training_file (-t)` parameter and their corresponding labels with the `--labels_file (-l)` parameter. The number of classes can be manually specified with the `--number_of_classes (-c)` parameter, and the maximum number of iterations of the L-BFGS optimizer can be specified with the `--max_iterations (-n)` parameter.  The L2 regularization constant can be specified with the `--lambda (-r)` parameter and if an intercept term is not desired in the model, the `--no_intercept (-N)` parameter can be specified.

The trained model can be saved with the `--output_model_file (-M)` output parameter. If training is not desired, but only testing is, a model can be loaded with the `--input_model_file (-m)` parameter.  At the current time, a loaded model cannot be trained further, so specifying both `--input_model_file (-m)` and `--training_file (-t)` is not allowed.

The program is also able to evaluate a model on test data.  A test dataset can be specified with the `--test_file (-T)` parameter. Class predictions can be saved with the `--predictions_file (-p)` output parameter.  If labels are specified for the test data with the `--test_labels_file (-L)` parameter, then the program will print the accuracy of the predictions on the given test set and its corresponding labels.

### Example
For example, to train a softmax regression model on the data `'dataset.csv'` with labels `'labels.csv'` with a maximum of 1000 iterations for training, saving the trained model to `'sr_model.bin'`, the following command can be used: 

```bash
$ mlpack_softmax_regression --training_file dataset.csv --labels_file
  labels.csv --output_model_file sr_model.bin
```

Then, to use `'sr_model.bin'` to classify the test points in `'test_points.csv'`, saving the output predictions to `'predictions.csv'`, the following command can be used:

```bash
$ mlpack_softmax_regression --input_model_file sr_model.bin --test_file
  test_points.csv --predictions_file predictions.csv
```

### See also

 - [mlpack_logistic_regression](#logistic_regression)
 - [mlpack_random_forest](#random_forest)
 - [Multinomial logistic regression (softmax regression) on Wikipedia](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)
 - [SoftmaxRegression C++ class documentation](../../user/methods/softmax_regression.md)

## mlpack_sparse_coding
{: #sparse_coding }

#### Sparse Coding
{: #sparse_coding_descr }

```bash
$ mlpack_sparse_coding [--atoms 15] [--help] [--info <string>]
        [--initial_dictionary_file <string>] [--input_model_file <string>]
        [--lambda1 0] [--lambda2 0] [--max_iterations 0] [--newton_tolerance
        1e-06] [--normalize] [--objective_tolerance 0.01] [--seed 0]
        [--test_file <string>] [--training_file <string>] [--verbose]
        [--version] [--codes_file <string>] [--dictionary_file <string>]
        [--output_model_file <string>]
```

An implementation of Sparse Coding with Dictionary Learning.  Given a dataset, this will decompose the dataset into a sparse combination of a few dictionary elements, where the dictionary is learned during computation; a dictionary can be reused for future sparse coding of new points. [Detailed documentation](#sparse_coding_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--atoms (-k)` | [`int`](#doc_int) | Number of atoms in the dictionary. | `15` |
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--initial_dictionary_file (-i)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Optional initial dictionary matrix. | `''` |
| `--input_model_file (-m)` | [`SparseCoding<> file`](#doc_model) | File containing input sparse coding model. | `''` |
| `--lambda1 (-l)` | [`double`](#doc_double) | Sparse coding l1-norm regularization parameter. | `0` |
| `--lambda2 (-L)` | [`double`](#doc_double) | Sparse coding l2-norm regularization parameter. | `0` |
| `--max_iterations (-n)` | [`int`](#doc_int) | Maximum number of iterations for sparse coding (0 indicates no limit). | `0` |
| `--newton_tolerance (-w)` | [`double`](#doc_double) | Tolerance for convergence of Newton method. | `1e-06` |
| `--normalize (-N)` | [`flag`](#doc_flag) | If set, the input data matrix will be normalized before coding. |  |
| `--objective_tolerance (-o)` | [`double`](#doc_double) | Tolerance for convergence of the objective function. | `0.01` |
| `--seed (-s)` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `--test_file (-T)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Optional matrix to be encoded by trained model. | `''` |
| `--training_file (-t)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix of training data (X). | `''` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--codes_file (-c)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to save the output sparse codes of the test matrix (--test_file) to. | 
| `--dictionary_file (-d)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to save the output dictionary to. | 
| `--output_model_file (-M)` | [`SparseCoding<> file`](#doc_model) | File to save trained sparse coding model to. | 

### Detailed documentation
{: #sparse_coding_detailed-documentation }

An implementation of Sparse Coding with Dictionary Learning, which achieves sparsity via an l1-norm regularizer on the codes (LASSO) or an (l1+l2)-norm regularizer on the codes (the Elastic Net).  Given a dense data matrix X with d dimensions and n points, sparse coding seeks to find a dense dictionary matrix D with k atoms in d dimensions, and a sparse coding matrix Z with n points in k dimensions.

The original data matrix X can then be reconstructed as Z * D.  Therefore, this program finds a representation of each point in X as a sparse linear combination of atoms in the dictionary D.

The sparse coding is found with an algorithm which alternates between a dictionary step, which updates the dictionary D, and a sparse coding step, which updates the sparse coding matrix.

Once a dictionary D is found, the sparse coding model may be used to encode other matrices, and saved for future usage.

To run this program, either an input matrix or an already-saved sparse coding model must be specified.  An input matrix may be specified with the `--training_file (-t)` option, along with the number of atoms in the dictionary (specified with the `--atoms (-k)` parameter).  It is also possible to specify an initial dictionary for the optimization, with the `--initial_dictionary_file (-i)` parameter.  An input model may be specified with the `--input_model_file (-m)` parameter.

### Example
As an example, to build a sparse coding model on the dataset `'data.csv'` using 200 atoms and an l1-regularization parameter of 0.1, saving the model into `'model.bin'`, use 

```bash
$ mlpack_sparse_coding --training_file data.csv --atoms 200 --lambda1 0.1
  --output_model_file model.bin
```

Then, this model could be used to encode a new matrix, `'otherdata.csv'`, and save the output codes to `'codes.csv'`: 

```bash
$ mlpack_sparse_coding --input_model_file model.bin --test_file otherdata.csv
  --codes_file codes.csv
```

### See also

 - [mlpack_local_coordinate_coding](#local_coordinate_coding)
 - [Sparse dictionary learning on Wikipedia](https://en.wikipedia.org/wiki/Sparse_dictionary_learning)
 - [Efficient sparse coding algorithms (pdf)](https://proceedings.neurips.cc/paper_files/paper/2006/file/2d71b2ae158c7c5912cc0bbde2bb9d95-Paper.pdf)
 - [Regularization and variable selection via the elastic net](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=46217f372a75dddc2254fdbc6b9418ba3554e453)
 - [SparseCoding C++ class documentation](../../user/methods/sparse_coding.md)

## mlpack_adaboost
{: #adaboost }

#### AdaBoost
{: #adaboost_descr }

```bash
$ mlpack_adaboost [--help] [--info <string>] [--input_model_file
        <string>] [--iterations 1000] [--labels_file <string>] [--test_file
        <string>] [--tolerance 1e-10] [--training_file <string>] [--verbose]
        [--version] [--weak_learner 'decision_stump'] [--output_model_file
        <string>] [--predictions_file <string>] [--probabilities_file <string>]
```

An implementation of the AdaBoost.MH (Adaptive Boosting) algorithm for classification.  This can be used to train an AdaBoost model on labeled data or use an existing AdaBoost model to predict the classes of new points. [Detailed documentation](#adaboost_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_model_file (-m)` | [`AdaBoostModel file`](#doc_model) | Input AdaBoost model. | `''` |
| `--iterations (-i)` | [`int`](#doc_int) | The maximum number of boosting iterations to be run (0 will run until convergence.) | `1000` |
| `--labels_file (-l)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | Labels for the training set. | `''` |
| `--test_file (-T)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Test dataset. | `''` |
| `--tolerance (-e)` | [`double`](#doc_double) | The tolerance for change in values of the weighted error during training. | `1e-10` |
| `--training_file (-t)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Dataset for training AdaBoost. | `''` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |
| `--weak_learner (-w)` | [`string`](#doc_string) | The type of weak learner to use: 'decision_stump', or 'perceptron'. | `'decision_stump'` |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_model_file (-M)` | [`AdaBoostModel file`](#doc_model) | Output trained AdaBoost model. | 
| `--predictions_file (-P)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | Predicted labels for the test set. | 
| `--probabilities_file (-p)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Predicted class probabilities for each point in the test set. | 

### Detailed documentation
{: #adaboost_detailed-documentation }

This program implements the AdaBoost (or Adaptive Boosting) algorithm. The variant of AdaBoost implemented here is AdaBoost.MH. It uses a weak learner, either decision stumps or perceptrons, and over many iterations, creates a strong learner that is a weighted ensemble of weak learners. It runs these iterations until a tolerance value is crossed for change in the value of the weighted training error.

For more information about the algorithm, see the paper "Improved Boosting Algorithms Using Confidence-Rated Predictions", by R.E. Schapire and Y. Singer.

This program allows training of an AdaBoost model, and then application of that model to a test dataset.  To train a model, a dataset must be passed with the `--training_file (-t)` option.  Labels can be given with the `--labels_file (-l)` option; if no labels are specified, the labels will be assumed to be the last column of the input dataset.  Alternately, an AdaBoost model may be loaded with the `--input_model_file (-m)` option.

Once a model is trained or loaded, it may be used to provide class predictions for a given test dataset.  A test dataset may be specified with the `--test_file (-T)` parameter.  The predicted classes for each point in the test dataset are output to the `--predictions_file (-P)` output parameter.  The AdaBoost model itself is output to the `--output_model_file (-M)` output parameter.

### Example
For example, to run AdaBoost on an input dataset `'data.csv'` with labels `'labels.csv'`and perceptrons as the weak learner type, storing the trained model in `'model.bin'`, one could use the following command: 

```bash
$ mlpack_adaboost --training_file data.csv --labels_file labels.csv
  --output_model_file model.bin --weak_learner perceptron
```

Similarly, an already-trained model in `'model.bin'` can be used to provide class predictions from test data `'test_data.csv'` and store the output in `'predictions.csv'` with the following command: 

```bash
$ mlpack_adaboost --input_model_file model.bin --test_file test_data.csv
  --predictions_file predictions.csv
```

### See also

 - [AdaBoost on Wikipedia](https://en.wikipedia.org/wiki/AdaBoost)
 - [Improved boosting algorithms using confidence-rated predictions (pdf)](http://rob.schapire.net/papers/SchapireSi98.pdf)
 - [Perceptron](#perceptron)
 - [Decision Trees](#decision_tree)
 - [AdaBoost C++ class documentation](../../user/methods/adaboost.md)

## mlpack_linear_regression
{: #linear_regression }

#### Simple Linear Regression and Prediction
{: #linear_regression_descr }

```bash
$ mlpack_linear_regression [--help] [--info <string>]
        [--input_model_file <string>] [--lambda 0] [--test_file <string>]
        [--training_file <string>] [--training_responses_file <string>]
        [--verbose] [--version] [--output_model_file <string>]
        [--output_predictions_file <string>]
```

An implementation of simple linear regression and ridge regression using ordinary least squares.  Given a dataset and responses, a model can be trained and saved for later use, or a pre-trained model can be used to output regression predictions for a test set. [Detailed documentation](#linear_regression_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_model_file (-m)` | [`LinearRegression<> file`](#doc_model) | Existing LinearRegression model to use. | `''` |
| `--lambda (-l)` | [`double`](#doc_double) | Tikhonov regularization for ridge regression.  If 0, the method reduces to linear regression. | `0` |
| `--test_file (-T)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix containing X' (test regressors). | `''` |
| `--training_file (-t)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix containing training set X (regressors). | `''` |
| `--training_responses_file (-r)` | [`1-d matrix file`](#doc_a_1_d_matrix_file) | Optional vector containing y (responses). If not given, the responses are assumed to be the last row of the input file. | `''` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_model_file (-M)` | [`LinearRegression<> file`](#doc_model) | Output LinearRegression model. | 
| `--output_predictions_file (-o)` | [`1-d matrix file`](#doc_a_1_d_matrix_file) | If --test_file is specified, this matrix is where the predicted responses will be saved. | 

### Detailed documentation
{: #linear_regression_detailed-documentation }

An implementation of simple linear regression and simple ridge regression using ordinary least squares. This solves the problem

  y = X * b + e

where X (specified by `--training_file (-t)`) and y (specified either as the last column of the input matrix `--training_file (-t)` or via the `--training_responses_file (-r)` parameter) are known and b is the desired variable.  If the covariance matrix (X'X) is not invertible, or if the solution is overdetermined, then specify a Tikhonov regularization constant (with `--lambda (-l)`) greater than 0, which will regularize the covariance matrix to make it invertible.  The calculated b may be saved with the `--output_predictions_file (-o)` output parameter.

Optionally, the calculated value of b is used to predict the responses for another matrix X' (specified by the `--test_file (-T)` parameter):

   y' = X' * b

and the predicted responses y' may be saved with the `--output_predictions_file (-o)` output parameter.  This type of regression is related to least-angle regression, which mlpack implements as the 'lars' program.

### Example
For example, to run a linear regression on the dataset `'X.csv'` with responses `'y.csv'`, saving the trained model to `'lr_model.bin'`, the following command could be used:

```bash
$ mlpack_linear_regression --training_file X.csv --training_responses_file
  y.csv --output_model_file lr_model.bin
```

Then, to use `'lr_model.bin'` to predict responses for a test set `'X_test.csv'`, saving the predictions to `'X_test_responses.csv'`, the following command could be used:

```bash
$ mlpack_linear_regression --input_model_file lr_model.bin --test_file
  X_test.csv --output_predictions_file X_test_responses.csv
```

### See also

 - [mlpack_lars](#lars)
 - [Linear regression on Wikipedia](https://en.wikipedia.org/wiki/Linear_regression)
 - [LinearRegression C++ class documentation](../../user/methods/linear_regression.md)

## mlpack_preprocess_imputer
{: #preprocess_imputer }

#### Impute Data
{: #preprocess_imputer_descr }

```bash
$ mlpack_preprocess_imputer [--custom_value 0] [--dimension 0] [--help]
        [--info <string>] --input_file <string> --missing_value <string>
        --strategy <string> [--verbose] [--version] [--output_file <string>]
```

This utility provides several imputation strategies for missing data. Given a dataset with missing values, this can impute according to several strategies, including user-defined values. [Detailed documentation](#preprocess_imputer_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--custom_value (-c)` | [`double`](#doc_double) | User-defined custom imputation value. | `0` |
| `--dimension (-d)` | [`int`](#doc_int) | The dimension to apply imputation to. | `0` |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_file (-i)` | [`string`](#doc_string) | File containing data. | `**--**` |
| `--missing_value (-m)` | [`string`](#doc_string) | User defined missing value. | `**--**` |
| `--strategy (-s)` | [`string`](#doc_string) | imputation strategy to be applied. Strategies should be one of 'custom', 'mean', 'median', and 'listwise_deletion'. | `**--**` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_file (-o)` | [`string`](#doc_string) | File to save output into. | 

### Detailed documentation
{: #preprocess_imputer_detailed-documentation }

This utility takes a dataset and converts a user-defined missing variable to another to provide more meaningful analysis.

The program does not modify the original file, but instead makes a separate file to save the output data; You can save the output by specifying the file name with`--output_file (-o)`.

### Example
For example, if we consider 'NULL' in dimension 0 to be a missing variable and want to delete whole row containing the NULL in the column-wise`'dataset.csv'`, and save the result to `'result.csv'`, we could run :

```bash
$ mlpack_preprocess_imputer --input_file dataset --output_file result
  --missing_value NULL --dimension 0 --strategy listwise_deletion
```

### See also

 - [mlpack_preprocess_binarize](#preprocess_binarize)
 - [mlpack_preprocess_describe](#preprocess_describe)
 - [mlpack_preprocess_split](#preprocess_split)

## mlpack_image_converter
{: #image_converter }

#### Image Converter
{: #image_converter_descr }

```bash
$ mlpack_image_converter [--channels 0] [--dataset_file <string>]
        [--height 0] [--help] [--info <string>] --input [] [--quality 90]
        [--save] [--verbose] [--version] [--width 0] [--output_file <string>]
```

A utility to load an image or set of images into a single dataset that can then be used by other mlpack methods and utilities. This can also unpack an image dataset into individual files, for instance after mlpack methods have been used. [Detailed documentation](#image_converter_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--channels (-c)` | [`int`](#doc_int) | Number of channels in the image. | `0` |
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--dataset_file (-I)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Input matrix to save as images. | `''` |
| `--height (-H)` | [`int`](#doc_int) | Height of the images. | `0` |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input (-i)` | [`string vector`](#doc_string_vector) | Image filenames which have to be loaded/saved. | `**--**` |
| `--quality (-q)` | [`int`](#doc_int) | Compression of the image if saved as jpg (0-100). | `90` |
| `--save (-s)` | [`flag`](#doc_flag) | Save a dataset as images. |  |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |
| `--width (-w)` | [`int`](#doc_int) | Width of the image. | `0` |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_file (-o)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix to save images data to, Onlyneeded if you are specifying 'save' option. | 

### Detailed documentation
{: #image_converter_detailed-documentation }

This utility takes an image or an array of images and loads them to a matrix. You can optionally specify the height `--height (-H)` width `--width (-w)` and channel `--channels (-c)` of the images that needs to be loaded; otherwise, these parameters will be automatically detected from the image.
There are other options too, that can be specified such as `--quality (-q)`.

You can also provide a dataset and save them as images using `--dataset_file (-I)` and `--save (-s)` as an parameter.

### Example
 An example to load an image : 

```bash
$ mlpack_image_converter --input X --height 256 --width 256 --channels 3
  --output_file Y.csv
```

 An example to save an image is :

```bash
$ mlpack_image_converter --input X --height 256 --width 256 --channels 3
  --dataset_file Y.csv --save
```

### See also

 - [mlpack_preprocess_binarize](#preprocess_binarize)
 - [mlpack_preprocess_describe](#preprocess_describe)

## mlpack_range_search
{: #range_search }

#### Range Search
{: #range_search_descr }

```bash
$ mlpack_range_search [--help] [--info <string>] [--input_model_file
        <string>] [--leaf_size 20] [--max 0] [--min 0] [--naive] [--query_file
        <string>] [--random_basis] [--reference_file <string>] [--seed 0]
        [--single_mode] [--tree_type 'kd'] [--verbose] [--version]
        [--distances_file <string>] [--neighbors_file <string>]
        [--output_model_file <string>]
```

An implementation of range search with single-tree and dual-tree algorithms.  Given a set of reference points and a set of query points and a range, this can find the set of reference points within the desired range for each query point, and any trees built during the computation can be saved for reuse with future range searches. [Detailed documentation](#range_search_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--input_model_file (-m)` | [`RSModel file`](#doc_model) | File containing pre-trained range search model. | `''` |
| `--leaf_size (-l)` | [`int`](#doc_int) | Leaf size for tree building (used for kd-trees, vp trees, random projection trees, UB trees, R trees, R* trees, X trees, Hilbert R trees, R+ trees, R++ trees, and octrees). | `20` |
| `--max (-U)` | [`double`](#doc_double) | Upper bound in range (if not specified, +inf will be used. | `0` |
| `--min (-L)` | [`double`](#doc_double) | Lower bound in range. | `0` |
| `--naive (-N)` | [`flag`](#doc_flag) | If true, O(n^2) naive mode is used for computation. |  |
| `--query_file (-q)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | File containing query points (optional). | `''` |
| `--random_basis (-R)` | [`flag`](#doc_flag) | Before tree-building, project the data onto a random orthogonal basis. |  |
| `--reference_file (-r)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | Matrix containing the reference dataset. | `''` |
| `--seed (-s)` | [`int`](#doc_int) | Random seed (if 0, std::time(NULL) is used). | `0` |
| `--single_mode (-S)` | [`flag`](#doc_flag) | If true, single-tree search is used (as opposed to dual-tree search). |  |
| `--tree_type (-t)` | [`string`](#doc_string) | Type of tree to use: 'kd', 'vp', 'rp', 'max-rp', 'ub', 'cover', 'r', 'r-star', 'x', 'ball', 'hilbert-r', 'r-plus', 'r-plus-plus', 'oct'. | `'kd'` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--distances_file (-d)` | [`string`](#doc_string) | File to output distances into. | 
| `--neighbors_file (-n)` | [`string`](#doc_string) | File to output neighbors into. | 
| `--output_model_file (-M)` | [`RSModel file`](#doc_model) | If specified, the range search model will be saved to the given file. | 

### Detailed documentation
{: #range_search_detailed-documentation }

This program implements range search with a Euclidean distance metric. For a given query point, a given range, and a given set of reference points, the program will return all of the reference points with distance to the query point in the given range.  This is performed for an entire set of query points. You may specify a separate set of reference and query points, or only a reference set -- which is then used as both the reference and query set.  The given range is taken to be inclusive (that is, points with a distance exactly equal to the minimum and maximum of the range are included in the results).

### Example
For example, the following will calculate the points within the range `[2, 5]` of each point in `'input.csv'` and store the distances in`'distances.csv'` and the neighbors in `'neighbors.csv'`

```bash
$ mlpack_range_search --min 2 --max 5 --distances_file input --distances_file
  distances --neighbors_file neighbors
```

The output files are organized such that line i corresponds to the points found for query point i.  Because sometimes 0 points may be found in the given range, lines of the output files may be empty.  The points are not ordered in any specific manner.

Because the number of points returned for each query point may differ, the resultant CSV-like files may not be loadable by many programs.  However, at this time a better way to store this non-square result is not known.  As a result, any output files will be written as CSVs in this manner, regardless of the given extension.

### See also

 - [mlpack_knn](#knn)
 - [Range search tutorial](../../tutorials/range_search.md)
 - [Range searching on Wikipedia](https://en.wikipedia.org/wiki/Range_searching)
 - [Tree-independent dual-tree algorithms (pdf)](http://proceedings.mlr.press/v28/curtin13.pdf)
 - [RangeSearch C++ class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/range_search/range_search.hpp)

