# mlpack Go binding documentation

## mlpack overview

mlpack is an intuitive, fast, and flexible header-only C++ machine learning library with bindings to other languages.  It aims to provide fast, lightweight implementations of both common and cutting-edge machine learning algorithms.

This reference page details mlpack's bindings to Go.

Further useful mlpack documentation links are given below.

 - [mlpack homepage](https://www.mlpack.org/)
 - [mlpack on Github](https://github.com/mlpack/mlpack)
 - [mlpack main documentation page](https://www.mlpack.org/doc/index.html)

See also the quickstart guide for Go:

 - [Go Quickstart](../../quickstart/go.md)

## Data Formats

<div id="data-formats-div" markdown="1">
mlpack bindings for Go take and return a restricted set of types, for simplicity.  These include primitive types, matrix/vector types, categorical matrix types, and model types. Each type is detailed below.

 - `int`{: #doc_int}: An integer (i.e., `1`).
 - `float64`{: #doc_float64 }: A floating-point number (i.e., `0.5`).
 - `bool`{: #doc_bool }: A boolean flag option (`true` or `false`).
 - `string`{: #doc_string }: A character string (i.e., `"hello"`).
 - `array of ints`{: #doc_array_of_ints }: An array of integers; i.e., `[]int{0, 1, 2}`.
 - `array of strings`{: #doc_array_of_strings }: An array of strings; i.e., `[]string{"hello", "goodbye"}`.
 - `*mat.Dense`{: #doc_a__mat_Dense }: A 2-d gonum Matrix. If the type is not already `float64`, it will be converted.
 - `*mat.Dense (1d)`{: #doc_a__mat_Dense__1d_ }: A 1-d gonum Matrix (that is, a Matrix where either the number of rows or number of columns is 1).
 - `matrixWithInfo`{: #doc_matrixWithInfo }: A Tuple(matrixWithInfo) containing `float64` data (Data) along with a boolean array (Categoricals) indicating which dimensions are categorical (represented by `true`) and which are numeric (represented by `false`).  The number of elements in the boolean array should be the same as the dimensionality of the data matrix.  It is expected that each row of the matrix corresponds to a single data point when calling mlpack bindings.
 - `mlpackModel`{: #doc_model }: An mlpack model pointer.  This type holds a pointer to C++ memory containing the mlpack model.  Note that this means the mlpack model itself cannot be easily inspected in Go.  However, the pointer can be passed to subsequent calls to mlpack functions.
</div>


## ApproxKfn()
{: #approx_kfn }

#### Approximate furthest neighbor search
{: #approx_kfn_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for ApproxKfn().
param := mlpack.ApproxKfnOptions()
param.Algorithm = "ds"
param.CalculateError = false
param.ExactDistances = mat.NewDense(1, 1, nil)
param.InputModel = nil
param.K = 0
param.NumProjections = 5
param.NumTables = 5
param.Query = mat.NewDense(1, 1, nil)
param.Reference = mat.NewDense(1, 1, nil)
param.Verbose = false

distances, neighbors, output_model := mlpack.ApproxKfn(param)
```

An implementation of two strategies for furthest neighbor search.  This can be used to compute the furthest neighbor of query point(s) from a set of points; furthest neighbor models can be saved and reused with future query point(s). [Detailed documentation](#approx_kfn_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `Algorithm` | [`string`](#doc_string) | Algorithm to use: 'ds' or 'qdafn'. | `"ds"` |
| `CalculateError` | [`bool`](#doc_bool) | If set, calculate the average distance error for the first furthest neighbor only. | `false` |
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `ExactDistances` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix containing exact distances to furthest neighbors; this can be used to avoid explicit calculation when --calculate_error is set. | `mat.NewDense(1, 1, nil)` |
| `InputModel` | [`approxkfnModel`](#doc_model) | File containing input model. | `nil` |
| `K` | [`int`](#doc_int) | Number of furthest neighbors to search for. | `0` |
| `NumProjections` | [`int`](#doc_int) | Number of projections to use in each hash table. | `5` |
| `NumTables` | [`int`](#doc_int) | Number of hash tables to use. | `5` |
| `Query` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix containing query points. | `mat.NewDense(1, 1, nil)` |
| `Reference` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix containing the reference dataset. | `mat.NewDense(1, 1, nil)` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Distances` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to save furthest neighbor distances to. | 
| `Neighbors` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to save neighbor indices to. | 
| `OutputModel` | [`approxkfnModel`](#doc_model) | File to save output model to. | 

### Detailed documentation
{: #approx_kfn_detailed-documentation }

This program implements two strategies for furthest neighbor search. These strategies are:

 - The 'qdafn' algorithm from "Approximate Furthest Neighbor in High Dimensions" by R. Pagh, F. Silvestri, J. Sivertsen, and M. Skala, in Similarity Search and Applications 2015 (SISAP).
 - The 'DrusillaSelect' algorithm from "Fast approximate furthest neighbors with data-dependent candidate selection", by R.R. Curtin and A.B. Gardner, in Similarity Search and Applications 2016 (SISAP).

These two strategies give approximate results for the furthest neighbor search problem and can be used as fast replacements for other furthest neighbor techniques such as those found in the mlpack_kfn program.  Note that typically, the 'ds' algorithm requires far fewer tables and projections than the 'qdafn' algorithm.

Specify a reference set (set to search in) with `Reference`, specify a query set with `Query`, and specify algorithm parameters with `NumTables` and `NumProjections` (or don't and defaults will be used).  The algorithm to be used (either 'ds'---the default---or 'qdafn')  may be specified with `Algorithm`.  Also specify the number of neighbors to search for with `K`.

Note that for 'qdafn' in lower dimensions, `NumProjections` may need to be set to a high value in order to return results for each query point.

If no query set is specified, the reference set will be used as the query set.  The `OutputModel` output parameter may be used to store the built model, and an input model may be loaded instead of specifying a reference set with the `InputModel` option.

Results for each query point can be stored with the `Neighbors` and `Distances` output parameters.  Each row of these output matrices holds the k distances or neighbor indices for each query point.

### Example
For example, to find the 5 approximate furthest neighbors with `reference_set` as the reference set and `query_set` as the query set using DrusillaSelect, storing the furthest neighbor indices to `neighbors` and the furthest neighbor distances to `distances`, one could call

```go
// Initialize optional parameters for ApproxKfn().
param := mlpack.ApproxKfnOptions()
param.Query = query_set
param.Reference = reference_set
param.K = 5
param.Algorithm = "ds"

distances, neighbors, _ := mlpack.ApproxKfn(param)
```

and to perform approximate all-furthest-neighbors search with k=1 on the set `data` storing only the furthest neighbor distances to `distances`, one could call

```go
// Initialize optional parameters for ApproxKfn().
param := mlpack.ApproxKfnOptions()
param.Reference = reference_set
param.K = 1

distances, _, _ := mlpack.ApproxKfn(param)
```

A trained model can be re-used.  If a model has been previously saved to `model`, then we may find 3 approximate furthest neighbors on a query set `new_query_set` using that model and store the furthest neighbor indices into `neighbors` by calling

```go
// Initialize optional parameters for ApproxKfn().
param := mlpack.ApproxKfnOptions()
param.InputModel = &model
param.Query = new_query_set
param.K = 3

_, neighbors, _ := mlpack.ApproxKfn(param)
```

### See also

 - [k-furthest-neighbor search](#kfn)
 - [k-nearest-neighbor search](#knn)
 - [Fast approximate furthest neighbors with data-dependent candidate selection (pdf)](http://ratml.org/pub/pdf/2016fast.pdf)
 - [Approximate furthest neighbor in high dimensions (pdf)](https://www.rasmuspagh.net/papers/approx-furthest-neighbor-SISAP15.pdf)
 - [QDAFN class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/approx_kfn/qdafn.hpp)
 - [DrusillaSelect class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/approx_kfn/drusilla_select.hpp)

## BayesianLinearRegression()
{: #bayesian_linear_regression }

#### BayesianLinearRegression
{: #bayesian_linear_regression_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for BayesianLinearRegression().
param := mlpack.BayesianLinearRegressionOptions()
param.Center = false
param.Input = mat.NewDense(1, 1, nil)
param.InputModel = nil
param.Responses = mat.NewDense(1, 1, nil)
param.Scale = false
param.Test = mat.NewDense(1, 1, nil)
param.Verbose = false

output_model, predictions, stds := mlpack.BayesianLinearRegression(param)
```

An implementation of the bayesian linear regression. [Detailed documentation](#bayesian_linear_regression_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `Center` | [`bool`](#doc_bool) | Center the data and fit the intercept if enabled. | `false` |
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `Input` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix of covariates (X). | `mat.NewDense(1, 1, nil)` |
| `InputModel` | [`bayesianLinearRegression`](#doc_model) | Trained BayesianLinearRegression model to use. | `nil` |
| `Responses` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | Matrix of responses/observations (y). | `mat.NewDense(1, 1, nil)` |
| `Scale` | [`bool`](#doc_bool) | Scale each feature by their standard deviations if enabled. | `false` |
| `Test` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix containing points to regress on (test points). | `mat.NewDense(1, 1, nil)` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `OutputModel` | [`bayesianLinearRegression`](#doc_model) | Output BayesianLinearRegression model. | 
| `Predictions` | [`*mat.Dense`](#doc_a__mat_Dense) | If --test_file is specified, this file is where the predicted responses will be saved. | 
| `Stds` | [`*mat.Dense`](#doc_a__mat_Dense) | If specified, this is where the standard deviations of the predictive distribution will be saved. | 

### Detailed documentation
{: #bayesian_linear_regression_detailed-documentation }

An implementation of the bayesian linear regression.
This model is a probabilistic view and implementation of the linear regression. The final solution is obtained by computing a posterior distribution from gaussian likelihood and a zero mean gaussian isotropic  prior distribution on the solution. 
Optimization is AUTOMATIC and does not require cross validation. The optimization is performed by maximization of the evidence function. Parameters are tuned during the maximization of the marginal likelihood. This procedure includes the Ockham's razor that penalizes over complex solutions. 

This program is able to train a Bayesian linear regression model or load a model from file, output regression predictions for a test set, and save the trained model to a file.

To train a BayesianLinearRegression model, the `Input` and `Responses`parameters must be given. The `Center`and `Scale` parameters control the centering and the normalizing options. A trained model can be saved with the `OutputModel`. If no training is desired at all, a model can be passed via the `InputModel` parameter.

The program can also provide predictions for test data using either the trained model or the given input model.  Test points can be specified with the `Test` parameter.  Predicted responses to the test points can be saved with the `Predictions` output parameter. The corresponding standard deviation can be save by precising the `Stds` parameter.

### Example
For example, the following command trains a model on the data `data` and responses `responses`with center set to true and scale set to false (so, Bayesian linear regression is being solved, and then the model is saved to `blr_model`:

```go
// Initialize optional parameters for BayesianLinearRegression().
param := mlpack.BayesianLinearRegressionOptions()
param.Input = data
param.Responses = responses
param.Center = 1
param.Scale = 0

blr_model, _, _ := mlpack.BayesianLinearRegression(param)
```

The following command uses the `blr_model` to provide predicted  responses for the data `test` and save those  responses to `test_predictions`: 

```go
// Initialize optional parameters for BayesianLinearRegression().
param := mlpack.BayesianLinearRegressionOptions()
param.InputModel = &blr_model
param.Test = test

_, test_predictions, _ := mlpack.BayesianLinearRegression(param)
```

Because the estimator computes a predictive distribution instead of a simple point estimate, the `Stds` parameter allows one to save the prediction uncertainties: 

```go
// Initialize optional parameters for BayesianLinearRegression().
param := mlpack.BayesianLinearRegressionOptions()
param.InputModel = &blr_model
param.Test = test

_, test_predictions, stds := mlpack.BayesianLinearRegression(param)
```

### See also

 - [Bayesian Interpolation](https://cs.uwaterloo.ca/~mannr/cs886-w10/mackay-bayesian.pdf)
 - [Bayesian Linear Regression, Section 3.3](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
 - [BayesianLinearRegression C++ class documentation](../../user/methods/bayesian_linear_regression.md)

## Cf()
{: #cf }

#### Collaborative Filtering
{: #cf_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for Cf().
param := mlpack.CfOptions()
param.Algorithm = "NMF"
param.AllUserRecommendations = false
param.InputModel = nil
param.Interpolation = "average"
param.IterationOnlyTermination = false
param.MaxIterations = 1000
param.MinResidue = 1e-05
param.NeighborSearch = "euclidean"
param.Neighborhood = 5
param.Normalization = "none"
param.Query = mat.NewDense(1, 1, nil)
param.Rank = 0
param.Recommendations = 5
param.Seed = 0
param.Test = mat.NewDense(1, 1, nil)
param.Training = mat.NewDense(1, 1, nil)
param.Verbose = false

output, output_model := mlpack.Cf(param)
```

An implementation of several collaborative filtering (CF) techniques for recommender systems.  This can be used to train a new CF model, or use an existing CF model to compute recommendations. [Detailed documentation](#cf_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `Algorithm` | [`string`](#doc_string) | Algorithm used for matrix factorization. | `"NMF"` |
| `AllUserRecommendations` | [`bool`](#doc_bool) | Generate recommendations for all users. | `false` |
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `InputModel` | [`cfModel`](#doc_model) | Trained CF model to load. | `nil` |
| `Interpolation` | [`string`](#doc_string) | Algorithm used for weight interpolation. | `"average"` |
| `IterationOnlyTermination` | [`bool`](#doc_bool) | Terminate only when the maximum number of iterations is reached. | `false` |
| `MaxIterations` | [`int`](#doc_int) | Maximum number of iterations. If set to zero, there is no limit on the number of iterations. | `1000` |
| `MinResidue` | [`float64`](#doc_float64) | Residue required to terminate the factorization (lower values generally mean better fits). | `1e-05` |
| `NeighborSearch` | [`string`](#doc_string) | Algorithm used for neighbor search. | `"euclidean"` |
| `Neighborhood` | [`int`](#doc_int) | Size of the neighborhood of similar users to consider for each query user. | `5` |
| `Normalization` | [`string`](#doc_string) | Normalization performed on the ratings. | `"none"` |
| `Query` | [`*mat.Dense`](#doc_a__mat_Dense) | List of query users for which recommendations should be generated. | `mat.NewDense(1, 1, nil)` |
| `Rank` | [`int`](#doc_int) | Rank of decomposed matrices (if 0, a heuristic is used to estimate the rank). | `0` |
| `Recommendations` | [`int`](#doc_int) | Number of recommendations to generate for each query user. | `5` |
| `Seed` | [`int`](#doc_int) | Set the random seed (0 uses std::time(NULL)). | `0` |
| `Test` | [`*mat.Dense`](#doc_a__mat_Dense) | Test set to calculate RMSE on. | `mat.NewDense(1, 1, nil)` |
| `Training` | [`*mat.Dense`](#doc_a__mat_Dense) | Input dataset to perform CF on. | `mat.NewDense(1, 1, nil)` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Output` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix that will store output recommendations. | 
| `OutputModel` | [`cfModel`](#doc_model) | Output for trained CF model. | 

### Detailed documentation
{: #cf_detailed-documentation }

This program performs collaborative filtering (CF) on the given dataset. Given a list of user, item and preferences (the `Training` parameter), the program will perform a matrix decomposition and then can perform a series of actions related to collaborative filtering.  Alternately, the program can load an existing saved CF model with the `InputModel` parameter and then use that model to provide recommendations or predict values.

The input matrix should be a 3-dimensional matrix of ratings, where the first dimension is the user, the second dimension is the item, and the third dimension is that user's rating of that item.  Both the users and items should be numeric indices, not names. The indices are assumed to start from 0.

A set of query users for which recommendations can be generated may be specified with the `Query` parameter; alternately, recommendations may be generated for every user in the dataset by specifying the `AllUserRecommendations` parameter.  In addition, the number of recommendations per user to generate can be specified with the `Recommendations` parameter, and the number of similar users (the size of the neighborhood) to be considered when generating recommendations can be specified with the `Neighborhood` parameter.

For performing the matrix decomposition, the following optimization algorithms can be specified via the `Algorithm` parameter:

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


The following neighbor search algorithms can be specified via the `NeighborSearch` parameter:

 - 'cosine'  -- Cosine Search Algorithm
 - 'euclidean'  -- Euclidean Search Algorithm
 - 'pearson'  -- Pearson Search Algorithm


The following weight interpolation algorithms can be specified via the `Interpolation` parameter:

 - 'average'  -- Average Interpolation Algorithm
 - 'regression'  -- Regression Interpolation Algorithm
 - 'similarity'  -- Similarity Interpolation Algorithm


The following ranking normalization algorithms can be specified via the `Normalization` parameter:

 - 'none'  -- No Normalization
 - 'item_mean'  -- Item Mean Normalization
 - 'overall_mean'  -- Overall Mean Normalization
 - 'user_mean'  -- User Mean Normalization
 - 'z_score'  -- Z-Score Normalization

A trained model may be saved to with the `OutputModel` output parameter.

### Example
To train a CF model on a dataset `training_set` using NMF for decomposition and saving the trained model to `model`, one could call: 

```go
// Initialize optional parameters for Cf().
param := mlpack.CfOptions()
param.Training = training_set
param.Algorithm = "NMF"

_, model := mlpack.Cf(param)
```

Then, to use this model to generate recommendations for the list of users in the query set `users`, storing 5 recommendations in `recommendations`, one could call 

```go
// Initialize optional parameters for Cf().
param := mlpack.CfOptions()
param.InputModel = &model
param.Query = users
param.Recommendations = 5

recommendations, _ := mlpack.Cf(param)
```

### See also

 - [Collaborative Filtering on Wikipedia](https://en.wikipedia.org/wiki/Collaborative_filtering)
 - [Matrix factorization on Wikipedia](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems))
 - [Matrix factorization techniques for recommender systems (pdf)](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=cf17f85a0a7991fa01dbfb3e5878fbf71ea4bdc5)
 - [CFType class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/cf/cf.hpp)

## Dbscan()
{: #dbscan }

#### DBSCAN clustering
{: #dbscan_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for Dbscan().
param := mlpack.DbscanOptions()
param.Epsilon = 1
param.MinSize = 5
param.Naive = false
param.SelectionType = "ordered"
param.SingleMode = false
param.TreeType = "kd"
param.Verbose = false

assignments, centroids := mlpack.Dbscan(input, param)
```

An implementation of DBSCAN clustering.  Given a dataset, this can compute and return a clustering of that dataset. [Detailed documentation](#dbscan_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `Epsilon` | [`float64`](#doc_float64) | Radius of each range search. | `1` |
| `input` | [`*mat.Dense`](#doc_a__mat_Dense) | Input dataset to cluster. | `**--**` |
| `MinSize` | [`int`](#doc_int) | Minimum number of points for a cluster. | `5` |
| `Naive` | [`bool`](#doc_bool) | If set, brute-force range search (not tree-based) will be used. | `false` |
| `SelectionType` | [`string`](#doc_string) | If using point selection policy, the type of selection to use ('ordered', 'random'). | `"ordered"` |
| `SingleMode` | [`bool`](#doc_bool) | If set, single-tree range search (not dual-tree) will be used. | `false` |
| `TreeType` | [`string`](#doc_string) | If using single-tree or dual-tree search, the type of tree to use ('kd', 'r', 'r-star', 'x', 'hilbert-r', 'r-plus', 'r-plus-plus', 'cover', 'ball'). | `"kd"` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Assignments` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | Output matrix for assignments of each point. | 
| `Centroids` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to save output centroids to. | 

### Detailed documentation
{: #dbscan_detailed-documentation }

This program implements the DBSCAN algorithm for clustering using accelerated tree-based range search.  The type of tree that is used may be parameterized, or brute-force range search may also be used.

The input dataset to be clustered may be specified with the `Input` parameter; the radius of each range search may be specified with the `Epsilon` parameters, and the minimum number of points in a cluster may be specified with the `MinSize` parameter.

The `Assignments` and `Centroids` output parameters may be used to save the output of the clustering. `Assignments` contains the cluster assignments of each point, and `Centroids` contains the centroids of each cluster.

The range search may be controlled with the `TreeType`, `SingleMode`, and `Naive` parameters.  `TreeType` can control the type of tree used for range search; this can take a variety of values: 'kd', 'r', 'r-star', 'x', 'hilbert-r', 'r-plus', 'r-plus-plus', 'cover', 'ball'. The `SingleMode` parameter will force single-tree search (as opposed to the default dual-tree search), and '`Naive` will force brute-force range search.

### Example
An example usage to run DBSCAN on the dataset in `input` with a radius of 0.5 and a minimum cluster size of 5 is given below:

```go
// Initialize optional parameters for Dbscan().
param := mlpack.DbscanOptions()
param.Epsilon = 0.5
param.MinSize = 5

_, _ := mlpack.Dbscan(input, param)
```

### See also

 - [DBSCAN on Wikipedia](https://en.wikipedia.org/wiki/DBSCAN)
 - [A density-based algorithm for discovering clusters in large spatial databases with noise (pdf)](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf)
 - [DBSCAN class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/dbscan/dbscan.hpp)

## DecisionTree()
{: #decision_tree }

#### Decision tree
{: #decision_tree_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for DecisionTree().
param := mlpack.DecisionTreeOptions()
param.InputModel = nil
param.Labels = mat.NewDense(1, 1, nil)
param.MaximumDepth = 0
param.MinimumGainSplit = 1e-07
param.MinimumLeafSize = 20
param.PrintTrainingAccuracy = false
param.Test = mat.NewDense(1, 1, nil)
param.TestLabels = mat.NewDense(1, 1, nil)
param.Training = mat.NewDense(1, 1, nil)
param.Verbose = false
param.Weights = mat.NewDense(1, 1, nil)

output_model, predictions, probabilities := mlpack.DecisionTree(param)
```

An implementation of an ID3-style decision tree for classification, which supports categorical data.  Given labeled data with numeric or categorical features, a decision tree can be trained and saved; or, an existing decision tree can be used for classification on new points. [Detailed documentation](#decision_tree_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `InputModel` | [`decisionTreeModel`](#doc_model) | Pre-trained decision tree, to be used with test points. | `nil` |
| `Labels` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | Training labels. | `mat.NewDense(1, 1, nil)` |
| `MaximumDepth` | [`int`](#doc_int) | Maximum depth of the tree (0 means no limit). | `0` |
| `MinimumGainSplit` | [`float64`](#doc_float64) | Minimum gain for node splitting. | `1e-07` |
| `MinimumLeafSize` | [`int`](#doc_int) | Minimum number of points in a leaf. | `20` |
| `PrintTrainingAccuracy` | [`bool`](#doc_bool) | Print the training accuracy. | `false` |
| `Test` | [`matrixWithInfo`](#doc_matrixWithInfo) | Testing dataset (may be categorical). | `mat.NewDense(1, 1, nil)` |
| `TestLabels` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | Test point labels, if accuracy calculation is desired. | `mat.NewDense(1, 1, nil)` |
| `Training` | [`matrixWithInfo`](#doc_matrixWithInfo) | Training dataset (may be categorical). | `mat.NewDense(1, 1, nil)` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |
| `Weights` | [`*mat.Dense`](#doc_a__mat_Dense) | The weight of labels | `mat.NewDense(1, 1, nil)` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `OutputModel` | [`decisionTreeModel`](#doc_model) | Output for trained decision tree. | 
| `Predictions` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | Class predictions for each test point. | 
| `Probabilities` | [`*mat.Dense`](#doc_a__mat_Dense) | Class probabilities for each test point. | 

### Detailed documentation
{: #decision_tree_detailed-documentation }

Train and evaluate using a decision tree.  Given a dataset containing numeric or categorical features, and associated labels for each point in the dataset, this program can train a decision tree on that data.

The training set and associated labels are specified with the `Training` and `Labels` parameters, respectively.  The labels should be in the range `[0, num_classes - 1]`. Optionally, if `Labels` is not specified, the labels are assumed to be the last dimension of the training dataset.

When a model is trained, the `OutputModel` output parameter may be used to save the trained model.  A model may be loaded for predictions with the `InputModel` parameter.  The `InputModel` parameter may not be specified when the `Training` parameter is specified.  The `MinimumLeafSize` parameter specifies the minimum number of training points that must fall into each leaf for it to be split.  The `MinimumGainSplit` parameter specifies the minimum gain that is needed for the node to split.  The `MaximumDepth` parameter specifies the maximum depth of the tree.  If `PrintTrainingAccuracy` is specified, the training accuracy will be printed.

Test data may be specified with the `Test` parameter, and if performance numbers are desired for that test set, labels may be specified with the `TestLabels` parameter.  Predictions for each test point may be saved via the `Predictions` output parameter.  Class probabilities for each prediction may be saved with the `Probabilities` output parameter.

### Example
For example, to train a decision tree with a minimum leaf size of 20 on the dataset contained in `data` with labels `labels`, saving the output model to `tree` and printing the training error, one could call

```go
// Initialize optional parameters for DecisionTree().
param := mlpack.DecisionTreeOptions()
param.Training = data
param.Labels = labels
param.MinimumLeafSize = 20
param.MinimumGainSplit = 0.001
param.PrintTrainingAccuracy = true

tree, _, _ := mlpack.DecisionTree(param)
```

Then, to use that model to classify points in `test_set` and print the test error given the labels `test_labels` using that model, while saving the predictions for each point to `predictions`, one could call 

```go
// Initialize optional parameters for DecisionTree().
param := mlpack.DecisionTreeOptions()
param.InputModel = &tree
param.Test = test_set
param.TestLabels = test_labels

_, predictions, _ := mlpack.DecisionTree(param)
```

### See also

 - [Random forest](#random_forest)
 - [Decision trees on Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning)
 - [Induction of Decision Trees (pdf)](https://www.hunch.net/~coms-4771/quinlan.pdf)
 - [DecisionTree C++ class documentation](../../user/methods/decision_tree.md)

## Det()
{: #det }

#### Density Estimation With Density Estimation Trees
{: #det_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for Det().
param := mlpack.DetOptions()
param.Folds = 10
param.InputModel = nil
param.MaxLeafSize = 10
param.MinLeafSize = 5
param.PathFormat = "lr"
param.SkipPruning = false
param.Test = mat.NewDense(1, 1, nil)
param.Training = mat.NewDense(1, 1, nil)
param.Verbose = false

output_model, tag_counters_file, tag_file, test_set_estimates,
    training_set_estimates, vi := mlpack.Det(param)
```

An implementation of density estimation trees for the density estimation task.  Density estimation trees can be trained or used to predict the density at locations given by query points. [Detailed documentation](#det_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `Folds` | [`int`](#doc_int) | The number of folds of cross-validation to perform for the estimation (0 is LOOCV) | `10` |
| `InputModel` | [`dTree`](#doc_model) | Trained density estimation tree to load. | `nil` |
| `MaxLeafSize` | [`int`](#doc_int) | The maximum size of a leaf in the unpruned, fully grown DET. | `10` |
| `MinLeafSize` | [`int`](#doc_int) | The minimum size of a leaf in the unpruned, fully grown DET. | `5` |
| `PathFormat` | [`string`](#doc_string) | The format of path printing: 'lr', 'id-lr', or 'lr-id'. | `"lr"` |
| `SkipPruning` | [`bool`](#doc_bool) | Whether to bypass the pruning process and output the unpruned tree only. | `false` |
| `Test` | [`*mat.Dense`](#doc_a__mat_Dense) | A set of test points to estimate the density of. | `mat.NewDense(1, 1, nil)` |
| `Training` | [`*mat.Dense`](#doc_a__mat_Dense) | The data set on which to build a density estimation tree. | `mat.NewDense(1, 1, nil)` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `OutputModel` | [`dTree`](#doc_model) | Output to save trained density estimation tree to. | 
| `TagCountersFile` | [`string`](#doc_string) | The file to output the number of points that went to each leaf. | 
| `TagFile` | [`string`](#doc_string) | The file to output the tags (and possibly paths) for each sample in the test set. | 
| `TestSetEstimates` | [`*mat.Dense`](#doc_a__mat_Dense) | The output estimates on the test set from the final optimally pruned tree. | 
| `TrainingSetEstimates` | [`*mat.Dense`](#doc_a__mat_Dense) | The output density estimates on the training set from the final optimally pruned tree. | 
| `Vi` | [`*mat.Dense`](#doc_a__mat_Dense) | The output variable importance values for each feature. | 

### Detailed documentation
{: #det_detailed-documentation }

This program performs a number of functions related to Density Estimation Trees.  The optimal Density Estimation Tree (DET) can be trained on a set of data (specified by `Training`) using cross-validation (with number of folds specified with the `Folds` parameter).  This trained density estimation tree may then be saved with the `OutputModel` output parameter.

The variable importances (that is, the feature importance values for each dimension) may be saved with the `Vi` output parameter, and the density estimates for each training point may be saved with the `TrainingSetEstimates` output parameter.

Enabling path printing for each node outputs the path from the root node to a leaf for each entry in the test set, or training set (if a test set is not provided).  Strings like 'LRLRLR' (indicating that traversal went to the left child, then the right child, then the left child, and so forth) will be output. If 'lr-id' or 'id-lr' are given as the `PathFormat` parameter, then the ID (tag) of every node along the path will be printed after or before the L or R character indicating the direction of traversal, respectively.

This program also can provide density estimates for a set of test points, specified in the `Test` parameter.  The density estimation tree used for this task will be the tree that was trained on the given training points, or a tree given as the parameter `InputModel`.  The density estimates for the test points may be saved using the `TestSetEstimates` output parameter.

### See also

 - [Density estimation on Wikipedia](https://en.wikipedia.org/wiki/Density_estimation)
 - [Density estimation trees (pdf)](https://www.mlpack.org/papers/det.pdf)
 - [DTree class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/det/dtree.hpp)

## Emst()
{: #emst }

#### Fast Euclidean Minimum Spanning Tree
{: #emst_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for Emst().
param := mlpack.EmstOptions()
param.LeafSize = 1
param.Naive = false
param.Verbose = false

output := mlpack.Emst(input, param)
```

An implementation of the Dual-Tree Boruvka algorithm for computing the Euclidean minimum spanning tree of a set of input points. [Detailed documentation](#emst_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `input` | [`*mat.Dense`](#doc_a__mat_Dense) | Input data matrix. | `**--**` |
| `LeafSize` | [`int`](#doc_int) | Leaf size in the kd-tree.  One-element leaves give the empirically best performance, but at the cost of greater memory requirements. | `1` |
| `Naive` | [`bool`](#doc_bool) | Compute the MST using O(n^2) naive algorithm. | `false` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Output` | [`*mat.Dense`](#doc_a__mat_Dense) | Output data.  Stored as an edge list. | 

### Detailed documentation
{: #emst_detailed-documentation }

This program can compute the Euclidean minimum spanning tree of a set of input points using the dual-tree Boruvka algorithm.

The set to calculate the minimum spanning tree of is specified with the `Input` parameter, and the output may be saved with the `Output` output parameter.

The `LeafSize` parameter controls the leaf size of the kd-tree that is used to calculate the minimum spanning tree, and if the `Naive` option is given, then brute-force search is used (this is typically much slower in low dimensions).  The leaf size does not affect the results, but it may have some effect on the runtime of the algorithm.

### Example
For example, the minimum spanning tree of the input dataset `data` can be calculated with a leaf size of 20 and stored as `spanning_tree` using the following command:

```go
// Initialize optional parameters for Emst().
param := mlpack.EmstOptions()
param.LeafSize = 20

spanning_tree := mlpack.Emst(data, param)
```

The output matrix is a three-dimensional matrix, where each row indicates an edge.  The first dimension corresponds to the lesser index of the edge; the second dimension corresponds to the greater index of the edge; and the third column corresponds to the distance between the two points.

### See also

 - [Minimum spanning tree on Wikipedia](https://en.wikipedia.org/wiki/Minimum_spanning_tree)
 - [Fast Euclidean Minimum Spanning Tree: Algorithm, Analysis, and Applications (pdf)](https://www.mlpack.org/papers/emst.pdf)
 - [DualTreeBoruvka class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/emst/dtb.hpp)

## Fastmks()
{: #fastmks }

#### FastMKS (Fast Max-Kernel Search)
{: #fastmks_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for Fastmks().
param := mlpack.FastmksOptions()
param.Bandwidth = 1
param.Base = 2
param.Degree = 2
param.InputModel = nil
param.K = 0
param.Kernel = "linear"
param.Naive = false
param.Offset = 0
param.Query = mat.NewDense(1, 1, nil)
param.Reference = mat.NewDense(1, 1, nil)
param.Scale = 1
param.Single = false
param.Verbose = false

indices, kernels, output_model := mlpack.Fastmks(param)
```

An implementation of the single-tree and dual-tree fast max-kernel search (FastMKS) algorithm.  Given a set of reference points and a set of query points, this can find the reference point with maximum kernel value for each query point; trained models can be reused for future queries. [Detailed documentation](#fastmks_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `Bandwidth` | [`float64`](#doc_float64) | Bandwidth (for Gaussian, Epanechnikov, and triangular kernels). | `1` |
| `Base` | [`float64`](#doc_float64) | Base to use during cover tree construction. | `2` |
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `Degree` | [`float64`](#doc_float64) | Degree of polynomial kernel. | `2` |
| `InputModel` | [`fastmksModel`](#doc_model) | Input FastMKS model to use. | `nil` |
| `K` | [`int`](#doc_int) | Number of maximum kernels to find. | `0` |
| `Kernel` | [`string`](#doc_string) | Kernel type to use: 'linear', 'polynomial', 'cosine', 'gaussian', 'epanechnikov', 'triangular', 'hyptan'. | `"linear"` |
| `Naive` | [`bool`](#doc_bool) | If true, O(n^2) naive mode is used for computation. | `false` |
| `Offset` | [`float64`](#doc_float64) | Offset of kernel (for polynomial and hyptan kernels). | `0` |
| `Query` | [`*mat.Dense`](#doc_a__mat_Dense) | The query dataset. | `mat.NewDense(1, 1, nil)` |
| `Reference` | [`*mat.Dense`](#doc_a__mat_Dense) | The reference dataset. | `mat.NewDense(1, 1, nil)` |
| `Scale` | [`float64`](#doc_float64) | Scale of kernel (for hyptan kernel). | `1` |
| `Single` | [`bool`](#doc_bool) | If true, single-tree search is used (as opposed to dual-tree search. | `false` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Indices` | [`*mat.Dense`](#doc_a__mat_Dense) | Output matrix of indices. | 
| `Kernels` | [`*mat.Dense`](#doc_a__mat_Dense) | Output matrix of kernels. | 
| `OutputModel` | [`fastmksModel`](#doc_model) | Output for FastMKS model. | 

### Detailed documentation
{: #fastmks_detailed-documentation }

This program will find the k maximum kernels of a set of points, using a query set and a reference set (which can optionally be the same set). More specifically, for each point in the query set, the k points in the reference set with maximum kernel evaluations are found.  The kernel function used is specified with the `Kernel` parameter.

### Example
For example, the following command will calculate, for each point in the query set `query`, the five points in the reference set `reference` with maximum kernel evaluation using the linear kernel.  The kernel evaluations may be saved with the  `kernels` output parameter and the indices may be saved with the `indices` output parameter.

```go
// Initialize optional parameters for Fastmks().
param := mlpack.FastmksOptions()
param.K = 5
param.Reference = reference
param.Query = query
param.Kernel = "linear"

indices, kernels, _ := mlpack.Fastmks(param)
```

The output matrices are organized such that row i and column j in the indices matrix corresponds to the index of the point in the reference set that has j'th largest kernel evaluation with the point in the query set with index i.  Row i and column j in the kernels matrix corresponds to the kernel evaluation between those two points.

This program performs FastMKS using a cover tree.  The base used to build the cover tree can be specified with the `Base` parameter.

### See also

 - [k-nearest-neighbor search](#knn)
 - [Dual-tree Fast Exact Max-Kernel Search (pdf)](https://mlpack.org/papers/fmks.pdf)
 - [FastMKS class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/fastmks/fastmks.hpp)

## GmmTrain()
{: #gmm_train }

#### Gaussian Mixture Model (GMM) Training
{: #gmm_train_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for GmmTrain().
param := mlpack.GmmTrainOptions()
param.DiagonalCovariance = false
param.InputModel = nil
param.KmeansMaxIterations = 1000
param.MaxIterations = 250
param.NoForcePositive = false
param.Noise = 0
param.Percentage = 0.02
param.RefinedStart = false
param.Samplings = 100
param.Seed = 0
param.Tolerance = 1e-10
param.Trials = 1
param.Verbose = false

output_model := mlpack.GmmTrain(gaussians, input, param)
```

An implementation of the EM algorithm for training Gaussian mixture models (GMMs).  Given a dataset, this can train a GMM for future use with other tools. [Detailed documentation](#gmm_train_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `DiagonalCovariance` | [`bool`](#doc_bool) | Force the covariance of the Gaussians to be diagonal.  This can accelerate training time significantly. | `false` |
| `gaussians` | [`int`](#doc_int) | Number of Gaussians in the GMM. | `**--**` |
| `input` | [`*mat.Dense`](#doc_a__mat_Dense) | The training data on which the model will be fit. | `**--**` |
| `InputModel` | [`gmm`](#doc_model) | Initial input GMM model to start training with. | `nil` |
| `KmeansMaxIterations` | [`int`](#doc_int) | Maximum number of iterations for the k-means algorithm (used to initialize EM). | `1000` |
| `MaxIterations` | [`int`](#doc_int) | Maximum number of iterations of EM algorithm (passing 0 will run until convergence). | `250` |
| `NoForcePositive` | [`bool`](#doc_bool) | Do not force the covariance matrices to be positive definite. | `false` |
| `Noise` | [`float64`](#doc_float64) | Variance of zero-mean Gaussian noise to add to data. | `0` |
| `Percentage` | [`float64`](#doc_float64) | If using --refined_start, specify the percentage of the dataset used for each sampling (should be between 0.0 and 1.0). | `0.02` |
| `RefinedStart` | [`bool`](#doc_bool) | During the initialization, use refined initial positions for k-means clustering (Bradley and Fayyad, 1998). | `false` |
| `Samplings` | [`int`](#doc_int) | If using --refined_start, specify the number of samplings used for initial points. | `100` |
| `Seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `Tolerance` | [`float64`](#doc_float64) | Tolerance for convergence of EM. | `1e-10` |
| `Trials` | [`int`](#doc_int) | Number of trials to perform in training GMM. | `1` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `OutputModel` | [`gmm`](#doc_model) | Output for trained GMM model. | 

### Detailed documentation
{: #gmm_train_detailed-documentation }

This program takes a parametric estimate of a Gaussian mixture model (GMM) using the EM algorithm to find the maximum likelihood estimate.  The model may be saved and reused by other mlpack GMM tools.

The input data to train on must be specified with the `Input` parameter, and the number of Gaussians in the model must be specified with the `Gaussians` parameter.  Optionally, many trials with different random initializations may be run, and the result with highest log-likelihood on the training data will be taken.  The number of trials to run is specified with the `Trials` parameter.  By default, only one trial is run.

The tolerance for convergence and maximum number of iterations of the EM algorithm are specified with the `Tolerance` and `MaxIterations` parameters, respectively.  The GMM may be initialized for training with another model, specified with the `InputModel` parameter. Otherwise, the model is initialized by running k-means on the data.  The k-means clustering initialization can be controlled with the `KmeansMaxIterations`, `RefinedStart`, `Samplings`, and `Percentage` parameters.  If `RefinedStart` is specified, then the Bradley-Fayyad refined start initialization will be used.  This can often lead to better clustering results.

The 'diagonal_covariance' flag will cause the learned covariances to be diagonal matrices.  This significantly simplifies the model itself and causes training to be faster, but restricts the ability to fit more complex GMMs.

If GMM training fails with an error indicating that a covariance matrix could not be inverted, make sure that the `NoForcePositive` parameter is not specified.  Alternately, adding a small amount of Gaussian noise (using the `Noise` parameter) to the entire dataset may help prevent Gaussians with zero variance in a particular dimension, which is usually the cause of non-invertible covariance matrices.

The `NoForcePositive` parameter, if set, will avoid the checks after each iteration of the EM algorithm which ensure that the covariance matrices are positive definite.  Specifying the flag can cause faster runtime, but may also cause non-positive definite covariance matrices, which will cause the program to crash.

### Example
As an example, to train a 6-Gaussian GMM on the data in `data` with a maximum of 100 iterations of EM and 3 trials, saving the trained GMM to `gmm`, the following command can be used:

```go
// Initialize optional parameters for GmmTrain().
param := mlpack.GmmTrainOptions()
param.Trials = 3

gmm := mlpack.GmmTrain(data, 6, param)
```

To re-train that GMM on another set of data `data2`, the following command may be used: 

```go
// Initialize optional parameters for GmmTrain().
param := mlpack.GmmTrainOptions()
param.InputModel = &gmm

new_gmm := mlpack.GmmTrain(data2, 6, param)
```

### See also

 - [GmmGenerate()](#gmm_generate)
 - [GmmProbability()](#gmm_probability)
 - [Gaussian Mixture Models on Wikipedia](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model)
 - [GMM class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/gmm/gmm.hpp)

## GmmGenerate()
{: #gmm_generate }

#### GMM Sample Generator
{: #gmm_generate_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for GmmGenerate().
param := mlpack.GmmGenerateOptions()
param.Seed = 0
param.Verbose = false

output := mlpack.GmmGenerate(inputModel, samples, param)
```

A sample generator for pre-trained GMMs.  Given a pre-trained GMM, this can sample new points randomly from that distribution. [Detailed documentation](#gmm_generate_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `inputModel` | [`gmm`](#doc_model) | Input GMM model to generate samples from. | `**--**` |
| `samples` | [`int`](#doc_int) | Number of samples to generate. | `**--**` |
| `Seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Output` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to save output samples in. | 

### Detailed documentation
{: #gmm_generate_detailed-documentation }

This program is able to generate samples from a pre-trained GMM (use gmm_train to train a GMM).  The pre-trained GMM must be specified with the `InputModel` parameter.  The number of samples to generate is specified by the `Samples` parameter.  Output samples may be saved with the `Output` output parameter.

### Example
The following command can be used to generate 100 samples from the pre-trained GMM `gmm` and store those generated samples in `samples`:

```go
// Initialize optional parameters for GmmGenerate().
param := mlpack.GmmGenerateOptions()

samples := mlpack.GmmGenerate(&gmm, 100, param)
```

### See also

 - [GmmTrain()](#gmm_train)
 - [GmmProbability()](#gmm_probability)
 - [Gaussian Mixture Models on Wikipedia](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model)
 - [GMM class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/gmm/gmm.hpp)

## GmmProbability()
{: #gmm_probability }

#### GMM Probability Calculator
{: #gmm_probability_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for GmmProbability().
param := mlpack.GmmProbabilityOptions()
param.Verbose = false

output := mlpack.GmmProbability(input, inputModel, param)
```

A probability calculator for GMMs.  Given a pre-trained GMM and a set of points, this can compute the probability that each point is from the given GMM. [Detailed documentation](#gmm_probability_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `input` | [`*mat.Dense`](#doc_a__mat_Dense) | Input matrix to calculate probabilities of. | `**--**` |
| `inputModel` | [`gmm`](#doc_model) | Input GMM to use as model. | `**--**` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Output` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to store calculated probabilities in. | 

### Detailed documentation
{: #gmm_probability_detailed-documentation }

This program calculates the probability that given points came from a given GMM (that is, P(X \| gmm)).  The GMM is specified with the `InputModel` parameter, and the points are specified with the `Input` parameter.  The output probabilities may be saved via the `Output` output parameter.

### Example
So, for example, to calculate the probabilities of each point in `points` coming from the pre-trained GMM `gmm`, while storing those probabilities in `probs`, the following command could be used:

```go
// Initialize optional parameters for GmmProbability().
param := mlpack.GmmProbabilityOptions()

probs := mlpack.GmmProbability(&gmm, points, param)
```

### See also

 - [GmmTrain()](#gmm_train)
 - [GmmGenerate()](#gmm_generate)
 - [Gaussian Mixture Models on Wikipedia](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model)
 - [GMM class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/gmm/gmm.hpp)

## HmmTrain()
{: #hmm_train }

#### Hidden Markov Model (HMM) Training
{: #hmm_train_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for HmmTrain().
param := mlpack.HmmTrainOptions()
param.Batch = false
param.Gaussians = 0
param.InputModel = nil
param.LabelsFile = ""
param.Seed = 0
param.States = 0
param.Tolerance = 1e-05
param.Type = "gaussian"
param.Verbose = false

output_model := mlpack.HmmTrain(inputFile, param)
```

An implementation of training algorithms for Hidden Markov Models (HMMs). Given labeled or unlabeled data, an HMM can be trained for further use with other mlpack HMM tools. [Detailed documentation](#hmm_train_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `Batch` | [`bool`](#doc_bool) | If true, input_file (and if passed, labels_file) are expected to contain a list of files to use as input observation sequences (and label sequences). | `false` |
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `Gaussians` | [`int`](#doc_int) | Number of gaussians in each GMM (necessary when type is 'gmm'). | `0` |
| `inputFile` | [`string`](#doc_string) | File containing input observations. | `**--**` |
| `InputModel` | [`hmmModel`](#doc_model) | Pre-existing HMM model to initialize training with. | `nil` |
| `LabelsFile` | [`string`](#doc_string) | Optional file of hidden states, used for labeled training. | `""` |
| `Seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `States` | [`int`](#doc_int) | Number of hidden states in HMM (necessary, unless model_file is specified). | `0` |
| `Tolerance` | [`float64`](#doc_float64) | Tolerance of the Baum-Welch algorithm. | `1e-05` |
| `Type` | [`string`](#doc_string) | Type of HMM: discrete \| gaussian \| diag_gmm \| gmm. | `"gaussian"` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `OutputModel` | [`hmmModel`](#doc_model) | Output for trained HMM. | 

### Detailed documentation
{: #hmm_train_detailed-documentation }

This program allows a Hidden Markov Model to be trained on labeled or unlabeled data.  It supports four types of HMMs: Discrete HMMs, Gaussian HMMs, GMM HMMs, or Diagonal GMM HMMs

Either one input sequence can be specified (with `InputFile`), or, a file containing files in which input sequences can be found (when `InputFile`and`Batch` are used together).  In addition, labels can be provided in the file specified by `LabelsFile`, and if `Batch` is used, the file given to `LabelsFile` should contain a list of files of labels corresponding to the sequences in the file given to `InputFile`.

The HMM is trained with the Baum-Welch algorithm if no labels are provided.  The tolerance of the Baum-Welch algorithm can be set with the `Tolerance`option.  By default, the transition matrix is randomly initialized and the emission distributions are initialized to fit the extent of the data.

Optionally, a pre-created HMM model can be used as a guess for the transition matrix and emission probabilities; this is specifiable with `OutputModel`.

### See also

 - [HmmGenerate()](#hmm_generate)
 - [HmmLoglik()](#hmm_loglik)
 - [HmmViterbi()](#hmm_viterbi)
 - [Hidden Mixture Models on Wikipedia](https://en.wikipedia.org/wiki/Hidden_Markov_model)
 - [HMM class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/hmm/hmm.hpp)

## HmmGenerate()
{: #hmm_generate }

#### Hidden Markov Model (HMM) Sequence Generator
{: #hmm_generate_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for HmmGenerate().
param := mlpack.HmmGenerateOptions()
param.Seed = 0
param.StartState = 0
param.Verbose = false

output, state := mlpack.HmmGenerate(length, model, param)
```

A utility to generate random sequences from a pre-trained Hidden Markov Model (HMM).  The length of the desired sequence can be specified, and a random sequence of observations is returned. [Detailed documentation](#hmm_generate_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `length` | [`int`](#doc_int) | Length of sequence to generate. | `**--**` |
| `model` | [`hmmModel`](#doc_model) | Trained HMM to generate sequences with. | `**--**` |
| `Seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `StartState` | [`int`](#doc_int) | Starting state of sequence. | `0` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Output` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to save observation sequence to. | 
| `State` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to save hidden state sequence to. | 

### Detailed documentation
{: #hmm_generate_detailed-documentation }

This utility takes an already-trained HMM, specified as the `Model` parameter, and generates a random observation sequence and hidden state sequence based on its parameters. The observation sequence may be saved with the `Output` output parameter, and the internal state  sequence may be saved with the `State` output parameter.

The state to start the sequence in may be specified with the `StartState` parameter.

### Example
For example, to generate a sequence of length 150 from the HMM `hmm` and save the observation sequence to `observations` and the hidden state sequence to `states`, the following command may be used: 

```go
// Initialize optional parameters for HmmGenerate().
param := mlpack.HmmGenerateOptions()

observations, states := mlpack.HmmGenerate(&hmm, 150, param)
```

### See also

 - [HmmTrain()](#hmm_train)
 - [HmmLoglik()](#hmm_loglik)
 - [HmmViterbi()](#hmm_viterbi)
 - [Hidden Mixture Models on Wikipedia](https://en.wikipedia.org/wiki/Hidden_Markov_model)
 - [HMM class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/hmm/hmm.hpp)

## HmmLoglik()
{: #hmm_loglik }

#### Hidden Markov Model (HMM) Sequence Log-Likelihood
{: #hmm_loglik_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for HmmLoglik().
param := mlpack.HmmLoglikOptions()
param.Verbose = false

log_likelihood := mlpack.HmmLoglik(input, inputModel, param)
```

A utility for computing the log-likelihood of a sequence for Hidden Markov Models (HMMs).  Given a pre-trained HMM and an observation sequence, this computes and returns the log-likelihood of that sequence being observed from that HMM. [Detailed documentation](#hmm_loglik_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `input` | [`*mat.Dense`](#doc_a__mat_Dense) | File containing observations, | `**--**` |
| `inputModel` | [`hmmModel`](#doc_model) | File containing HMM. | `**--**` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `LogLikelihood` | [`float64`](#doc_float64) | Log-likelihood of the sequence. | 

### Detailed documentation
{: #hmm_loglik_detailed-documentation }

This utility takes an already-trained HMM, specified with the `InputModel` parameter, and evaluates the log-likelihood of a sequence of observations, given with the `Input` parameter.  The computed log-likelihood is given as output.

### Example
For example, to compute the log-likelihood of the sequence `seq` with the pre-trained HMM `hmm`, the following command may be used: 

```go
// Initialize optional parameters for HmmLoglik().
param := mlpack.HmmLoglikOptions()

_ := mlpack.HmmLoglik(seq, &hmm, param)
```

### See also

 - [HmmTrain()](#hmm_train)
 - [HmmGenerate()](#hmm_generate)
 - [HmmViterbi()](#hmm_viterbi)
 - [Hidden Mixture Models on Wikipedia](https://en.wikipedia.org/wiki/Hidden_Markov_model)
 - [HMM class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/hmm/hmm.hpp)

## HmmViterbi()
{: #hmm_viterbi }

#### Hidden Markov Model (HMM) Viterbi State Prediction
{: #hmm_viterbi_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for HmmViterbi().
param := mlpack.HmmViterbiOptions()
param.Verbose = false

output := mlpack.HmmViterbi(input, inputModel, param)
```

A utility for computing the most probable hidden state sequence for Hidden Markov Models (HMMs).  Given a pre-trained HMM and an observed sequence, this uses the Viterbi algorithm to compute and return the most probable hidden state sequence. [Detailed documentation](#hmm_viterbi_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `input` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix containing observations, | `**--**` |
| `inputModel` | [`hmmModel`](#doc_model) | Trained HMM to use. | `**--**` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Output` | [`*mat.Dense`](#doc_a__mat_Dense) | File to save predicted state sequence to. | 

### Detailed documentation
{: #hmm_viterbi_detailed-documentation }

This utility takes an already-trained HMM, specified as `InputModel`, and evaluates the most probable hidden state sequence of a given sequence of observations (specified as '`Input`, using the Viterbi algorithm.  The computed state sequence may be saved using the `Output` output parameter.

### Example
For example, to predict the state sequence of the observations `obs` using the HMM `hmm`, storing the predicted state sequence to `states`, the following command could be used:

```go
// Initialize optional parameters for HmmViterbi().
param := mlpack.HmmViterbiOptions()

states := mlpack.HmmViterbi(obs, &hmm, param)
```

### See also

 - [HmmTrain()](#hmm_train)
 - [HmmGenerate()](#hmm_generate)
 - [HmmLoglik()](#hmm_loglik)
 - [Hidden Mixture Models on Wikipedia](https://en.wikipedia.org/wiki/Hidden_Markov_model)
 - [HMM class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/hmm/hmm.hpp)

## HoeffdingTree()
{: #hoeffding_tree }

#### Hoeffding trees
{: #hoeffding_tree_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for HoeffdingTree().
param := mlpack.HoeffdingTreeOptions()
param.BatchMode = false
param.Bins = 10
param.Confidence = 0.95
param.InfoGain = false
param.InputModel = nil
param.Labels = mat.NewDense(1, 1, nil)
param.MaxSamples = 5000
param.MinSamples = 100
param.NumericSplitStrategy = "binary"
param.ObservationsBeforeBinning = 100
param.Passes = 1
param.Test = mat.NewDense(1, 1, nil)
param.TestLabels = mat.NewDense(1, 1, nil)
param.Training = mat.NewDense(1, 1, nil)
param.Verbose = false

output_model, predictions, probabilities := mlpack.HoeffdingTree(param)
```

An implementation of Hoeffding trees, a form of streaming decision tree for classification.  Given labeled data, a Hoeffding tree can be trained and saved for later use, or a pre-trained Hoeffding tree can be used for predicting the classifications of new points. [Detailed documentation](#hoeffding_tree_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `BatchMode` | [`bool`](#doc_bool) | If true, samples will be considered in batch instead of as a stream.  This generally results in better trees but at the cost of memory usage and runtime. | `false` |
| `Bins` | [`int`](#doc_int) | If the 'domingos' split strategy is used, this specifies the number of bins for each numeric split. | `10` |
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `Confidence` | [`float64`](#doc_float64) | Confidence before splitting (between 0 and 1). | `0.95` |
| `InfoGain` | [`bool`](#doc_bool) | If set, information gain is used instead of Gini impurity for calculating Hoeffding bounds. | `false` |
| `InputModel` | [`hoeffdingTreeModel`](#doc_model) | Input trained Hoeffding tree model. | `nil` |
| `Labels` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | Labels for training dataset. | `mat.NewDense(1, 1, nil)` |
| `MaxSamples` | [`int`](#doc_int) | Maximum number of samples before splitting. | `5000` |
| `MinSamples` | [`int`](#doc_int) | Minimum number of samples before splitting. | `100` |
| `NumericSplitStrategy` | [`string`](#doc_string) | The splitting strategy to use for numeric features: 'domingos' or 'binary'. | `"binary"` |
| `ObservationsBeforeBinning` | [`int`](#doc_int) | If the 'domingos' split strategy is used, this specifies the number of samples observed before binning is performed. | `100` |
| `Passes` | [`int`](#doc_int) | Number of passes to take over the dataset. | `1` |
| `Test` | [`matrixWithInfo`](#doc_matrixWithInfo) | Testing dataset (may be categorical). | `mat.NewDense(1, 1, nil)` |
| `TestLabels` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | Labels of test data. | `mat.NewDense(1, 1, nil)` |
| `Training` | [`matrixWithInfo`](#doc_matrixWithInfo) | Training dataset (may be categorical). | `mat.NewDense(1, 1, nil)` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `OutputModel` | [`hoeffdingTreeModel`](#doc_model) | Output for trained Hoeffding tree model. | 
| `Predictions` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | Matrix to output label predictions for test data into. | 
| `Probabilities` | [`*mat.Dense`](#doc_a__mat_Dense) | In addition to predicting labels, provide rediction probabilities in this matrix. | 

### Detailed documentation
{: #hoeffding_tree_detailed-documentation }

This program implements Hoeffding trees, a form of streaming decision tree suited best for large (or streaming) datasets.  This program supports both categorical and numeric data.  Given an input dataset, this program is able to train the tree with numerous training options, and save the model to a file.  The program is also able to use a trained model or a model from file in order to predict classes for a given test set.

The training file and associated labels are specified with the `Training` and `Labels` parameters, respectively. Optionally, if `Labels` is not specified, the labels are assumed to be the last dimension of the training dataset.

The training may be performed in batch mode (like a typical decision tree algorithm) by specifying the `BatchMode` option, but this may not be the best option for large datasets.

When a model is trained, it may be saved via the `OutputModel` output parameter.  A model may be loaded from file for further training or testing with the `InputModel` parameter.

Test data may be specified with the `Test` parameter, and if performance statistics are desired for that test set, labels may be specified with the `TestLabels` parameter.  Predictions for each test point may be saved with the `Predictions` output parameter, and class probabilities for each prediction may be saved with the `Probabilities` output parameter.

### Example
For example, to train a Hoeffding tree with confidence 0.99 with data `dataset`, saving the trained tree to `tree`, the following command may be used:

```go
// Initialize optional parameters for HoeffdingTree().
param := mlpack.HoeffdingTreeOptions()
param.Training = dataset
param.Confidence = 0.99

tree, _, _ := mlpack.HoeffdingTree(param)
```

Then, this tree may be used to make predictions on the test set `test_set`, saving the predictions into `predictions` and the class probabilities into `class_probs` with the following command: 

```go
// Initialize optional parameters for HoeffdingTree().
param := mlpack.HoeffdingTreeOptions()
param.InputModel = &tree
param.Test = test_set

_, predictions, class_probs := mlpack.HoeffdingTree(param)
```

### See also

 - [DecisionTree()](#decision_tree)
 - [RandomForest()](#random_forest)
 - [Mining High-Speed Data Streams (pdf)](http://dm.cs.washington.edu/papers/vfdt-kdd00.pdf)
 - [HoeffdingTree class documentation](../../user/methods/hoeffding_tree.md)

## Kde()
{: #kde }

#### Kernel Density Estimation
{: #kde_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for Kde().
param := mlpack.KdeOptions()
param.AbsError = 0
param.Algorithm = "dual-tree"
param.Bandwidth = 1
param.InitialSampleSize = 100
param.InputModel = nil
param.Kernel = "gaussian"
param.McBreakCoef = 0.4
param.McEntryCoef = 3
param.McProbability = 0.95
param.MonteCarlo = false
param.Query = mat.NewDense(1, 1, nil)
param.Reference = mat.NewDense(1, 1, nil)
param.RelError = 0.05
param.Tree = "kd-tree"
param.Verbose = false

output_model, predictions := mlpack.Kde(param)
```

An implementation of kernel density estimation with dual-tree algorithms. Given a set of reference points and query points and a kernel function, this can estimate the density function at the location of each query point using trees; trees that are built can be saved for later use. [Detailed documentation](#kde_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `AbsError` | [`float64`](#doc_float64) | Relative error tolerance for the prediction. | `0` |
| `Algorithm` | [`string`](#doc_string) | Algorithm to use for the prediction.('dual-tree', 'single-tree'). | `"dual-tree"` |
| `Bandwidth` | [`float64`](#doc_float64) | Bandwidth of the kernel. | `1` |
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `InitialSampleSize` | [`int`](#doc_int) | Initial sample size for Monte Carlo estimations. | `100` |
| `InputModel` | [`kdeModel`](#doc_model) | Contains pre-trained KDE model. | `nil` |
| `Kernel` | [`string`](#doc_string) | Kernel to use for the prediction.('gaussian', 'epanechnikov', 'laplacian', 'spherical', 'triangular'). | `"gaussian"` |
| `McBreakCoef` | [`float64`](#doc_float64) | Controls what fraction of the amount of node's descendants is the limit for the sample size before it recurses. | `0.4` |
| `McEntryCoef` | [`float64`](#doc_float64) | Controls how much larger does the amount of node descendants has to be compared to the initial sample size in order to be a candidate for Monte Carlo estimations. | `3` |
| `McProbability` | [`float64`](#doc_float64) | Probability of the estimation being bounded by relative error when using Monte Carlo estimations. | `0.95` |
| `MonteCarlo` | [`bool`](#doc_bool) | Whether to use Monte Carlo estimations when possible. | `false` |
| `Query` | [`*mat.Dense`](#doc_a__mat_Dense) | Query dataset to KDE on. | `mat.NewDense(1, 1, nil)` |
| `Reference` | [`*mat.Dense`](#doc_a__mat_Dense) | Input reference dataset use for KDE. | `mat.NewDense(1, 1, nil)` |
| `RelError` | [`float64`](#doc_float64) | Relative error tolerance for the prediction. | `0.05` |
| `Tree` | [`string`](#doc_string) | Tree to use for the prediction.('kd-tree', 'ball-tree', 'cover-tree', 'octree', 'r-tree'). | `"kd-tree"` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `OutputModel` | [`kdeModel`](#doc_model) | If specified, the KDE model will be saved here. | 
| `Predictions` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | Vector to store density predictions. | 

### Detailed documentation
{: #kde_detailed-documentation }

This program performs a Kernel Density Estimation. KDE is a non-parametric way of estimating probability density function. For each query point the program will estimate its probability density by applying a kernel function to each reference point. The computational complexity of this is O(N^2) where there are N query points and N reference points, but this implementation will typically see better performance as it uses an approximate dual or single tree algorithm for acceleration.

Dual or single tree optimization avoids many barely relevant calculations (as kernel function values decrease with distance), so it is an approximate computation. You can specify the maximum relative error tolerance for each query value with `RelError` as well as the maximum absolute error tolerance with the parameter `AbsError`. This program runs using an Euclidean metric. Kernel function can be selected using the `Kernel` option. You can also choose what which type of tree to use for the dual-tree algorithm with `Tree`. It is also possible to select whether to use dual-tree algorithm or single-tree algorithm using the `Algorithm` option.

Monte Carlo estimations can be used to accelerate the KDE estimate when the Gaussian Kernel is used. This provides a probabilistic guarantee on the the error of the resulting KDE instead of an absolute guarantee.To enable Monte Carlo estimations, the `MonteCarlo` flag can be used, and success probability can be set with the `McProbability` option. It is possible to set the initial sample size for the Monte Carlo estimation using `InitialSampleSize`. This implementation will only consider a node, as a candidate for the Monte Carlo estimation, if its number of descendant nodes is bigger than the initial sample size. This can be controlled using a coefficient that will multiply the initial sample size and can be set using `McEntryCoef`. To avoid using the same amount of computations an exact approach would take, this program recurses the tree whenever a fraction of the amount of the node's descendant points have already been computed. This fraction is set using `McBreakCoef`.

### Example
For example, the following will run KDE using the data in `ref_data` for training and the data in `qu_data` as query data. It will apply an Epanechnikov kernel with a 0.2 bandwidth to each reference point and use a KD-Tree for the dual-tree optimization. The returned predictions will be within 5% of the real KDE value for each query point.

```go
// Initialize optional parameters for Kde().
param := mlpack.KdeOptions()
param.Reference = ref_data
param.Query = qu_data
param.Bandwidth = 0.2
param.Kernel = "epanechnikov"
param.Tree = "kd-tree"
param.RelError = 0.05

_, out_data := mlpack.Kde(param)
```

the predicted density estimations will be stored in `out_data`.
If no `Query` is provided, then KDE will be computed on the `Reference` dataset.
It is possible to select either a reference dataset or an input model but not both at the same time. If an input model is selected and parameter values are not set (e.g. `Bandwidth`) then default parameter values will be used.

In addition to the last program call, it is also possible to activate Monte Carlo estimations if a Gaussian kernel is used. This can provide faster results, but the KDE will only have a probabilistic guarantee of meeting the desired error bound (instead of an absolute guarantee). The following example will run KDE using a Monte Carlo estimation when possible. The results will be within a 5% of the real KDE value with a 95% probability. Initial sample size for the Monte Carlo estimation will be 200 points and a node will be a candidate for the estimation only when it contains 700 (i.e. 3.5*200) points. If a node contains 700 points and 420 (i.e. 0.6*700) have already been sampled, then the algorithm will recurse instead of keep sampling.

```go
// Initialize optional parameters for Kde().
param := mlpack.KdeOptions()
param.Reference = ref_data
param.Query = qu_data
param.Bandwidth = 0.2
param.Kernel = "gaussian"
param.Tree = "kd-tree"
param.RelError = 0.05
param.MonteCarlo = 
param.McProbability = 0.95
param.InitialSampleSize = 200
param.McEntryCoef = 3.5
param.McBreakCoef = 0.6

_, out_data := mlpack.Kde(param)
```

### See also

 - [Knn()](#knn)
 - [Kernel density estimation on Wikipedia](https://en.wikipedia.org/wiki/Kernel_density_estimation)
 - [Tree-Independent Dual-Tree Algorithms](https://arxiv.org/pdf/1304.4327)
 - [Fast High-dimensional Kernel Summations Using the Monte Carlo Multipole Method](https://proceedings.neurips.cc/paper_files/paper/2008/file/39059724f73a9969845dfe4146c5660e-Paper.pdf)
 - [KDE C++ class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/kde/kde.hpp)

## KernelPca()
{: #kernel_pca }

#### Kernel Principal Components Analysis
{: #kernel_pca_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for KernelPca().
param := mlpack.KernelPcaOptions()
param.Bandwidth = 1
param.Center = false
param.Degree = 1
param.KernelScale = 1
param.NewDimensionality = 0
param.NystroemMethod = false
param.Offset = 0
param.Sampling = "kmeans"
param.Verbose = false

output := mlpack.KernelPca(input, kernel, param)
```

An implementation of Kernel Principal Components Analysis (KPCA).  This can be used to perform nonlinear dimensionality reduction or preprocessing on a given dataset. [Detailed documentation](#kernel_pca_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `Bandwidth` | [`float64`](#doc_float64) | Bandwidth, for 'gaussian' and 'laplacian' kernels. | `1` |
| `Center` | [`bool`](#doc_bool) | If set, the transformed data will be centered about the origin. | `false` |
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `Degree` | [`float64`](#doc_float64) | Degree of polynomial, for 'polynomial' kernel. | `1` |
| `input` | [`*mat.Dense`](#doc_a__mat_Dense) | Input dataset to perform KPCA on. | `**--**` |
| `kernel` | [`string`](#doc_string) | The kernel to use; see the above documentation for the list of usable kernels. | `**--**` |
| `KernelScale` | [`float64`](#doc_float64) | Scale, for 'hyptan' kernel. | `1` |
| `NewDimensionality` | [`int`](#doc_int) | If not 0, reduce the dimensionality of the output dataset by ignoring the dimensions with the smallest eigenvalues. | `0` |
| `NystroemMethod` | [`bool`](#doc_bool) | If set, the Nystroem method will be used. | `false` |
| `Offset` | [`float64`](#doc_float64) | Offset, for 'hyptan' and 'polynomial' kernels. | `0` |
| `Sampling` | [`string`](#doc_string) | Sampling scheme to use for the Nystroem method: 'kmeans', 'random', 'ordered' | `"kmeans"` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Output` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to save modified dataset to. | 

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

The parameters for each of the kernels should be specified with the options `Bandwidth`, `KernelScale`, `Offset`, or `Degree` (or a combination of those parameters).

Optionally, the Nystroem method ("Using the Nystroem method to speed up kernel machines", 2001) can be used to calculate the kernel matrix by specifying the `NystroemMethod` parameter. This approach works by using a subset of the data as basis to reconstruct the kernel matrix; to specify the sampling scheme, the `Sampling` parameter is used.  The sampling scheme for the Nystroem method can be chosen from the following list: 'kmeans', 'random', 'ordered'.

### Example
For example, the following command will perform KPCA on the dataset `input` using the Gaussian kernel, and saving the transformed data to `transformed`: 

```go
// Initialize optional parameters for KernelPca().
param := mlpack.KernelPcaOptions()

transformed := mlpack.KernelPca(input, "gaussian", param)
```

### See also

 - [Kernel principal component analysis on Wikipedia](https://en.wikipedia.org/wiki/Kernel_principal_component_analysis)
 - [Nonlinear Component Analysis as a Kernel Eigenvalue Problem](https://www.mlpack.org/papers/kpca.pdf)
 - [KernelPCA class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/kernel_pca/kernel_pca.hpp)

## Kmeans()
{: #kmeans }

#### K-Means Clustering
{: #kmeans_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for Kmeans().
param := mlpack.KmeansOptions()
param.Algorithm = "naive"
param.AllowEmptyClusters = false
param.InPlace = false
param.InitialCentroids = mat.NewDense(1, 1, nil)
param.KillEmptyClusters = false
param.KmeansPlusPlus = false
param.LabelsOnly = false
param.MaxIterations = 1000
param.Percentage = 0.02
param.RefinedStart = false
param.Samplings = 100
param.Seed = 0
param.Verbose = false

centroid, output := mlpack.Kmeans(clusters, input, param)
```

An implementation of several strategies for efficient k-means clustering. Given a dataset and a value of k, this computes and returns a k-means clustering on that data. [Detailed documentation](#kmeans_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `Algorithm` | [`string`](#doc_string) | Algorithm to use for the Lloyd iteration ('naive', 'pelleg-moore', 'elkan', 'hamerly', 'dualtree', or 'dualtree-covertree'). | `"naive"` |
| `AllowEmptyClusters` | [`bool`](#doc_bool) | Allow empty clusters to be persist. | `false` |
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `clusters` | [`int`](#doc_int) | Number of clusters to find (0 autodetects from initial centroids). | `**--**` |
| `InPlace` | [`bool`](#doc_bool) | If specified, a column containing the learned cluster assignments will be added to the input dataset file.  In this case, --output_file is overridden. (Do not use in Python.) | `false` |
| `InitialCentroids` | [`*mat.Dense`](#doc_a__mat_Dense) | Start with the specified initial centroids. | `mat.NewDense(1, 1, nil)` |
| `input` | [`*mat.Dense`](#doc_a__mat_Dense) | Input dataset to perform clustering on. | `**--**` |
| `KillEmptyClusters` | [`bool`](#doc_bool) | Remove empty clusters when they occur. | `false` |
| `KmeansPlusPlus` | [`bool`](#doc_bool) | Use the k-means++ initialization strategy to choose initial points. | `false` |
| `LabelsOnly` | [`bool`](#doc_bool) | Only output labels into output file. | `false` |
| `MaxIterations` | [`int`](#doc_int) | Maximum number of iterations before k-means terminates. | `1000` |
| `Percentage` | [`float64`](#doc_float64) | Percentage of dataset to use for each refined start sampling (use when --refined_start is specified). | `0.02` |
| `RefinedStart` | [`bool`](#doc_bool) | Use the refined initial point strategy by Bradley and Fayyad to choose initial points. | `false` |
| `Samplings` | [`int`](#doc_int) | Number of samplings to perform for refined start (use when --refined_start is specified). | `100` |
| `Seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Centroid` | [`*mat.Dense`](#doc_a__mat_Dense) | If specified, the centroids of each cluster will  be written to the given file. | 
| `Output` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to store output labels or labeled data to. | 

### Detailed documentation
{: #kmeans_detailed-documentation }

This program performs K-Means clustering on the given dataset.  It can return the learned cluster assignments, and the centroids of the clusters.  Empty clusters are not allowed by default; when a cluster becomes empty, the point furthest from the centroid of the cluster with maximum variance is taken to fill that cluster.

Optionally, the strategy to choose initial centroids can be specified.  The k-means++ algorithm can be used to choose initial centroids with the `KmeansPlusPlus` parameter.  The Bradley and Fayyad approach ("Refining initial points for k-means clustering", 1998) can be used to select initial points by specifying the `RefinedStart` parameter.  This approach works by taking random samplings of the dataset; to specify the number of samplings, the `Samplings` parameter is used, and to specify the percentage of the dataset to be used in each sample, the `Percentage` parameter is used (it should be a value between 0.0 and 1.0).

There are several options available for the algorithm used for each Lloyd iteration, specified with the `Algorithm`  option.  The standard O(kN) approach can be used ('naive').  Other options include the Pelleg-Moore tree-based algorithm ('pelleg-moore'), Elkan's triangle-inequality based algorithm ('elkan'), Hamerly's modification to Elkan's algorithm ('hamerly'), the dual-tree k-means algorithm ('dualtree'), and the dual-tree k-means algorithm using the cover tree ('dualtree-covertree').

The behavior for when an empty cluster is encountered can be modified with the `AllowEmptyClusters` option.  When this option is specified and there is a cluster owning no points at the end of an iteration, that cluster's centroid will simply remain in its position from the previous iteration. If the `KillEmptyClusters` option is specified, then when a cluster owns no points at the end of an iteration, the cluster centroid is simply filled with DBL_MAX, killing it and effectively reducing k for the rest of the computation.  Note that the default option when neither empty cluster option is specified can be time-consuming to calculate; therefore, specifying either of these parameters will often accelerate runtime.

Initial clustering assignments may be specified using the `InitialCentroids` parameter, and the maximum number of iterations may be specified with the `MaxIterations` parameter.

### Example
As an example, to use Hamerly's algorithm to perform k-means clustering with k=10 on the dataset `data`, saving the centroids to `centroids` and the assignments for each point to `assignments`, the following command could be used:

```go
// Initialize optional parameters for Kmeans().
param := mlpack.KmeansOptions()

centroids, assignments := mlpack.Kmeans(data, 10, param)
```

To run k-means on that same dataset with initial centroids specified in `initial` with a maximum of 500 iterations, storing the output centroids in `final` the following command may be used:

```go
// Initialize optional parameters for Kmeans().
param := mlpack.KmeansOptions()
param.InitialCentroids = initial
param.MaxIterations = 500

final, _ := mlpack.Kmeans(data, 10, param)
```

### See also

 - [Dbscan()](#dbscan)
 - [k-means++](https://en.wikipedia.org/wiki/K-means%2B%2B)
 - [Using the triangle inequality to accelerate k-means (pdf)](https://cdn.aaai.org/ICML/2003/ICML03-022.pdf)
 - [Making k-means even faster (pdf)](https://www.cse.iitd.ac.in/~rjaiswal/2015/col870/Project/Faster-k-means/Hamerly.pdf)
 - [Accelerating exact k-means algorithms with geometric reasoning (pdf)](http://reports-archive.adm.cs.cmu.edu/anon/anon/usr/ftp/usr0/ftp/2000/CMU-CS-00-105.pdf)
 - [A dual-tree algorithm for fast k-means clustering with large k (pdf)](http://www.ratml.org/pub/pdf/2017dual.pdf)
 - [KMeans class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/kmeans/kmeans.hpp)

## Lars()
{: #lars }

#### LARS
{: #lars_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for Lars().
param := mlpack.LarsOptions()
param.Input = mat.NewDense(1, 1, nil)
param.InputModel = nil
param.Lambda1 = 0
param.Lambda2 = 0
param.NoIntercept = false
param.NoNormalize = false
param.Responses = mat.NewDense(1, 1, nil)
param.Test = mat.NewDense(1, 1, nil)
param.UseCholesky = false
param.Verbose = false

output_model, output_predictions := mlpack.Lars(param)
```

An implementation of Least Angle Regression (Stagewise/laSso), also known as LARS.  This can train a LARS/LASSO/Elastic Net model and use that model or a pre-trained model to output regression predictions for a test set. [Detailed documentation](#lars_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `Input` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix of covariates (X). | `mat.NewDense(1, 1, nil)` |
| `InputModel` | [`lars`](#doc_model) | Trained LARS model to use. | `nil` |
| `Lambda1` | [`float64`](#doc_float64) | Regularization parameter for l1-norm penalty. | `0` |
| `Lambda2` | [`float64`](#doc_float64) | Regularization parameter for l2-norm penalty. | `0` |
| `NoIntercept` | [`bool`](#doc_bool) | Do not fit an intercept in the model. | `false` |
| `NoNormalize` | [`bool`](#doc_bool) | Do not normalize data to unit variance before modeling. | `false` |
| `Responses` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix of responses/observations (y). | `mat.NewDense(1, 1, nil)` |
| `Test` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix containing points to regress on (test points). | `mat.NewDense(1, 1, nil)` |
| `UseCholesky` | [`bool`](#doc_bool) | Use Cholesky decomposition during computation rather than explicitly computing the full Gram matrix. | `false` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `OutputModel` | [`lars`](#doc_model) | Output LARS model. | 
| `OutputPredictions` | [`*mat.Dense`](#doc_a__mat_Dense) | If --test_file is specified, this file is where the predicted responses will be saved. | 

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

For efficiency reasons, it is not recommended to use this algorithm with `Lambda1` = 0.  In that case, use the 'linear_regression' program, which implements both unregularized linear regression and ridge regression.

To train a LARS/LASSO/Elastic Net model, the `Input` and `Responses` parameters must be given.  The `Lambda1`, `Lambda2`, and `UseCholesky` parameters control the training options.  A trained model can be saved with the `OutputModel`.  If no training is desired at all, a model can be passed via the `InputModel` parameter.

The program can also provide predictions for test data using either the trained model or the given input model.  Test points can be specified with the `Test` parameter.  Predicted responses to the test points can be saved with the `OutputPredictions` output parameter.

### Example
For example, the following command trains a model on the data `data` and responses `responses` with lambda1 set to 0.4 and lambda2 set to 0 (so, LASSO is being solved), and then the model is saved to `lasso_model`:

```go
// Initialize optional parameters for Lars().
param := mlpack.LarsOptions()
param.Input = data
param.Responses = responses
param.Lambda1 = 0.4
param.Lambda2 = 0

lasso_model, _ := mlpack.Lars(param)
```

The following command uses the `lasso_model` to provide predicted responses for the data `test` and save those responses to `test_predictions`: 

```go
// Initialize optional parameters for Lars().
param := mlpack.LarsOptions()
param.InputModel = &lasso_model
param.Test = test

_, test_predictions := mlpack.Lars(param)
```

### See also

 - [LinearRegression()](#linear_regression)
 - [Least angle regression (pdf)](https://mlpack.org/papers/lars.pdf)
 - [LARS C++ class documentation](../../user/methods/lars.md)

## LinearSvm()
{: #linear_svm }

#### Linear SVM is an L2-regularized support vector machine.
{: #linear_svm_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for LinearSvm().
param := mlpack.LinearSvmOptions()
param.Delta = 1
param.Epochs = 50
param.InputModel = nil
param.Labels = mat.NewDense(1, 1, nil)
param.Lambda = 0.0001
param.MaxIterations = 10000
param.NoIntercept = false
param.NumClasses = 0
param.Optimizer = "lbfgs"
param.Seed = 0
param.Shuffle = false
param.StepSize = 0.01
param.Test = mat.NewDense(1, 1, nil)
param.TestLabels = mat.NewDense(1, 1, nil)
param.Tolerance = 1e-10
param.Training = mat.NewDense(1, 1, nil)
param.Verbose = false

output_model, predictions, probabilities := mlpack.LinearSvm(param)
```

An implementation of linear SVM for multiclass classification. Given labeled data, a model can be trained and saved for future use; or, a pre-trained model can be used to classify new points. [Detailed documentation](#linear_svm_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `Delta` | [`float64`](#doc_float64) | Margin of difference between correct class and other classes. | `1` |
| `Epochs` | [`int`](#doc_int) | Maximum number of full epochs over dataset for psgd | `50` |
| `InputModel` | [`linearsvmModel`](#doc_model) | Existing model (parameters). | `nil` |
| `Labels` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | A matrix containing labels (0 or 1) for the points in the training set (y). | `mat.NewDense(1, 1, nil)` |
| `Lambda` | [`float64`](#doc_float64) | L2-regularization parameter for training. | `0.0001` |
| `MaxIterations` | [`int`](#doc_int) | Maximum iterations for optimizer (0 indicates no limit). | `10000` |
| `NoIntercept` | [`bool`](#doc_bool) | Do not add the intercept term to the model. | `false` |
| `NumClasses` | [`int`](#doc_int) | Number of classes for classification; if unspecified (or 0), the number of classes found in the labels will be used. | `0` |
| `Optimizer` | [`string`](#doc_string) | Optimizer to use for training ('lbfgs' or 'psgd'). | `"lbfgs"` |
| `Seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `Shuffle` | [`bool`](#doc_bool) | Don't shuffle the order in which data points are visited for parallel SGD. | `false` |
| `StepSize` | [`float64`](#doc_float64) | Step size for parallel SGD optimizer. | `0.01` |
| `Test` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix containing test dataset. | `mat.NewDense(1, 1, nil)` |
| `TestLabels` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | Matrix containing test labels. | `mat.NewDense(1, 1, nil)` |
| `Tolerance` | [`float64`](#doc_float64) | Convergence tolerance for optimizer. | `1e-10` |
| `Training` | [`*mat.Dense`](#doc_a__mat_Dense) | A matrix containing the training set (the matrix of predictors, X). | `mat.NewDense(1, 1, nil)` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `OutputModel` | [`linearsvmModel`](#doc_model) | Output for trained linear svm model. | 
| `Predictions` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | If test data is specified, this matrix is where the predictions for the test set will be saved. | 
| `Probabilities` | [`*mat.Dense`](#doc_a__mat_Dense) | If test data is specified, this matrix is where the class probabilities for the test set will be saved. | 

### Detailed documentation
{: #linear_svm_detailed-documentation }

An implementation of linear SVMs that uses either L-BFGS or parallel SGD (stochastic gradient descent) to train the model.

This program allows loading a linear SVM model (via the `InputModel` parameter) or training a linear SVM model given training data (specified with the `Training` parameter), or both those things at once.  In addition, this program allows classification on a test dataset (specified with the `Test` parameter) and the classification results may be saved with the `Predictions` output parameter. The trained linear SVM model may be saved using the `OutputModel` output parameter.

The training data, if specified, may have class labels as its last dimension.  Alternately, the `Labels` parameter may be used to specify a separate vector of labels.

When a model is being trained, there are many options.  L2 regularization (to prevent overfitting) can be specified with the `Lambda` option, and the number of classes can be manually specified with the `NumClasses`and if an intercept term is not desired in the model, the `NoIntercept` parameter can be specified.Margin of difference between correct class and other classes can be specified with the `Delta` option.The optimizer used to train the model can be specified with the `Optimizer` parameter.  Available options are 'psgd' (parallel stochastic gradient descent) and 'lbfgs' (the L-BFGS optimizer).  There are also various parameters for the optimizer; the `MaxIterations` parameter specifies the maximum number of allowed iterations, and the `Tolerance` parameter specifies the tolerance for convergence.  For the parallel SGD optimizer, the `StepSize` parameter controls the step size taken at each iteration by the optimizer and the maximum number of epochs (specified with `Epochs`). If the objective function for your data is oscillating between Inf and 0, the step size is probably too large.  There are more parameters for the optimizers, but the C++ interface must be used to access these.

Optionally, the model can be used to predict the labels for another matrix of data points, if `Test` is specified.  The `Test` parameter can be specified without the `Training` parameter, so long as an existing linear SVM model is given with the `InputModel` parameter.  The output predictions from the linear SVM model may be saved with the `Predictions` parameter.

### Example
As an example, to train a LinaerSVM on the data '`data`' with labels '`labels`' with L2 regularization of 0.1, saving the model to '`lsvm_model`', the following command may be used:

```go
// Initialize optional parameters for LinearSvm().
param := mlpack.LinearSvmOptions()
param.Training = data
param.Labels = labels
param.Lambda = 0.1
param.Delta = 1
param.NumClasses = 0

lsvm_model, _, _ := mlpack.LinearSvm(param)
```

Then, to use that model to predict classes for the dataset '`test`', storing the output predictions in '`predictions`', the following command may be used: 

```go
// Initialize optional parameters for LinearSvm().
param := mlpack.LinearSvmOptions()
param.InputModel = &lsvm_model
param.Test = test

_, predictions, _ := mlpack.LinearSvm(param)
```

### See also

 - [RandomForest()](#random_forest)
 - [LogisticRegression()](#logistic_regression)
 - [LinearSVM on Wikipedia](https://en.wikipedia.org/wiki/Support-vector_machine)
 - [LinearSVM C++ class documentation](../../user/methods/linear_svm.md)

## Lmnn()
{: #lmnn }

#### Large Margin Nearest Neighbors (LMNN)
{: #lmnn_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for Lmnn().
param := mlpack.LmnnOptions()
param.BatchSize = 50
param.Center = false
param.Distance = mat.NewDense(1, 1, nil)
param.K = 1
param.Labels = mat.NewDense(1, 1, nil)
param.LinearScan = false
param.MaxIterations = 100000
param.Normalize = false
param.Optimizer = "amsgrad"
param.Passes = 50
param.PrintAccuracy = false
param.Rank = 0
param.Regularization = 0.5
param.Seed = 0
param.StepSize = 0.01
param.Tolerance = 1e-07
param.UpdateInterval = 1
param.Verbose = false

centered_data, output, transformed_data := mlpack.Lmnn(input, param)
```

An implementation of Large Margin Nearest Neighbors (LMNN), a distance learning technique.  Given a labeled dataset, this learns a transformation of the data that improves k-nearest-neighbor performance; this can be useful as a preprocessing step. [Detailed documentation](#lmnn_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `BatchSize` | [`int`](#doc_int) | Batch size for mini-batch SGD. | `50` |
| `Center` | [`bool`](#doc_bool) | Perform mean-centering on the dataset. It is useful when the centroid of the data is far from the origin. | `false` |
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `Distance` | [`*mat.Dense`](#doc_a__mat_Dense) | Initial distance matrix to be used as starting point | `mat.NewDense(1, 1, nil)` |
| `input` | [`*mat.Dense`](#doc_a__mat_Dense) | Input dataset to run LMNN on. | `**--**` |
| `K` | [`int`](#doc_int) | Number of target neighbors to use for each datapoint. | `1` |
| `Labels` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | Labels for input dataset. | `mat.NewDense(1, 1, nil)` |
| `LinearScan` | [`bool`](#doc_bool) | Don't shuffle the order in which data points are visited for SGD or mini-batch SGD. | `false` |
| `MaxIterations` | [`int`](#doc_int) | Maximum number of iterations for L-BFGS (0 indicates no limit). | `100000` |
| `Normalize` | [`bool`](#doc_bool) | Use a normalized starting point for optimization. Itis useful for when points are far apart, or when SGD is returning NaN. | `false` |
| `Optimizer` | [`string`](#doc_string) | Optimizer to use; 'amsgrad', 'bbsgd', 'sgd', or 'lbfgs'. | `"amsgrad"` |
| `Passes` | [`int`](#doc_int) | Maximum number of full passes over dataset for AMSGrad, BB_SGD and SGD. | `50` |
| `PrintAccuracy` | [`bool`](#doc_bool) | Print accuracies on initial and transformed dataset | `false` |
| `Rank` | [`int`](#doc_int) | Rank of distance matrix to be optimized.  | `0` |
| `Regularization` | [`float64`](#doc_float64) | Regularization for LMNN objective function  | `0.5` |
| `Seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `StepSize` | [`float64`](#doc_float64) | Step size for AMSGrad, BB_SGD and SGD (alpha). | `0.01` |
| `Tolerance` | [`float64`](#doc_float64) | Maximum tolerance for termination of AMSGrad, BB_SGD, SGD or L-BFGS. | `1e-07` |
| `UpdateInterval` | [`int`](#doc_int) | Number of iterations after which impostors need to be recalculated. | `1` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `CenteredData` | [`*mat.Dense`](#doc_a__mat_Dense) | Output matrix for mean-centered dataset. | 
| `Output` | [`*mat.Dense`](#doc_a__mat_Dense) | Output matrix for learned distance matrix. | 
| `TransformedData` | [`*mat.Dense`](#doc_a__mat_Dense) | Output matrix for transformed dataset. | 

### Detailed documentation
{: #lmnn_detailed-documentation }

This program implements Large Margin Nearest Neighbors, a distance learning technique.  The method seeks to improve k-nearest-neighbor classification on a dataset.  The method employes the strategy of reducing distance between similar labeled data points (a.k.a target neighbors) and increasing distance between differently labeled points (a.k.a impostors) using standard optimization techniques over the gradient of the distance between data points.

To work, this algorithm needs labeled data.  It can be given as the last row of the input dataset (specified with `Input`), or alternatively as a separate matrix (specified with `Labels`).  Additionally, a starting point for optimization (specified with `Distance`can be given, having (r x d) dimensionality.  Here r should satisfy 1 <= r <= d, Consequently a Low-Rank matrix will be optimized. Alternatively, Low-Rank distance can be learned by specifying the `Rank`parameter (A Low-Rank matrix with uniformly distributed values will be used as initial learning point). 

The program also requires number of targets neighbors to work with ( specified with `K`), A regularization parameter can also be passed, It acts as a trade of between the pulling and pushing terms (specified with `Regularization`), In addition, this implementation of LMNN includes a parameter to decide the interval after which impostors must be re-calculated (specified with `UpdateInterval`).

Output can either be the learned distance matrix (specified with `Output`), or the transformed dataset  (specified with `TransformedData`), or both. Additionally mean-centered dataset (specified with `CenteredData`) can be accessed given mean-centering (specified with `Center`) is performed on the dataset. Accuracy on initial dataset and final transformed dataset can be printed by specifying the `PrintAccuracy`parameter. 

This implementation of LMNN uses AdaGrad, BigBatch_SGD, stochastic gradient descent, mini-batch stochastic gradient descent, or the L_BFGS optimizer. 

AdaGrad, specified by the value 'adagrad' for the parameter `Optimizer`, uses maximum of past squared gradients. It primarily on six parameters: the step size (specified with `StepSize`), the batch size (specified with `BatchSize`), the maximum number of passes (specified with `Passes`). Inaddition, a normalized starting point can be used by specifying the `Normalize` parameter. 

BigBatch_SGD, specified by the value 'bbsgd' for the parameter `Optimizer`, depends primarily on four parameters: the step size (specified with `StepSize`), the batch size (specified with `BatchSize`), the maximum number of passes (specified with `Passes`).  In addition, a normalized starting point can be used by specifying the `Normalize` parameter. 

Stochastic gradient descent, specified by the value 'sgd' for the parameter `Optimizer`, depends primarily on three parameters: the step size (specified with `StepSize`), the batch size (specified with `BatchSize`), and the maximum number of passes (specified with `Passes`).  In addition, a normalized starting point can be used by specifying the `Normalize` parameter. Furthermore, mean-centering can be performed on the dataset by specifying the `Center`parameter. 

The L-BFGS optimizer, specified by the value 'lbfgs' for the parameter `Optimizer`, uses a back-tracking line search algorithm to minimize a function.  The following parameters are used by L-BFGS: `MaxIterations`, `Tolerance`(the optimization is terminated when the gradient norm is below this value).  For more details on the L-BFGS optimizer, consult either the mlpack L-BFGS documentation (in lbfgs.hpp) or the vast set of published literature on L-BFGS.  In addition, a normalized starting point can be used by specifying the `Normalize` parameter.

By default, the AMSGrad optimizer is used.

### Example
Example - Let's say we want to learn distance on iris dataset with number of targets as 3 using BigBatch_SGD optimizer. A simple call for the same will look like: 

```go
// Initialize optional parameters for Lmnn().
param := mlpack.LmnnOptions()
param.Labels = iris_labels
param.K = 3
param.Optimizer = "bbsgd"

_, output, _ := mlpack.Lmnn(iris, param)
```

Another program call making use of update interval & regularization parameter with dataset having labels as last column can be made as: 

```go
// Initialize optional parameters for Lmnn().
param := mlpack.LmnnOptions()
param.K = 5
param.UpdateInterval = 10
param.Regularization = 0.4

_, output, _ := mlpack.Lmnn(letter_recognition, param)
```

### See also

 - [Nca()](#nca)
 - [Large margin nearest neighbor on Wikipedia](https://en.wikipedia.org/wiki/Large_margin_nearest_neighbor)
 - [Distance metric learning for large margin nearest neighbor classification (pdf)](https://proceedings.neurips.cc/paper_files/paper/2005/file/a7f592cef8b130a6967a90617db5681b-Paper.pdf)
 - [LMNN C++ class documentation](../../user/methods/lmnn.md)

## LocalCoordinateCoding()
{: #local_coordinate_coding }

#### Local Coordinate Coding
{: #local_coordinate_coding_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for LocalCoordinateCoding().
param := mlpack.LocalCoordinateCodingOptions()
param.Atoms = 0
param.InitialDictionary = mat.NewDense(1, 1, nil)
param.InputModel = nil
param.Lambda = 0
param.MaxIterations = 0
param.Normalize = false
param.Seed = 0
param.Test = mat.NewDense(1, 1, nil)
param.Tolerance = 0.01
param.Training = mat.NewDense(1, 1, nil)
param.Verbose = false

codes, dictionary, output_model := mlpack.LocalCoordinateCoding(param)
```

An implementation of Local Coordinate Coding (LCC), a data transformation technique.  Given input data, this transforms each point to be expressed as a linear combination of a few points in the dataset; once an LCC model is trained, it can be used to transform points later also. [Detailed documentation](#local_coordinate_coding_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `Atoms` | [`int`](#doc_int) | Number of atoms in the dictionary. | `0` |
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `InitialDictionary` | [`*mat.Dense`](#doc_a__mat_Dense) | Optional initial dictionary. | `mat.NewDense(1, 1, nil)` |
| `InputModel` | [`localCoordinateCoding`](#doc_model) | Input LCC model. | `nil` |
| `Lambda` | [`float64`](#doc_float64) | Weighted l1-norm regularization parameter. | `0` |
| `MaxIterations` | [`int`](#doc_int) | Maximum number of iterations for LCC (0 indicates no limit). | `0` |
| `Normalize` | [`bool`](#doc_bool) | If set, the input data matrix will be normalized before coding. | `false` |
| `Seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `Test` | [`*mat.Dense`](#doc_a__mat_Dense) | Test points to encode. | `mat.NewDense(1, 1, nil)` |
| `Tolerance` | [`float64`](#doc_float64) | Tolerance for objective function. | `0.01` |
| `Training` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix of training data (X). | `mat.NewDense(1, 1, nil)` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Codes` | [`*mat.Dense`](#doc_a__mat_Dense) | Output codes matrix. | 
| `Dictionary` | [`*mat.Dense`](#doc_a__mat_Dense) | Output dictionary matrix. | 
| `OutputModel` | [`localCoordinateCoding`](#doc_model) | Output for trained LCC model. | 

### Detailed documentation
{: #local_coordinate_coding_detailed-documentation }

An implementation of Local Coordinate Coding (LCC), which codes data that approximately lives on a manifold using a variation of l1-norm regularized sparse coding.  Given a dense data matrix X with n points and d dimensions, LCC seeks to find a dense dictionary matrix D with k atoms in d dimensions, and a coding matrix Z with n points in k dimensions.  Because of the regularization method used, the atoms in D should lie close to the manifold on which the data points lie.

The original data matrix X can then be reconstructed as D * Z.  Therefore, this program finds a representation of each point in X as a sparse linear combination of atoms in the dictionary D.

The coding is found with an algorithm which alternates between a dictionary step, which updates the dictionary D, and a coding step, which updates the coding matrix Z.

To run this program, the input matrix X must be specified (with -i), along with the number of atoms in the dictionary (-k).  An initial dictionary may also be specified with the `InitialDictionary` parameter.  The l1-norm regularization parameter is specified with the `Lambda` parameter.

### Example
For example, to run LCC on the dataset `data` using 200 atoms and an l1-regularization parameter of 0.1, saving the dictionary `Dictionary` and the codes into `Codes`, use

```go
// Initialize optional parameters for LocalCoordinateCoding().
param := mlpack.LocalCoordinateCodingOptions()
param.Training = data
param.Atoms = 200
param.Lambda = 0.1

codes, dict, _ := mlpack.LocalCoordinateCoding(param)
```

The maximum number of iterations may be specified with the `MaxIterations` parameter. Optionally, the input data matrix X can be normalized before coding with the `Normalize` parameter.

An LCC model may be saved using the `OutputModel` output parameter.  Then, to encode new points from the dataset `points` with the previously saved model `lcc_model`, saving the new codes to `new_codes`, the following command can be used:

```go
// Initialize optional parameters for LocalCoordinateCoding().
param := mlpack.LocalCoordinateCodingOptions()
param.InputModel = &lcc_model
param.Test = points

new_codes, _, _ := mlpack.LocalCoordinateCoding(param)
```

### See also

 - [SparseCoding()](#sparse_coding)
 - [Nonlinear learning using local coordinate coding (pdf)](https://proceedings.neurips.cc/paper_files/paper/2009/file/2afe4567e1bf64d32a5527244d104cea-Paper.pdf)
 - [LocalCoordinateCoding C++ class documentation](../../user/methods/local_coordinate_coding.md)

## LogisticRegression()
{: #logistic_regression }

#### L2-regularized Logistic Regression and Prediction
{: #logistic_regression_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for LogisticRegression().
param := mlpack.LogisticRegressionOptions()
param.BatchSize = 64
param.DecisionBoundary = 0.5
param.InputModel = nil
param.Labels = mat.NewDense(1, 1, nil)
param.Lambda = 0
param.MaxIterations = 10000
param.Optimizer = "lbfgs"
param.PrintTrainingAccuracy = false
param.StepSize = 0.01
param.Test = mat.NewDense(1, 1, nil)
param.Tolerance = 1e-10
param.Training = mat.NewDense(1, 1, nil)
param.Verbose = false

output_model, predictions, probabilities :=
    mlpack.LogisticRegression(param)
```

An implementation of L2-regularized logistic regression for two-class classification.  Given labeled data, a model can be trained and saved for future use; or, a pre-trained model can be used to classify new points. [Detailed documentation](#logistic_regression_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `BatchSize` | [`int`](#doc_int) | Batch size for SGD. | `64` |
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `DecisionBoundary` | [`float64`](#doc_float64) | Decision boundary for prediction; if the logistic function for a point is less than the boundary, the class is taken to be 0; otherwise, the class is 1. | `0.5` |
| `InputModel` | [`logisticRegression`](#doc_model) | Existing model (parameters). | `nil` |
| `Labels` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | A matrix containing labels (0 or 1) for the points in the training set (y). | `mat.NewDense(1, 1, nil)` |
| `Lambda` | [`float64`](#doc_float64) | L2-regularization parameter for training. | `0` |
| `MaxIterations` | [`int`](#doc_int) | Maximum iterations for optimizer (0 indicates no limit). | `10000` |
| `Optimizer` | [`string`](#doc_string) | Optimizer to use for training ('lbfgs' or 'sgd'). | `"lbfgs"` |
| `PrintTrainingAccuracy` | [`bool`](#doc_bool) | If set, then the accuracy of the model on the training set will be printed (verbose must also be specified). | `false` |
| `StepSize` | [`float64`](#doc_float64) | Step size for SGD optimizer. | `0.01` |
| `Test` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix containing test dataset. | `mat.NewDense(1, 1, nil)` |
| `Tolerance` | [`float64`](#doc_float64) | Convergence tolerance for optimizer. | `1e-10` |
| `Training` | [`*mat.Dense`](#doc_a__mat_Dense) | A matrix containing the training set (the matrix of predictors, X). | `mat.NewDense(1, 1, nil)` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `OutputModel` | [`logisticRegression`](#doc_model) | Output for trained logistic regression model. | 
| `Predictions` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | If test data is specified, this matrix is where the predictions for the test set will be saved. | 
| `Probabilities` | [`*mat.Dense`](#doc_a__mat_Dense) | If test data is specified, this matrix is where the class probabilities for the test set will be saved. | 

### Detailed documentation
{: #logistic_regression_detailed-documentation }

An implementation of L2-regularized logistic regression using either the L-BFGS optimizer or SGD (stochastic gradient descent).  This solves the regression problem

  y = (1 / 1 + e^-(X * b)).

In this setting, y corresponds to class labels and X corresponds to data.

This program allows loading a logistic regression model (via the `InputModel` parameter) or training a logistic regression model given training data (specified with the `Training` parameter), or both those things at once.  In addition, this program allows classification on a test dataset (specified with the `Test` parameter) and the classification results may be saved with the `Predictions` output parameter. The trained logistic regression model may be saved using the `OutputModel` output parameter.

The training data, if specified, may have class labels as its last dimension.  Alternately, the `Labels` parameter may be used to specify a separate matrix of labels.

When a model is being trained, there are many options.  L2 regularization (to prevent overfitting) can be specified with the `Lambda` option, and the optimizer used to train the model can be specified with the `Optimizer` parameter.  Available options are 'sgd' (stochastic gradient descent) and 'lbfgs' (the L-BFGS optimizer).  There are also various parameters for the optimizer; the `MaxIterations` parameter specifies the maximum number of allowed iterations, and the `Tolerance` parameter specifies the tolerance for convergence.  For the SGD optimizer, the `StepSize` parameter controls the step size taken at each iteration by the optimizer.  The batch size for SGD is controlled with the `BatchSize` parameter. If the objective function for your data is oscillating between Inf and 0, the step size is probably too large.  There are more parameters for the optimizers, but the C++ interface must be used to access these.

For SGD, an iteration refers to a single point. So to take a single pass over the dataset with SGD, `MaxIterations` should be set to the number of points in the dataset.

Optionally, the model can be used to predict the responses for another matrix of data points, if `Test` is specified.  The `Test` parameter can be specified without the `Training` parameter, so long as an existing logistic regression model is given with the `InputModel` parameter.  The output predictions from the logistic regression model may be saved with the `Predictions` parameter.

This implementation of logistic regression does not support the general multi-class case but instead only the two-class case.  Any labels must be either 0 or 1.  For more classes, see the softmax regression implementation.

### Example
As an example, to train a logistic regression model on the data '`data`' with labels '`labels`' with L2 regularization of 0.1, saving the model to '`lr_model`', the following command may be used:

```go
// Initialize optional parameters for LogisticRegression().
param := mlpack.LogisticRegressionOptions()
param.Training = data
param.Labels = labels
param.Lambda = 0.1
param.PrintTrainingAccuracy = true

lr_model, _, _ := mlpack.LogisticRegression(param)
```

Then, to use that model to predict classes for the dataset '`test`', storing the output predictions in '`predictions`', the following command may be used: 

```go
// Initialize optional parameters for LogisticRegression().
param := mlpack.LogisticRegressionOptions()
param.InputModel = &lr_model
param.Test = test

_, predictions, _ := mlpack.LogisticRegression(param)
```

### See also

 - [SoftmaxRegression()](#softmax_regression)
 - [RandomForest()](#random_forest)
 - [Logistic regression on Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)
 - [:LogisticRegression C++ class documentation](../../user/methods/logistic_regression.md)

## Lsh()
{: #lsh }

#### K-Approximate-Nearest-Neighbor Search with LSH
{: #lsh_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for Lsh().
param := mlpack.LshOptions()
param.BucketSize = 500
param.HashWidth = 0
param.InputModel = nil
param.K = 0
param.NumProbes = 0
param.Projections = 10
param.Query = mat.NewDense(1, 1, nil)
param.Reference = mat.NewDense(1, 1, nil)
param.SecondHashSize = 99901
param.Seed = 0
param.Tables = 30
param.TrueNeighbors = mat.NewDense(1, 1, nil)
param.Verbose = false

distances, neighbors, output_model := mlpack.Lsh(param)
```

An implementation of approximate k-nearest-neighbor search with locality-sensitive hashing (LSH).  Given a set of reference points and a set of query points, this will compute the k approximate nearest neighbors of each query point in the reference set; models can be saved for future use. [Detailed documentation](#lsh_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `BucketSize` | [`int`](#doc_int) | The size of a bucket in the second level hash. | `500` |
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `HashWidth` | [`float64`](#doc_float64) | The hash width for the first-level hashing in the LSH preprocessing. By default, the LSH class automatically estimates a hash width for its use. | `0` |
| `InputModel` | [`lshSearch`](#doc_model) | Input LSH model. | `nil` |
| `K` | [`int`](#doc_int) | Number of nearest neighbors to find. | `0` |
| `NumProbes` | [`int`](#doc_int) | Number of additional probes for multiprobe LSH; if 0, traditional LSH is used. | `0` |
| `Projections` | [`int`](#doc_int) | The number of hash functions for each table | `10` |
| `Query` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix containing query points (optional). | `mat.NewDense(1, 1, nil)` |
| `Reference` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix containing the reference dataset. | `mat.NewDense(1, 1, nil)` |
| `SecondHashSize` | [`int`](#doc_int) | The size of the second level hash table. | `99901` |
| `Seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `Tables` | [`int`](#doc_int) | The number of hash tables to be used. | `30` |
| `TrueNeighbors` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix of true neighbors to compute recall with (the recall is printed when -v is specified). | `mat.NewDense(1, 1, nil)` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Distances` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to output distances into. | 
| `Neighbors` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to output neighbors into. | 
| `OutputModel` | [`lshSearch`](#doc_model) | Output for trained LSH model. | 

### Detailed documentation
{: #lsh_detailed-documentation }

This program will calculate the k approximate-nearest-neighbors of a set of points using locality-sensitive hashing. You may specify a separate set of reference points and query points, or just a reference set which will be used as both the reference and query set. 

### Example
For example, the following will return 5 neighbors from the data for each point in `input` and store the distances in `distances` and the neighbors in `neighbors`:

```go
// Initialize optional parameters for Lsh().
param := mlpack.LshOptions()
param.K = 5
param.Reference = input

distances, neighbors, _ := mlpack.Lsh(param)
```

The output is organized such that row i and column j in the neighbors output corresponds to the index of the point in the reference set which is the j'th nearest neighbor from the point in the query set with index i.  Row j and column i in the distances output file corresponds to the distance between those two points.

Because this is approximate-nearest-neighbors search, results may be different from run to run.  Thus, the `Seed` parameter can be specified to set the random seed.

This program also has many other parameters to control its functionality; see the parameter-specific documentation for more information.

### See also

 - [Knn()](#knn)
 - [Krann()](#krann)
 - [Locality-sensitive hashing on Wikipedia](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)
 - [Locality-sensitive hashing scheme based on p-stable  distributions(pdf)](https://www.mlpack.org/papers/lsh.pdf)
 - [LSHSearch C++ class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/lsh/lsh.hpp)

## MeanShift()
{: #mean_shift }

#### Mean Shift Clustering
{: #mean_shift_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for MeanShift().
param := mlpack.MeanShiftOptions()
param.ForceConvergence = false
param.InPlace = false
param.LabelsOnly = false
param.MaxIterations = 1000
param.Radius = 0
param.Verbose = false

centroid, output := mlpack.MeanShift(input, param)
```

A fast implementation of mean-shift clustering using dual-tree range search.  Given a dataset, this uses the mean shift algorithm to produce and return a clustering of the data. [Detailed documentation](#mean_shift_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `ForceConvergence` | [`bool`](#doc_bool) | If specified, the mean shift algorithm will continue running regardless of max_iterations until the clusters converge. | `false` |
| `InPlace` | [`bool`](#doc_bool) | If specified, a column containing the learned cluster assignments will be added to the input dataset file.  In this case, --output_file is overridden.  (Do not use with Python.) | `false` |
| `input` | [`*mat.Dense`](#doc_a__mat_Dense) | Input dataset to perform clustering on. | `**--**` |
| `LabelsOnly` | [`bool`](#doc_bool) | If specified, only the output labels will be written to the file specified by --output_file. | `false` |
| `MaxIterations` | [`int`](#doc_int) | Maximum number of iterations before mean shift terminates. | `1000` |
| `Radius` | [`float64`](#doc_float64) | If the distance between two centroids is less than the given radius, one will be removed.  A radius of 0 or less means an estimate will be calculated and used for the radius. | `0` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Centroid` | [`*mat.Dense`](#doc_a__mat_Dense) | If specified, the centroids of each cluster will be written to the given matrix. | 
| `Output` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to write output labels or labeled data to. | 

### Detailed documentation
{: #mean_shift_detailed-documentation }

This program performs mean shift clustering on the given dataset, storing the learned cluster assignments either as a column of labels in the input dataset or separately.

The input dataset should be specified with the `Input` parameter, and the radius used for search can be specified with the `Radius` parameter.  The maximum number of iterations before algorithm termination is controlled with the `MaxIterations` parameter.

The output labels may be saved with the `Output` output parameter and the centroids of each cluster may be saved with the `Centroid` output parameter.

### Example
For example, to run mean shift clustering on the dataset `data` and store the centroids to `centroids`, the following command may be used: 

```go
// Initialize optional parameters for MeanShift().
param := mlpack.MeanShiftOptions()

centroids, _ := mlpack.MeanShift(data, param)
```

### See also

 - [Kmeans()](#kmeans)
 - [Dbscan()](#dbscan)
 - [Mean shift on Wikipedia](https://en.wikipedia.org/wiki/Mean_shift)
 - [Mean Shift, Mode Seeking, and Clustering (pdf)](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=1c168275c59ba382588350ee1443537f59978183)
 - [mlpack::mean_shift::MeanShift C++ class documentation](../../user/methods/mean_shift.md)

## Nbc()
{: #nbc }

#### Parametric Naive Bayes Classifier
{: #nbc_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for Nbc().
param := mlpack.NbcOptions()
param.IncrementalVariance = false
param.InputModel = nil
param.Labels = mat.NewDense(1, 1, nil)
param.Test = mat.NewDense(1, 1, nil)
param.Training = mat.NewDense(1, 1, nil)
param.Verbose = false

output_model, predictions, probabilities := mlpack.Nbc(param)
```

An implementation of the Naive Bayes Classifier, used for classification. Given labeled data, an NBC model can be trained and saved, or, a pre-trained model can be used for classification. [Detailed documentation](#nbc_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `IncrementalVariance` | [`bool`](#doc_bool) | The variance of each class will be calculated incrementally. | `false` |
| `InputModel` | [`nbcModel`](#doc_model) | Input Naive Bayes model. | `nil` |
| `Labels` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | A file containing labels for the training set. | `mat.NewDense(1, 1, nil)` |
| `Test` | [`*mat.Dense`](#doc_a__mat_Dense) | A matrix containing the test set. | `mat.NewDense(1, 1, nil)` |
| `Training` | [`*mat.Dense`](#doc_a__mat_Dense) | A matrix containing the training set. | `mat.NewDense(1, 1, nil)` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `OutputModel` | [`nbcModel`](#doc_model) | File to save trained Naive Bayes model to. | 
| `Predictions` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | The matrix in which the predicted labels for the test set will be written. | 
| `Probabilities` | [`*mat.Dense`](#doc_a__mat_Dense) | The matrix in which the predicted probability of labels for the test set will be written. | 

### Detailed documentation
{: #nbc_detailed-documentation }

This program trains the Naive Bayes classifier on the given labeled training set, or loads a model from the given model file, and then may use that trained model to classify the points in a given test set.

The training set is specified with the `Training` parameter.  Labels may be either the last row of the training set, or alternately the `Labels` parameter may be specified to pass a separate matrix of labels.

If training is not desired, a pre-existing model may be loaded with the `InputModel` parameter.



The `IncrementalVariance` parameter can be used to force the training to use an incremental algorithm for calculating variance.  This is slower, but can help avoid loss of precision in some cases.

If classifying a test set is desired, the test set may be specified with the `Test` parameter, and the classifications may be saved with the `Predictions`predictions  parameter.  If saving the trained model is desired, this may be done with the `OutputModel` output parameter.

### Example
For example, to train a Naive Bayes classifier on the dataset `data` with labels `labels` and save the model to `nbc_model`, the following command may be used:

```go
// Initialize optional parameters for Nbc().
param := mlpack.NbcOptions()
param.Training = data
param.Labels = labels

nbc_model, _, _ := mlpack.Nbc(param)
```

Then, to use `nbc_model` to predict the classes of the dataset `test_set` and save the predicted classes to `predictions`, the following command may be used:

```go
// Initialize optional parameters for Nbc().
param := mlpack.NbcOptions()
param.InputModel = &nbc_model
param.Test = test_set

_, predictions, _ := mlpack.Nbc(param)
```

### See also

 - [SoftmaxRegression()](#softmax_regression)
 - [RandomForest()](#random_forest)
 - [Naive Bayes classifier on Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
 - [NaiveBayesClassifier C++ class documentation](../../user/methods/naive_bayes_classifier.md)

## Nca()
{: #nca }

#### Neighborhood Components Analysis (NCA)
{: #nca_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for Nca().
param := mlpack.NcaOptions()
param.ArmijoConstant = 0.0001
param.BatchSize = 50
param.Labels = mat.NewDense(1, 1, nil)
param.LinearScan = false
param.MaxIterations = 500000
param.MaxLineSearchTrials = 50
param.MaxStep = 1e+20
param.MinStep = 1e-20
param.Normalize = false
param.NumBasis = 5
param.Optimizer = "sgd"
param.Seed = 0
param.StepSize = 0.01
param.Tolerance = 1e-07
param.Verbose = false
param.Wolfe = 0.9

output := mlpack.Nca(input, param)
```

An implementation of neighborhood components analysis, a distance learning technique that can be used for preprocessing.  Given a labeled dataset, this uses NCA, which seeks to improve the k-nearest-neighbor classification, and returns the learned distance metric. [Detailed documentation](#nca_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `ArmijoConstant` | [`float64`](#doc_float64) | Armijo constant for L-BFGS. | `0.0001` |
| `BatchSize` | [`int`](#doc_int) | Batch size for mini-batch SGD. | `50` |
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `input` | [`*mat.Dense`](#doc_a__mat_Dense) | Input dataset to run NCA on. | `**--**` |
| `Labels` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | Labels for input dataset. | `mat.NewDense(1, 1, nil)` |
| `LinearScan` | [`bool`](#doc_bool) | Don't shuffle the order in which data points are visited for SGD or mini-batch SGD. | `false` |
| `MaxIterations` | [`int`](#doc_int) | Maximum number of iterations for SGD or L-BFGS (0 indicates no limit). | `500000` |
| `MaxLineSearchTrials` | [`int`](#doc_int) | Maximum number of line search trials for L-BFGS. | `50` |
| `MaxStep` | [`float64`](#doc_float64) | Maximum step of line search for L-BFGS. | `1e+20` |
| `MinStep` | [`float64`](#doc_float64) | Minimum step of line search for L-BFGS. | `1e-20` |
| `Normalize` | [`bool`](#doc_bool) | Use a normalized starting point for optimization. This is useful for when points are far apart, or when SGD is returning NaN. | `false` |
| `NumBasis` | [`int`](#doc_int) | Number of memory points to be stored for L-BFGS. | `5` |
| `Optimizer` | [`string`](#doc_string) | Optimizer to use; 'sgd' or 'lbfgs'. | `"sgd"` |
| `Seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `StepSize` | [`float64`](#doc_float64) | Step size for stochastic gradient descent (alpha). | `0.01` |
| `Tolerance` | [`float64`](#doc_float64) | Maximum tolerance for termination of SGD or L-BFGS. | `1e-07` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |
| `Wolfe` | [`float64`](#doc_float64) | Wolfe condition parameter for L-BFGS. | `0.9` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Output` | [`*mat.Dense`](#doc_a__mat_Dense) | Output matrix for learned distance matrix. | 

### Detailed documentation
{: #nca_detailed-documentation }

This program implements Neighborhood Components Analysis, both a linear dimensionality reduction technique and a distance learning technique.  The method seeks to improve k-nearest-neighbor classification on a dataset by scaling the dimensions.  The method is nonparametric, and does not require a value of k.  It works by using stochastic ("soft") neighbor assignments and using optimization techniques over the gradient of the accuracy of the neighbor assignments.

To work, this algorithm needs labeled data.  It can be given as the last row of the input dataset (specified with `Input`), or alternatively as a separate matrix (specified with `Labels`).

This implementation of NCA uses stochastic gradient descent, mini-batch stochastic gradient descent, or the L_BFGS optimizer.  These optimizers do not guarantee global convergence for a nonconvex objective function (NCA's objective function is nonconvex), so the final results could depend on the random seed or other optimizer parameters.

Stochastic gradient descent, specified by the value 'sgd' for the parameter `Optimizer`, depends primarily on three parameters: the step size (specified with `StepSize`), the batch size (specified with `BatchSize`), and the maximum number of iterations (specified with `MaxIterations`).  In addition, a normalized starting point can be used by specifying the `Normalize` parameter, which is necessary if many warnings of the form 'Denominator of p_i is 0!' are given.  Tuning the step size can be a tedious affair.  In general, the step size is too large if the objective is not mostly uniformly decreasing, or if zero-valued denominator warnings are being issued.  The step size is too small if the objective is changing very slowly.  Setting the termination condition can be done easily once a good step size parameter is found; either increase the maximum iterations to a large number and allow SGD to find a minimum, or set the maximum iterations to 0 (allowing infinite iterations) and set the tolerance (specified by `Tolerance`) to define the maximum allowed difference between objectives for SGD to terminate.  Be careful---setting the tolerance instead of the maximum iterations can take a very long time and may actually never converge due to the properties of the SGD optimizer. Note that a single iteration of SGD refers to a single point, so to take a single pass over the dataset, set the value of the `MaxIterations` parameter equal to the number of points in the dataset.

The L-BFGS optimizer, specified by the value 'lbfgs' for the parameter `Optimizer`, uses a back-tracking line search algorithm to minimize a function.  The following parameters are used by L-BFGS: `NumBasis` (specifies the number of memory points used by L-BFGS), `MaxIterations`, `ArmijoConstant`, `Wolfe`, `Tolerance` (the optimization is terminated when the gradient norm is below this value), `MaxLineSearchTrials`, `MinStep`, and `MaxStep` (which both refer to the line search routine).  For more details on the L-BFGS optimizer, consult either the mlpack L-BFGS documentation (in lbfgs.hpp) or the vast set of published literature on L-BFGS.

By default, the SGD optimizer is used.

### See also

 - [Lmnn()](#lmnn)
 - [Neighbourhood components analysis on Wikipedia](https://en.wikipedia.org/wiki/Neighbourhood_components_analysis)
 - [Neighbourhood components analysis (pdf)](https://proceedings.neurips.cc/paper_files/paper/2004/file/42fe880812925e520249e808937738d2-Paper.pdf)
 - [NCA C++ class documentation](../../user/methods/nca.md)

## Knn()
{: #knn }

#### k-Nearest-Neighbors Search
{: #knn_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for Knn().
param := mlpack.KnnOptions()
param.Algorithm = "dual_tree"
param.Epsilon = 0
param.InputModel = nil
param.K = 0
param.LeafSize = 20
param.Query = mat.NewDense(1, 1, nil)
param.RandomBasis = false
param.Reference = mat.NewDense(1, 1, nil)
param.Rho = 0.7
param.Seed = 0
param.Tau = 0
param.TreeType = "kd"
param.TrueDistances = mat.NewDense(1, 1, nil)
param.TrueNeighbors = mat.NewDense(1, 1, nil)
param.Verbose = false

distances, neighbors, output_model := mlpack.Knn(param)
```

An implementation of k-nearest-neighbor search using single-tree and dual-tree algorithms.  Given a set of reference points and query points, this can find the k nearest neighbors in the reference set of each query point using trees; trees that are built can be saved for future use. [Detailed documentation](#knn_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `Algorithm` | [`string`](#doc_string) | Type of neighbor search: 'naive', 'single_tree', 'dual_tree', 'greedy'. | `"dual_tree"` |
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `Epsilon` | [`float64`](#doc_float64) | If specified, will do approximate nearest neighbor search with given relative error. | `0` |
| `InputModel` | [`knnModel`](#doc_model) | Pre-trained kNN model. | `nil` |
| `K` | [`int`](#doc_int) | Number of nearest neighbors to find. | `0` |
| `LeafSize` | [`int`](#doc_int) | Leaf size for tree building (used for kd-trees, vp trees, random projection trees, UB trees, R trees, R* trees, X trees, Hilbert R trees, R+ trees, R++ trees, spill trees, and octrees). | `20` |
| `Query` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix containing query points (optional). | `mat.NewDense(1, 1, nil)` |
| `RandomBasis` | [`bool`](#doc_bool) | Before tree-building, project the data onto a random orthogonal basis. | `false` |
| `Reference` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix containing the reference dataset. | `mat.NewDense(1, 1, nil)` |
| `Rho` | [`float64`](#doc_float64) | Balance threshold (only valid for spill trees). | `0.7` |
| `Seed` | [`int`](#doc_int) | Random seed (if 0, std::time(NULL) is used). | `0` |
| `Tau` | [`float64`](#doc_float64) | Overlapping size (only valid for spill trees). | `0` |
| `TreeType` | [`string`](#doc_string) | Type of tree to use: 'kd', 'vp', 'rp', 'max-rp', 'ub', 'cover', 'r', 'r-star', 'x', 'ball', 'hilbert-r', 'r-plus', 'r-plus-plus', 'spill', 'oct'. | `"kd"` |
| `TrueDistances` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix of true distances to compute the effective error (average relative error) (it is printed when -v is specified). | `mat.NewDense(1, 1, nil)` |
| `TrueNeighbors` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix of true neighbors to compute the recall (it is printed when -v is specified). | `mat.NewDense(1, 1, nil)` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Distances` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to output distances into. | 
| `Neighbors` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to output neighbors into. | 
| `OutputModel` | [`knnModel`](#doc_model) | If specified, the kNN model will be output here. | 

### Detailed documentation
{: #knn_detailed-documentation }

This program will calculate the k-nearest-neighbors of a set of points using kd-trees or cover trees (cover tree support is experimental and may be slow). You may specify a separate set of reference points and query points, or just a reference set which will be used as both the reference and query set.

### Example
For example, the following command will calculate the 5 nearest neighbors of each point in `input` and store the distances in `distances` and the neighbors in `neighbors`: 

```go
// Initialize optional parameters for Knn().
param := mlpack.KnnOptions()
param.K = 5
param.Reference = input

distances, neighbors, _ := mlpack.Knn(param)
```

The output is organized such that row i and column j in the neighbors output matrix corresponds to the index of the point in the reference set which is the j'th nearest neighbor from the point in the query set with index i.  Row j and column i in the distances output matrix corresponds to the distance between those two points.

### See also

 - [Lsh()](#lsh)
 - [Krann()](#krann)
 - [Kfn()](#kfn)
 - [Tree-independent dual-tree algorithms (pdf)](http://proceedings.mlr.press/v28/curtin13.pdf)
 - [NeighborSearch C++ class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/neighbor_search/neighbor_search.hpp)

## Kfn()
{: #kfn }

#### k-Furthest-Neighbors Search
{: #kfn_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for Kfn().
param := mlpack.KfnOptions()
param.Algorithm = "dual_tree"
param.Epsilon = 0
param.InputModel = nil
param.K = 0
param.LeafSize = 20
param.Percentage = 1
param.Query = mat.NewDense(1, 1, nil)
param.RandomBasis = false
param.Reference = mat.NewDense(1, 1, nil)
param.Seed = 0
param.TreeType = "kd"
param.TrueDistances = mat.NewDense(1, 1, nil)
param.TrueNeighbors = mat.NewDense(1, 1, nil)
param.Verbose = false

distances, neighbors, output_model := mlpack.Kfn(param)
```

An implementation of k-furthest-neighbor search using single-tree and dual-tree algorithms.  Given a set of reference points and query points, this can find the k furthest neighbors in the reference set of each query point using trees; trees that are built can be saved for future use. [Detailed documentation](#kfn_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `Algorithm` | [`string`](#doc_string) | Type of neighbor search: 'naive', 'single_tree', 'dual_tree', 'greedy'. | `"dual_tree"` |
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `Epsilon` | [`float64`](#doc_float64) | If specified, will do approximate furthest neighbor search with given relative error. Must be in the range [0,1). | `0` |
| `InputModel` | [`kfnModel`](#doc_model) | Pre-trained kFN model. | `nil` |
| `K` | [`int`](#doc_int) | Number of furthest neighbors to find. | `0` |
| `LeafSize` | [`int`](#doc_int) | Leaf size for tree building (used for kd-trees, vp trees, random projection trees, UB trees, R trees, R* trees, X trees, Hilbert R trees, R+ trees, R++ trees, and octrees). | `20` |
| `Percentage` | [`float64`](#doc_float64) | If specified, will do approximate furthest neighbor search. Must be in the range (0,1] (decimal form). Resultant neighbors will be at least (p*100) % of the distance as the true furthest neighbor. | `1` |
| `Query` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix containing query points (optional). | `mat.NewDense(1, 1, nil)` |
| `RandomBasis` | [`bool`](#doc_bool) | Before tree-building, project the data onto a random orthogonal basis. | `false` |
| `Reference` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix containing the reference dataset. | `mat.NewDense(1, 1, nil)` |
| `Seed` | [`int`](#doc_int) | Random seed (if 0, std::time(NULL) is used). | `0` |
| `TreeType` | [`string`](#doc_string) | Type of tree to use: 'kd', 'vp', 'rp', 'max-rp', 'ub', 'cover', 'r', 'r-star', 'x', 'ball', 'hilbert-r', 'r-plus', 'r-plus-plus', 'oct'. | `"kd"` |
| `TrueDistances` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix of true distances to compute the effective error (average relative error) (it is printed when -v is specified). | `mat.NewDense(1, 1, nil)` |
| `TrueNeighbors` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix of true neighbors to compute the recall (it is printed when -v is specified). | `mat.NewDense(1, 1, nil)` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Distances` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to output distances into. | 
| `Neighbors` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to output neighbors into. | 
| `OutputModel` | [`kfnModel`](#doc_model) | If specified, the kFN model will be output here. | 

### Detailed documentation
{: #kfn_detailed-documentation }

This program will calculate the k-furthest-neighbors of a set of points. You may specify a separate set of reference points and query points, or just a reference set which will be used as both the reference and query set.

### Example
For example, the following will calculate the 5 furthest neighbors of eachpoint in `input` and store the distances in `distances` and the neighbors in `neighbors`: 

```go
// Initialize optional parameters for Kfn().
param := mlpack.KfnOptions()
param.K = 5
param.Reference = input

distances, neighbors, _ := mlpack.Kfn(param)
```

The output files are organized such that row i and column j in the neighbors output matrix corresponds to the index of the point in the reference set which is the j'th furthest neighbor from the point in the query set with index i.  Row i and column j in the distances output file corresponds to the distance between those two points.

### See also

 - [ApproxKfn()](#approx_kfn)
 - [Knn()](#knn)
 - [Tree-independent dual-tree algorithms (pdf)](http://proceedings.mlr.press/v28/curtin13.pdf)
 - [NeighborSearch C++ class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/neighbor_search/neighbor_search.hpp)

## Nmf()
{: #nmf }

#### Non-negative Matrix Factorization
{: #nmf_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for Nmf().
param := mlpack.NmfOptions()
param.InitialH = mat.NewDense(1, 1, nil)
param.InitialW = mat.NewDense(1, 1, nil)
param.MaxIterations = 10000
param.MinResidue = 1e-05
param.Seed = 0
param.UpdateRules = "multdist"
param.Verbose = false

h, w := mlpack.Nmf(input, rank, param)
```

An implementation of non-negative matrix factorization.  This can be used to decompose an input dataset into two low-rank non-negative components. [Detailed documentation](#nmf_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `InitialH` | [`*mat.Dense`](#doc_a__mat_Dense) | Initial H matrix. | `mat.NewDense(1, 1, nil)` |
| `InitialW` | [`*mat.Dense`](#doc_a__mat_Dense) | Initial W matrix. | `mat.NewDense(1, 1, nil)` |
| `input` | [`*mat.Dense`](#doc_a__mat_Dense) | Input dataset to perform NMF on. | `**--**` |
| `MaxIterations` | [`int`](#doc_int) | Number of iterations before NMF terminates (0 runs until convergence. | `10000` |
| `MinResidue` | [`float64`](#doc_float64) | The minimum root mean square residue allowed for each iteration, below which the program terminates. | `1e-05` |
| `rank` | [`int`](#doc_int) | Rank of the factorization. | `**--**` |
| `Seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `UpdateRules` | [`string`](#doc_string) | Update rules for each iteration; ( multdist \| multdiv \| als ). | `"multdist"` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `H` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to save the calculated H to. | 
| `W` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to save the calculated W to. | 

### Detailed documentation
{: #nmf_detailed-documentation }

This program performs non-negative matrix factorization on the given dataset, storing the resulting decomposed matrices in the specified files.  For an input dataset V, NMF decomposes V into two matrices W and H such that 

V = W * H

where all elements in W and H are non-negative.  If V is of size (n x m), then W will be of size (n x r) and H will be of size (r x m), where r is the rank of the factorization (specified by the `Rank` parameter).

Optionally, the desired update rules for each NMF iteration can be chosen from the following list:

 - multdist: multiplicative distance-based update rules (Lee and Seung 1999)
 - multdiv: multiplicative divergence-based update rules (Lee and Seung 1999)
 - als: alternating least squares update rules (Paatero and Tapper 1994)

The maximum number of iterations is specified with `MaxIterations`, and the minimum residue required for algorithm termination is specified with the `MinResidue` parameter.

### Example
For example, to run NMF on the input matrix `V` using the 'multdist' update rules with a rank-10 decomposition and storing the decomposed matrices into `W` and `H`, the following command could be used: 

```go
// Initialize optional parameters for Nmf().
param := mlpack.NmfOptions()
param.UpdateRules = "multdist"

H, W := mlpack.Nmf(V, 10, param)
```

### See also

 - [Cf()](#cf)
 - [Non-negative matrix factorization on Wikipedia](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization)
 - [Algorithms for non-negative matrix factorization (pdf)](https://proceedings.neurips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf)
 - [NMF C++ class documentation](../../user/methods/nmf.md)
 - [AMF C++ class documentation](../../user/methods/amf.md)

## Pca()
{: #pca }

#### Principal Components Analysis
{: #pca_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for Pca().
param := mlpack.PcaOptions()
param.DecompositionMethod = "exact"
param.NewDimensionality = 0
param.Scale = false
param.VarToRetain = 0
param.Verbose = false

output := mlpack.Pca(input, param)
```

An implementation of several strategies for principal components analysis (PCA), a common preprocessing step.  Given a dataset and a desired new dimensionality, this can reduce the dimensionality of the data using the linear transformation determined by PCA. [Detailed documentation](#pca_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `DecompositionMethod` | [`string`](#doc_string) | Method used for the principal components analysis: 'exact', 'randomized', 'randomized-block-krylov', 'quic'. | `"exact"` |
| `input` | [`*mat.Dense`](#doc_a__mat_Dense) | Input dataset to perform PCA on. | `**--**` |
| `NewDimensionality` | [`int`](#doc_int) | Desired dimensionality of output dataset. If 0, no dimensionality reduction is performed. | `0` |
| `Scale` | [`bool`](#doc_bool) | If set, the data will be scaled before running PCA, such that the variance of each feature is 1. | `false` |
| `VarToRetain` | [`float64`](#doc_float64) | Amount of variance to retain; should be between 0 and 1.  If 1, all variance is retained.  Overrides -d. | `0` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Output` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to save modified dataset to. | 

### Detailed documentation
{: #pca_detailed-documentation }

This program performs principal components analysis on the given dataset using the exact, randomized, randomized block Krylov, or QUIC SVD method. It will transform the data onto its principal components, optionally performing dimensionality reduction by ignoring the principal components with the smallest eigenvalues.

Use the `Input` parameter to specify the dataset to perform PCA on.  A desired new dimensionality can be specified with the `NewDimensionality` parameter, or the desired variance to retain can be specified with the `VarToRetain` parameter.  If desired, the dataset can be scaled before running PCA with the `Scale` parameter.

Multiple different decomposition techniques can be used.  The method to use can be specified with the `DecompositionMethod` parameter, and it may take the values 'exact', 'randomized', or 'quic'.

### Example
For example, to reduce the dimensionality of the matrix `data` to 5 dimensions using randomized SVD for the decomposition, storing the output matrix to `data_mod`, the following command can be used:

```go
// Initialize optional parameters for Pca().
param := mlpack.PcaOptions()
param.NewDimensionality = 5
param.DecompositionMethod = "randomized"

data_mod := mlpack.Pca(data, param)
```

### See also

 - [Principal component analysis on Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)
 - [PCA C++ class documentation](../../user/methods/pca.md)

## Perceptron()
{: #perceptron }

#### Perceptron
{: #perceptron_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for Perceptron().
param := mlpack.PerceptronOptions()
param.InputModel = nil
param.Labels = mat.NewDense(1, 1, nil)
param.MaxIterations = 1000
param.Test = mat.NewDense(1, 1, nil)
param.Training = mat.NewDense(1, 1, nil)
param.Verbose = false

output_model, predictions := mlpack.Perceptron(param)
```

An implementation of a perceptron---a single level neural network--=for classification.  Given labeled data, a perceptron can be trained and saved for future use; or, a pre-trained perceptron can be used for classification on new points. [Detailed documentation](#perceptron_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `InputModel` | [`perceptronModel`](#doc_model) | Input perceptron model. | `nil` |
| `Labels` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | A matrix containing labels for the training set. | `mat.NewDense(1, 1, nil)` |
| `MaxIterations` | [`int`](#doc_int) | The maximum number of iterations the perceptron is to be run | `1000` |
| `Test` | [`*mat.Dense`](#doc_a__mat_Dense) | A matrix containing the test set. | `mat.NewDense(1, 1, nil)` |
| `Training` | [`*mat.Dense`](#doc_a__mat_Dense) | A matrix containing the training set. | `mat.NewDense(1, 1, nil)` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `OutputModel` | [`perceptronModel`](#doc_model) | Output for trained perceptron model. | 
| `Predictions` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | The matrix in which the predicted labels for the test set will be written. | 

### Detailed documentation
{: #perceptron_detailed-documentation }

This program implements a perceptron, which is a single level neural network. The perceptron makes its predictions based on a linear predictor function combining a set of weights with the feature vector.  The perceptron learning rule is able to converge, given enough iterations (specified using the `MaxIterations` parameter), if the data supplied is linearly separable.  The perceptron is parameterized by a matrix of weight vectors that denote the numerical weights of the neural network.

This program allows loading a perceptron from a model (via the `InputModel` parameter) or training a perceptron given training data (via the `Training` parameter), or both those things at once.  In addition, this program allows classification on a test dataset (via the `Test` parameter) and the classification results on the test set may be saved with the `Predictions` output parameter.  The perceptron model may be saved with the `OutputModel` output parameter.

### Example
The training data given with the `Training` option may have class labels as its last dimension (so, if the training data is in CSV format, labels should be the last column).  Alternately, the `Labels` parameter may be used to specify a separate matrix of labels.

All these options make it easy to train a perceptron, and then re-use that perceptron for later classification.  The invocation below trains a perceptron on `training_data` with labels `training_labels`, and saves the model to `perceptron_model`.

```go
// Initialize optional parameters for Perceptron().
param := mlpack.PerceptronOptions()
param.Training = training_data
param.Labels = training_labels

perceptron_model, _ := mlpack.Perceptron(param)
```

Then, this model can be re-used for classification on the test data `test_data`.  The example below does precisely that, saving the predicted classes to `predictions`.

```go
// Initialize optional parameters for Perceptron().
param := mlpack.PerceptronOptions()
param.InputModel = &perceptron_model
param.Test = test_data

_, predictions := mlpack.Perceptron(param)
```

Note that all of the options may be specified at once: predictions may be calculated right after training a model, and model training can occur even if an existing perceptron model is passed with the `InputModel` parameter.  However, note that the number of classes and the dimensionality of all data must match.  So you cannot pass a perceptron model trained on 2 classes and then re-train with a 4-class dataset.  Similarly, attempting classification on a 3-dimensional dataset with a perceptron that has been trained on 8 dimensions will cause an error.

### See also

 - [Adaboost()](#adaboost)
 - [Perceptron on Wikipedia](https://en.wikipedia.org/wiki/Perceptron)
 - [Perceptron C++ class documentation](../../user/methods/perceptron.md)

## PreprocessSplit()
{: #preprocess_split }

#### Split Data
{: #preprocess_split_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for PreprocessSplit().
param := mlpack.PreprocessSplitOptions()
param.InputLabels = mat.NewDense(1, 1, nil)
param.NoShuffle = false
param.Seed = 0
param.StratifyData = false
param.TestRatio = 0.2
param.Verbose = false

test, test_labels, training, training_labels :=
    mlpack.PreprocessSplit(input, param)
```

A utility to split data into a training and testing dataset.  This can also split labels according to the same split. [Detailed documentation](#preprocess_split_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `input` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix containing data. | `**--**` |
| `InputLabels` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix containing labels. | `mat.NewDense(1, 1, nil)` |
| `NoShuffle` | [`bool`](#doc_bool) | Avoid shuffling the data before splitting. | `false` |
| `Seed` | [`int`](#doc_int) | Random seed (0 for std::time(NULL)). | `0` |
| `StratifyData` | [`bool`](#doc_bool) | Stratify the data according to labels | `false` |
| `TestRatio` | [`float64`](#doc_float64) | Ratio of test set; if not set,the ratio defaults to 0.2 | `0.2` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Test` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to save test data to. | 
| `TestLabels` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to save test labels to. | 
| `Training` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to save training data to. | 
| `TrainingLabels` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to save train labels to. | 

### Detailed documentation
{: #preprocess_split_detailed-documentation }

This utility takes a dataset and optionally labels and splits them into a training set and a test set. Before the split, the points in the dataset are randomly reordered. The percentage of the dataset to be used as the test set can be specified with the `TestRatio` parameter; the default is 0.2 (20%).

The output training and test matrices may be saved with the `Training` and `Test` output parameters.

Optionally, labels can also be split along with the data by specifying the `InputLabels` parameter.  Splitting labels works the same way as splitting the data. The output training and test labels may be saved with the `TrainingLabels` and `TestLabels` output parameters, respectively.

### Example
So, a simple example where we want to split the dataset `X` into `X_train` and `X_test` with 60% of the data in the training set and 40% of the dataset in the test set, we could run 

```go
// Initialize optional parameters for PreprocessSplit().
param := mlpack.PreprocessSplitOptions()
param.TestRatio = 0.4

X_test, _, X_train, _ := mlpack.PreprocessSplit(X, param)
```

Also by default the dataset is shuffled and split; you can provide the `NoShuffle` option to avoid shuffling the data; an example to avoid shuffling of data is:

```go
// Initialize optional parameters for PreprocessSplit().
param := mlpack.PreprocessSplitOptions()
param.TestRatio = 0.4
param.NoShuffle = true

X_test, _, X_train, _ := mlpack.PreprocessSplit(X, param)
```

If we had a dataset `X` and associated labels `y`, and we wanted to split these into `X_train`, `y_train`, `X_test`, and `y_test`, with 30% of the data in the test set, we could run

```go
// Initialize optional parameters for PreprocessSplit().
param := mlpack.PreprocessSplitOptions()
param.InputLabels = y
param.TestRatio = 0.3

X_test, y_test, X_train, y_train := mlpack.PreprocessSplit(X, param)
```

To maintain the ratio of each class in the train and test sets, the`StratifyData` option can be used.

```go
// Initialize optional parameters for PreprocessSplit().
param := mlpack.PreprocessSplitOptions()
param.TestRatio = 0.4
param.StratifyData = true

X_test, _, X_train, _ := mlpack.PreprocessSplit(X, param)
```

### See also

 - [PreprocessBinarize()](#preprocess_binarize)
 - [PreprocessDescribe()](#preprocess_describe)

## PreprocessBinarize()
{: #preprocess_binarize }

#### Binarize Data
{: #preprocess_binarize_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for PreprocessBinarize().
param := mlpack.PreprocessBinarizeOptions()
param.Dimension = 0
param.Threshold = 0
param.Verbose = false

output := mlpack.PreprocessBinarize(input, param)
```

A utility to binarize a dataset.  Given a dataset, this utility converts each value in the desired dimension(s) to 0 or 1; this can be a useful preprocessing step. [Detailed documentation](#preprocess_binarize_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `Dimension` | [`int`](#doc_int) | Dimension to apply the binarization. If not set, the program will binarize every dimension by default. | `0` |
| `input` | [`*mat.Dense`](#doc_a__mat_Dense) | Input data matrix. | `**--**` |
| `Threshold` | [`float64`](#doc_float64) | Threshold to be applied for binarization. If not set, the threshold defaults to 0.0. | `0` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Output` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix in which to save the output. | 

### Detailed documentation
{: #preprocess_binarize_detailed-documentation }

This utility takes a dataset and binarizes the variables into either 0 or 1 given threshold. User can apply binarization on a dimension or the whole dataset.  The dimension to apply binarization to can be specified using the `Dimension` parameter; if left unspecified, every dimension will be binarized.  The threshold for binarization can also be specified with the `Threshold` parameter; the default threshold is 0.0.

The binarized matrix may be saved with the `Output` output parameter.

### Example
For example, if we want to set all variables greater than 5 in the dataset `X` to 1 and variables less than or equal to 5.0 to 0, and save the result to `Y`, we could run

```go
// Initialize optional parameters for PreprocessBinarize().
param := mlpack.PreprocessBinarizeOptions()
param.Threshold = 5

Y := mlpack.PreprocessBinarize(X, param)
```

But if we want to apply this to only the first (0th) dimension of `X`,  we could instead run

```go
// Initialize optional parameters for PreprocessBinarize().
param := mlpack.PreprocessBinarizeOptions()
param.Threshold = 5
param.Dimension = 0

Y := mlpack.PreprocessBinarize(X, param)
```

### See also

 - [PreprocessDescribe()](#preprocess_describe)
 - [PreprocessSplit()](#preprocess_split)

## PreprocessDescribe()
{: #preprocess_describe }

#### Descriptive Statistics
{: #preprocess_describe_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for PreprocessDescribe().
param := mlpack.PreprocessDescribeOptions()
param.Dimension = 0
param.Population = false
param.Precision = 4
param.RowMajor = false
param.Verbose = false
param.Width = 8

 := mlpack.PreprocessDescribe(input, param)
```

A utility for printing descriptive statistics about a dataset.  This prints a number of details about a dataset in a tabular format. [Detailed documentation](#preprocess_describe_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `Dimension` | [`int`](#doc_int) | Dimension of the data. Use this to specify a dimension | `0` |
| `input` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix containing data, | `**--**` |
| `Population` | [`bool`](#doc_bool) | If specified, the program will calculate statistics assuming the dataset is the population. By default, the program will assume the dataset as a sample. | `false` |
| `Precision` | [`int`](#doc_int) | Precision of the output statistics. | `4` |
| `RowMajor` | [`bool`](#doc_bool) | If specified, the program will calculate statistics across rows, not across columns.  (Remember that in mlpack, a column represents a point, so this option is generally not necessary.) | `false` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |
| `Width` | [`int`](#doc_int) | Width of the output table. | `8` |


### Detailed documentation
{: #preprocess_describe_detailed-documentation }

This utility takes a dataset and prints out the descriptive statistics of the data. Descriptive statistics is the discipline of quantitatively describing the main features of a collection of information, or the quantitative description itself. The program does not modify the original file, but instead prints out the statistics to the console. The printed result will look like a table.

Optionally, width and precision of the output can be adjusted by a user using the `Width` and `Precision` parameters. A user can also select a specific dimension to analyze if there are too many dimensions. The `Population` parameter can be specified when the dataset should be considered as a population.  Otherwise, the dataset will be considered as a sample.

### Example
So, a simple example where we want to print out statistical facts about the dataset `X` using the default settings, we could run 

```go
// Initialize optional parameters for PreprocessDescribe().
param := mlpack.PreprocessDescribeOptions()
param.Verbose = true

 := mlpack.PreprocessDescribe(X, param)
```

If we want to customize the width to 10 and precision to 5 and consider the dataset as a population, we could run

```go
// Initialize optional parameters for PreprocessDescribe().
param := mlpack.PreprocessDescribeOptions()
param.Width = 10
param.Precision = 5
param.Verbose = true

 := mlpack.PreprocessDescribe(X, param)
```

### See also

 - [PreprocessBinarize()](#preprocess_binarize)
 - [PreprocessSplit()](#preprocess_split)

## PreprocessScale()
{: #preprocess_scale }

#### Scale Data
{: #preprocess_scale_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for PreprocessScale().
param := mlpack.PreprocessScaleOptions()
param.Epsilon = 1e-06
param.InputModel = nil
param.InverseScaling = false
param.MaxValue = 1
param.MinValue = 0
param.ScalerMethod = "standard_scaler"
param.Seed = 0
param.Verbose = false

output, output_model := mlpack.PreprocessScale(input, param)
```

A utility to perform feature scaling on datasets using one of sixtechniques.  Both scaling and inverse scaling are supported, andscalers can be saved and then applied to other datasets. [Detailed documentation](#preprocess_scale_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `Epsilon` | [`float64`](#doc_float64) | regularization Parameter for pcawhitening, or zcawhitening, should be between -1 to 1. | `1e-06` |
| `input` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix containing data. | `**--**` |
| `InputModel` | [`scalingModel`](#doc_model) | Input Scaling model. | `nil` |
| `InverseScaling` | [`bool`](#doc_bool) | Inverse Scaling to get original dataset | `false` |
| `MaxValue` | [`int`](#doc_int) | Ending value of range for min_max_scaler. | `1` |
| `MinValue` | [`int`](#doc_int) | Starting value of range for min_max_scaler. | `0` |
| `ScalerMethod` | [`string`](#doc_string) | method to use for scaling, the default is standard_scaler. | `"standard_scaler"` |
| `Seed` | [`int`](#doc_int) | Random seed (0 for std::time(NULL)). | `0` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Output` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to save scaled data to. | 
| `OutputModel` | [`scalingModel`](#doc_model) | Output scaling model. | 

### Detailed documentation
{: #preprocess_scale_detailed-documentation }

This utility takes a dataset and performs feature scaling using one of the six scaler methods namely: 'max_abs_scaler', 'mean_normalization', 'min_max_scaler' ,'standard_scaler', 'pca_whitening' and 'zca_whitening'. The function takes a matrix as `Input` and a scaling method type which you can specify using `ScalerMethod` parameter; the default is standard scaler, and outputs a matrix with scaled feature.

The output scaled feature matrix may be saved with the `Output` output parameters.

The model to scale features can be saved using `OutputModel` and later can be loaded back using`InputModel`.

### Example
So, a simple example where we want to scale the dataset `X` into `X_scaled` with  standard_scaler as scaler_method, we could run 

```go
// Initialize optional parameters for PreprocessScale().
param := mlpack.PreprocessScaleOptions()
param.ScalerMethod = "standard_scaler"

X_scaled, _ := mlpack.PreprocessScale(X, param)
```

A simple example where we want to whiten the dataset `X` into `X_whitened` with  PCA as whitening_method and use 0.01 as regularization parameter, we could run 

```go
// Initialize optional parameters for PreprocessScale().
param := mlpack.PreprocessScaleOptions()
param.ScalerMethod = "pca_whitening"
param.Epsilon = 0.01

X_scaled, _ := mlpack.PreprocessScale(X, param)
```

You can also retransform the scaled dataset back using`InverseScaling`. An example to rescale : `X_scaled` into `X`using the saved model `InputModel` is:

```go
// Initialize optional parameters for PreprocessScale().
param := mlpack.PreprocessScaleOptions()
param.InverseScaling = true
param.InputModel = &saved

X, _ := mlpack.PreprocessScale(X_scaled, param)
```

Another simple example where we want to scale the dataset `X` into `X_scaled` with  min_max_scaler as scaler method, where scaling range is 1 to 3 instead of default 0 to 1. We could run 

```go
// Initialize optional parameters for PreprocessScale().
param := mlpack.PreprocessScaleOptions()
param.ScalerMethod = "min_max_scaler"
param.MinValue = 1
param.MaxValue = 3

X_scaled, _ := mlpack.PreprocessScale(X, param)
```

### See also

 - [PreprocessBinarize()](#preprocess_binarize)
 - [PreprocessDescribe()](#preprocess_describe)

## PreprocessOneHotEncoding()
{: #preprocess_one_hot_encoding }

#### One Hot Encoding
{: #preprocess_one_hot_encoding_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for PreprocessOneHotEncoding().
param := mlpack.PreprocessOneHotEncodingOptions()
param.Dimensions = []int{}
param.Verbose = false

output := mlpack.PreprocessOneHotEncoding(input, param)
```

A utility to do one-hot encoding on features of dataset. [Detailed documentation](#preprocess_one_hot_encoding_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `Dimensions` | [`array of ints`](#doc_array_of_ints) | Index of dimensions that need to be one-hot encoded (if unspecified, all categorical dimensions are one-hot encoded). | `[]int{}` |
| `input` | [`matrixWithInfo`](#doc_matrixWithInfo) | Matrix containing data. | `**--**` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Output` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to save one-hot encoded features data to. | 

### Detailed documentation
{: #preprocess_one_hot_encoding_detailed-documentation }

This utility takes a dataset and a vector of indices and does one-hot encoding of the respective features at those indices. Indices represent the IDs of the dimensions to be one-hot encoded.

If no dimensions are specified with `Dimensions`, then all categorical-type dimensions will be one-hot encoded. Otherwise, only the dimensions given in `Dimensions` will be one-hot encoded.

The output matrix with encoded features may be saved with the `Output` parameters.

### Example
So, a simple example where we want to encode 1st and 3rd feature from dataset `X` into `X_output` would be

```go
// Initialize optional parameters for PreprocessOneHotEncoding().
param := mlpack.PreprocessOneHotEncodingOptions()
param.Dimensions = 1
param.Dimensions = 3

X_ouput := mlpack.PreprocessOneHotEncoding(X, param)
```

### See also

 - [PreprocessBinarize()](#preprocess_binarize)
 - [PreprocessDescribe()](#preprocess_describe)
 - [One-hot encoding on Wikipedia](https://en.m.wikipedia.org/wiki/One-hot)

## Radical()
{: #radical }

#### RADICAL
{: #radical_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for Radical().
param := mlpack.RadicalOptions()
param.Angles = 150
param.NoiseStdDev = 0.175
param.Objective = false
param.Replicates = 30
param.Seed = 0
param.Sweeps = 0
param.Verbose = false

output_ic, output_unmixing := mlpack.Radical(input, param)
```

An implementation of RADICAL, a method for independent component analysis (ICA).  Given a dataset, this can decompose the dataset into an unmixing matrix and an independent component matrix; this can be useful for preprocessing. [Detailed documentation](#radical_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `Angles` | [`int`](#doc_int) | Number of angles to consider in brute-force search during Radical2D. | `150` |
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `input` | [`*mat.Dense`](#doc_a__mat_Dense) | Input dataset for ICA. | `**--**` |
| `NoiseStdDev` | [`float64`](#doc_float64) | Standard deviation of Gaussian noise. | `0.175` |
| `Objective` | [`bool`](#doc_bool) | If set, an estimate of the final objective function is printed. | `false` |
| `Replicates` | [`int`](#doc_int) | Number of Gaussian-perturbed replicates to use (per point) in Radical2D. | `30` |
| `Seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `Sweeps` | [`int`](#doc_int) | Number of sweeps; each sweep calls Radical2D once for each pair of dimensions. | `0` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `OutputIc` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to save independent components to. | 
| `OutputUnmixing` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to save unmixing matrix to. | 

### Detailed documentation
{: #radical_detailed-documentation }

An implementation of RADICAL, a method for independent component analysis (ICA).  Assuming that we have an input matrix X, the goal is to find a square unmixing matrix W such that Y = W * X and the dimensions of Y are independent components.  If the algorithm is running particularly slowly, try reducing the number of replicates.

The input matrix to perform ICA on should be specified with the `Input` parameter.  The output matrix Y may be saved with the `OutputIc` output parameter, and the output unmixing matrix W may be saved with the `OutputUnmixing` output parameter.

### Example
For example, to perform ICA on the matrix `X` with 40 replicates, saving the independent components to `ic`, the following command may be used: 

```go
// Initialize optional parameters for Radical().
param := mlpack.RadicalOptions()
param.Replicates = 40

ic, _ := mlpack.Radical(X, param)
```

### See also

 - [Independent component analysis on Wikipedia](https://en.wikipedia.org/wiki/Independent_component_analysis)
 - [ICA using spacings estimates of entropy (pdf)](https://www.jmlr.org/papers/volume4/learned-miller03a/learned-miller03a.pdf)
 - [Radical C++ class documentation](../../user/methods/radical.md)

## RandomForest()
{: #random_forest }

#### Random forests
{: #random_forest_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for RandomForest().
param := mlpack.RandomForestOptions()
param.InputModel = nil
param.Labels = mat.NewDense(1, 1, nil)
param.MaximumDepth = 0
param.MinimumGainSplit = 0
param.MinimumLeafSize = 1
param.NumTrees = 10
param.PrintTrainingAccuracy = false
param.Seed = 0
param.SubspaceDim = 0
param.Test = mat.NewDense(1, 1, nil)
param.TestLabels = mat.NewDense(1, 1, nil)
param.Training = mat.NewDense(1, 1, nil)
param.Verbose = false
param.WarmStart = false

output_model, predictions, probabilities := mlpack.RandomForest(param)
```

An implementation of the standard random forest algorithm by Leo Breiman for classification.  Given labeled data, a random forest can be trained and saved for future use; or, a pre-trained random forest can be used for classification. [Detailed documentation](#random_forest_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `InputModel` | [`randomForestModel`](#doc_model) | Pre-trained random forest to use for classification. | `nil` |
| `Labels` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | Labels for training dataset. | `mat.NewDense(1, 1, nil)` |
| `MaximumDepth` | [`int`](#doc_int) | Maximum depth of the tree (0 means no limit). | `0` |
| `MinimumGainSplit` | [`float64`](#doc_float64) | Minimum gain needed to make a split when building a tree. | `0` |
| `MinimumLeafSize` | [`int`](#doc_int) | Minimum number of points in each leaf node. | `1` |
| `NumTrees` | [`int`](#doc_int) | Number of trees in the random forest. | `10` |
| `PrintTrainingAccuracy` | [`bool`](#doc_bool) | If set, then the accuracy of the model on the training set will be predicted (verbose must also be specified). | `false` |
| `Seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `SubspaceDim` | [`int`](#doc_int) | Dimensionality of random subspace to use for each split.  '0' will autoselect the square root of data dimensionality. | `0` |
| `Test` | [`*mat.Dense`](#doc_a__mat_Dense) | Test dataset to produce predictions for. | `mat.NewDense(1, 1, nil)` |
| `TestLabels` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | Test dataset labels, if accuracy calculation is desired. | `mat.NewDense(1, 1, nil)` |
| `Training` | [`*mat.Dense`](#doc_a__mat_Dense) | Training dataset. | `mat.NewDense(1, 1, nil)` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |
| `WarmStart` | [`bool`](#doc_bool) | If true and passed along with `training` and `input_model` then trains more trees on top of existing model. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `OutputModel` | [`randomForestModel`](#doc_model) | Model to save trained random forest to. | 
| `Predictions` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | Predicted classes for each point in the test set. | 
| `Probabilities` | [`*mat.Dense`](#doc_a__mat_Dense) | Predicted class probabilities for each point in the test set. | 

### Detailed documentation
{: #random_forest_detailed-documentation }

This program is an implementation of the standard random forest classification algorithm by Leo Breiman.  A random forest can be trained and saved for later use, or a random forest may be loaded and predictions or class probabilities for points may be generated.

The training set and associated labels are specified with the `Training` and `Labels` parameters, respectively.  The labels should be in the range `[0, num_classes - 1]`. Optionally, if `Labels` is not specified, the labels are assumed to be the last dimension of the training dataset.

When a model is trained, the `OutputModel` output parameter may be used to save the trained model.  A model may be loaded for predictions with the `InputModel`parameter. The `InputModel` parameter may not be specified when the `Training` parameter is specified.  The `MinimumLeafSize` parameter specifies the minimum number of training points that must fall into each leaf for it to be split.  The `NumTrees` controls the number of trees in the random forest.  The `MinimumGainSplit` parameter controls the minimum required gain for a decision tree node to split.  Larger values will force higher-confidence splits.  The `MaximumDepth` parameter specifies the maximum depth of the tree.  The `SubspaceDim` parameter is used to control the number of random dimensions chosen for an individual node's split.  If `PrintTrainingAccuracy` is specified, the calculated accuracy on the training set will be printed.

Test data may be specified with the `Test` parameter, and if performance measures are desired for that test set, labels for the test points may be specified with the `TestLabels` parameter.  Predictions for each test point may be saved via the `Predictions`output parameter.  Class probabilities for each prediction may be saved with the `Probabilities` output parameter.

### Example
For example, to train a random forest with a minimum leaf size of 20 using 10 trees on the dataset contained in `data`with labels `labels`, saving the output random forest to `rf_model` and printing the training error, one could call

```go
// Initialize optional parameters for RandomForest().
param := mlpack.RandomForestOptions()
param.Training = data
param.Labels = labels
param.MinimumLeafSize = 20
param.NumTrees = 10
param.PrintTrainingAccuracy = true

rf_model, _, _ := mlpack.RandomForest(param)
```

Then, to use that model to classify points in `test_set` and print the test error given the labels `test_labels` using that model, while saving the predictions for each point to `predictions`, one could call 

```go
// Initialize optional parameters for RandomForest().
param := mlpack.RandomForestOptions()
param.InputModel = &rf_model
param.Test = test_set
param.TestLabels = test_labels

_, predictions, _ := mlpack.RandomForest(param)
```

### See also

 - [DecisionTree()](#decision_tree)
 - [HoeffdingTree()](#hoeffding_tree)
 - [SoftmaxRegression()](#softmax_regression)
 - [Random forest on Wikipedia](https://en.wikipedia.org/wiki/Random_forest)
 - [Random forests (pdf)](https://www.eecis.udel.edu/~shatkay/Course/papers/BreimanRandomForests2001.pdf)
 - [RandomForest C++ class documentation](../../user/methods/random_forest.md)

## Krann()
{: #krann }

#### K-Rank-Approximate-Nearest-Neighbors (kRANN)
{: #krann_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for Krann().
param := mlpack.KrannOptions()
param.Alpha = 0.95
param.FirstLeafExact = false
param.InputModel = nil
param.K = 0
param.LeafSize = 20
param.Naive = false
param.Query = mat.NewDense(1, 1, nil)
param.RandomBasis = false
param.Reference = mat.NewDense(1, 1, nil)
param.SampleAtLeaves = false
param.Seed = 0
param.SingleMode = false
param.SingleSampleLimit = 20
param.Tau = 5
param.TreeType = "kd"
param.Verbose = false

distances, neighbors, output_model := mlpack.Krann(param)
```

An implementation of rank-approximate k-nearest-neighbor search (kRANN)  using single-tree and dual-tree algorithms.  Given a set of reference points and query points, this can find the k nearest neighbors in the reference set of each query point using trees; trees that are built can be saved for future use. [Detailed documentation](#krann_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `Alpha` | [`float64`](#doc_float64) | The desired success probability. | `0.95` |
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `FirstLeafExact` | [`bool`](#doc_bool) | The flag to trigger sampling only after exactly exploring the first leaf. | `false` |
| `InputModel` | [`raModel`](#doc_model) | Pre-trained kNN model. | `nil` |
| `K` | [`int`](#doc_int) | Number of nearest neighbors to find. | `0` |
| `LeafSize` | [`int`](#doc_int) | Leaf size for tree building (used for kd-trees, UB trees, R trees, R* trees, X trees, Hilbert R trees, R+ trees, R++ trees, and octrees). | `20` |
| `Naive` | [`bool`](#doc_bool) | If true, sampling will be done without using a tree. | `false` |
| `Query` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix containing query points (optional). | `mat.NewDense(1, 1, nil)` |
| `RandomBasis` | [`bool`](#doc_bool) | Before tree-building, project the data onto a random orthogonal basis. | `false` |
| `Reference` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix containing the reference dataset. | `mat.NewDense(1, 1, nil)` |
| `SampleAtLeaves` | [`bool`](#doc_bool) | The flag to trigger sampling at leaves. | `false` |
| `Seed` | [`int`](#doc_int) | Random seed (if 0, std::time(NULL) is used). | `0` |
| `SingleMode` | [`bool`](#doc_bool) | If true, single-tree search is used (as opposed to dual-tree search. | `false` |
| `SingleSampleLimit` | [`int`](#doc_int) | The limit on the maximum number of samples (and hence the largest node you can approximate). | `20` |
| `Tau` | [`float64`](#doc_float64) | The allowed rank-error in terms of the percentile of the data. | `5` |
| `TreeType` | [`string`](#doc_string) | Type of tree to use: 'kd', 'ub', 'cover', 'r', 'x', 'r-star', 'hilbert-r', 'r-plus', 'r-plus-plus', 'oct'. | `"kd"` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Distances` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to output distances into. | 
| `Neighbors` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to output neighbors into. | 
| `OutputModel` | [`raModel`](#doc_model) | If specified, the kNN model will be output here. | 

### Detailed documentation
{: #krann_detailed-documentation }

This program will calculate the k rank-approximate-nearest-neighbors of a set of points. You may specify a separate set of reference points and query points, or just a reference set which will be used as both the reference and query set. You must specify the rank approximation (in %) (and optionally the success probability).

### Example
For example, the following will return 5 neighbors from the top 0.1% of the data (with probability 0.95) for each point in `input` and store the distances in `distances` and the neighbors in `neighbors.csv`:

```go
// Initialize optional parameters for Krann().
param := mlpack.KrannOptions()
param.Reference = input
param.K = 5
param.Tau = 0.1

distances, neighbors, _ := mlpack.Krann(param)
```

Note that tau must be set such that the number of points in the corresponding percentile of the data is greater than k.  Thus, if we choose tau = 0.1 with a dataset of 1000 points and k = 5, then we are attempting to choose 5 nearest neighbors out of the closest 1 point -- this is invalid and the program will terminate with an error message.

The output matrices are organized such that row i and column j in the neighbors output file corresponds to the index of the point in the reference set which is the i'th nearest neighbor from the point in the query set with index j.  Row i and column j in the distances output file corresponds to the distance between those two points.

### See also

 - [Knn()](#knn)
 - [Lsh()](#lsh)
 - [Rank-approximate nearest neighbor search: Retaining meaning and speed in high dimensions (pdf)](https://proceedings.neurips.cc/paper_files/paper/2009/file/ddb30680a691d157187ee1cf9e896d03-Paper.pdf)
 - [RASearch C++ class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/rann/ra_search.hpp)

## SoftmaxRegression()
{: #softmax_regression }

#### Softmax Regression
{: #softmax_regression_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for SoftmaxRegression().
param := mlpack.SoftmaxRegressionOptions()
param.InputModel = nil
param.Labels = mat.NewDense(1, 1, nil)
param.Lambda = 0.0001
param.MaxIterations = 400
param.NoIntercept = false
param.NumberOfClasses = 0
param.Test = mat.NewDense(1, 1, nil)
param.TestLabels = mat.NewDense(1, 1, nil)
param.Training = mat.NewDense(1, 1, nil)
param.Verbose = false

output_model, predictions, probabilities := mlpack.SoftmaxRegression(param)
```

An implementation of softmax regression for classification, which is a multiclass generalization of logistic regression.  Given labeled data, a softmax regression model can be trained and saved for future use, or, a pre-trained softmax regression model can be used for classification of new points. [Detailed documentation](#softmax_regression_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `InputModel` | [`softmaxRegression`](#doc_model) | File containing existing model (parameters). | `nil` |
| `Labels` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | A matrix containing labels (0 or 1) for the points in the training set (y). The labels must order as a row. | `mat.NewDense(1, 1, nil)` |
| `Lambda` | [`float64`](#doc_float64) | L2-regularization constant | `0.0001` |
| `MaxIterations` | [`int`](#doc_int) | Maximum number of iterations before termination. | `400` |
| `NoIntercept` | [`bool`](#doc_bool) | Do not add the intercept term to the model. | `false` |
| `NumberOfClasses` | [`int`](#doc_int) | Number of classes for classification; if unspecified (or 0), the number of classes found in the labels will be used. | `0` |
| `Test` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix containing test dataset. | `mat.NewDense(1, 1, nil)` |
| `TestLabels` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | Matrix containing test labels. | `mat.NewDense(1, 1, nil)` |
| `Training` | [`*mat.Dense`](#doc_a__mat_Dense) | A matrix containing the training set (the matrix of predictors, X). | `mat.NewDense(1, 1, nil)` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `OutputModel` | [`softmaxRegression`](#doc_model) | File to save trained softmax regression model to. | 
| `Predictions` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | Matrix to save predictions for test dataset into. | 
| `Probabilities` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to save class probabilities for test dataset into. | 

### Detailed documentation
{: #softmax_regression_detailed-documentation }

This program performs softmax regression, a generalization of logistic regression to the multiclass case, and has support for L2 regularization.  The program is able to train a model, load  an existing model, and give predictions (and optionally their accuracy) for test data.

Training a softmax regression model is done by giving a file of training points with the `Training` parameter and their corresponding labels with the `Labels` parameter. The number of classes can be manually specified with the `NumberOfClasses` parameter, and the maximum number of iterations of the L-BFGS optimizer can be specified with the `MaxIterations` parameter.  The L2 regularization constant can be specified with the `Lambda` parameter and if an intercept term is not desired in the model, the `NoIntercept` parameter can be specified.

The trained model can be saved with the `OutputModel` output parameter. If training is not desired, but only testing is, a model can be loaded with the `InputModel` parameter.  At the current time, a loaded model cannot be trained further, so specifying both `InputModel` and `Training` is not allowed.

The program is also able to evaluate a model on test data.  A test dataset can be specified with the `Test` parameter. Class predictions can be saved with the `Predictions` output parameter.  If labels are specified for the test data with the `TestLabels` parameter, then the program will print the accuracy of the predictions on the given test set and its corresponding labels.

### Example
For example, to train a softmax regression model on the data `dataset` with labels `labels` with a maximum of 1000 iterations for training, saving the trained model to `sr_model`, the following command can be used: 

```go
// Initialize optional parameters for SoftmaxRegression().
param := mlpack.SoftmaxRegressionOptions()
param.Training = dataset
param.Labels = labels

sr_model, _, _ := mlpack.SoftmaxRegression(param)
```

Then, to use `sr_model` to classify the test points in `test_points`, saving the output predictions to `predictions`, the following command can be used:

```go
// Initialize optional parameters for SoftmaxRegression().
param := mlpack.SoftmaxRegressionOptions()
param.InputModel = &sr_model
param.Test = test_points

_, predictions, _ := mlpack.SoftmaxRegression(param)
```

### See also

 - [LogisticRegression()](#logistic_regression)
 - [RandomForest()](#random_forest)
 - [Multinomial logistic regression (softmax regression) on Wikipedia](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)
 - [SoftmaxRegression C++ class documentation](../../user/methods/softmax_regression.md)

## SparseCoding()
{: #sparse_coding }

#### Sparse Coding
{: #sparse_coding_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for SparseCoding().
param := mlpack.SparseCodingOptions()
param.Atoms = 15
param.InitialDictionary = mat.NewDense(1, 1, nil)
param.InputModel = nil
param.Lambda1 = 0
param.Lambda2 = 0
param.MaxIterations = 0
param.NewtonTolerance = 1e-06
param.Normalize = false
param.ObjectiveTolerance = 0.01
param.Seed = 0
param.Test = mat.NewDense(1, 1, nil)
param.Training = mat.NewDense(1, 1, nil)
param.Verbose = false

codes, dictionary, output_model := mlpack.SparseCoding(param)
```

An implementation of Sparse Coding with Dictionary Learning.  Given a dataset, this will decompose the dataset into a sparse combination of a few dictionary elements, where the dictionary is learned during computation; a dictionary can be reused for future sparse coding of new points. [Detailed documentation](#sparse_coding_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `Atoms` | [`int`](#doc_int) | Number of atoms in the dictionary. | `15` |
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `InitialDictionary` | [`*mat.Dense`](#doc_a__mat_Dense) | Optional initial dictionary matrix. | `mat.NewDense(1, 1, nil)` |
| `InputModel` | [`sparseCoding`](#doc_model) | File containing input sparse coding model. | `nil` |
| `Lambda1` | [`float64`](#doc_float64) | Sparse coding l1-norm regularization parameter. | `0` |
| `Lambda2` | [`float64`](#doc_float64) | Sparse coding l2-norm regularization parameter. | `0` |
| `MaxIterations` | [`int`](#doc_int) | Maximum number of iterations for sparse coding (0 indicates no limit). | `0` |
| `NewtonTolerance` | [`float64`](#doc_float64) | Tolerance for convergence of Newton method. | `1e-06` |
| `Normalize` | [`bool`](#doc_bool) | If set, the input data matrix will be normalized before coding. | `false` |
| `ObjectiveTolerance` | [`float64`](#doc_float64) | Tolerance for convergence of the objective function. | `0.01` |
| `Seed` | [`int`](#doc_int) | Random seed.  If 0, 'std::time(NULL)' is used. | `0` |
| `Test` | [`*mat.Dense`](#doc_a__mat_Dense) | Optional matrix to be encoded by trained model. | `mat.NewDense(1, 1, nil)` |
| `Training` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix of training data (X). | `mat.NewDense(1, 1, nil)` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Codes` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to save the output sparse codes of the test matrix (--test_file) to. | 
| `Dictionary` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to save the output dictionary to. | 
| `OutputModel` | [`sparseCoding`](#doc_model) | File to save trained sparse coding model to. | 

### Detailed documentation
{: #sparse_coding_detailed-documentation }

An implementation of Sparse Coding with Dictionary Learning, which achieves sparsity via an l1-norm regularizer on the codes (LASSO) or an (l1+l2)-norm regularizer on the codes (the Elastic Net).  Given a dense data matrix X with d dimensions and n points, sparse coding seeks to find a dense dictionary matrix D with k atoms in d dimensions, and a sparse coding matrix Z with n points in k dimensions.

The original data matrix X can then be reconstructed as Z * D.  Therefore, this program finds a representation of each point in X as a sparse linear combination of atoms in the dictionary D.

The sparse coding is found with an algorithm which alternates between a dictionary step, which updates the dictionary D, and a sparse coding step, which updates the sparse coding matrix.

Once a dictionary D is found, the sparse coding model may be used to encode other matrices, and saved for future usage.

To run this program, either an input matrix or an already-saved sparse coding model must be specified.  An input matrix may be specified with the `Training` option, along with the number of atoms in the dictionary (specified with the `Atoms` parameter).  It is also possible to specify an initial dictionary for the optimization, with the `InitialDictionary` parameter.  An input model may be specified with the `InputModel` parameter.

### Example
As an example, to build a sparse coding model on the dataset `data` using 200 atoms and an l1-regularization parameter of 0.1, saving the model into `model`, use 

```go
// Initialize optional parameters for SparseCoding().
param := mlpack.SparseCodingOptions()
param.Training = data
param.Atoms = 200
param.Lambda1 = 0.1

_, _, model := mlpack.SparseCoding(param)
```

Then, this model could be used to encode a new matrix, `otherdata`, and save the output codes to `codes`: 

```go
// Initialize optional parameters for SparseCoding().
param := mlpack.SparseCodingOptions()
param.InputModel = &model
param.Test = otherdata

codes, _, _ := mlpack.SparseCoding(param)
```

### See also

 - [LocalCoordinateCoding()](#local_coordinate_coding)
 - [Sparse dictionary learning on Wikipedia](https://en.wikipedia.org/wiki/Sparse_dictionary_learning)
 - [Efficient sparse coding algorithms (pdf)](https://proceedings.neurips.cc/paper_files/paper/2006/file/2d71b2ae158c7c5912cc0bbde2bb9d95-Paper.pdf)
 - [Regularization and variable selection via the elastic net](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=46217f372a75dddc2254fdbc6b9418ba3554e453)
 - [SparseCoding C++ class documentation](../../user/methods/sparse_coding.md)

## Adaboost()
{: #adaboost }

#### AdaBoost
{: #adaboost_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for Adaboost().
param := mlpack.AdaboostOptions()
param.InputModel = nil
param.Iterations = 1000
param.Labels = mat.NewDense(1, 1, nil)
param.Test = mat.NewDense(1, 1, nil)
param.Tolerance = 1e-10
param.Training = mat.NewDense(1, 1, nil)
param.Verbose = false
param.WeakLearner = "decision_stump"

output_model, predictions, probabilities := mlpack.Adaboost(param)
```

An implementation of the AdaBoost.MH (Adaptive Boosting) algorithm for classification.  This can be used to train an AdaBoost model on labeled data or use an existing AdaBoost model to predict the classes of new points. [Detailed documentation](#adaboost_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `InputModel` | [`adaBoostModel`](#doc_model) | Input AdaBoost model. | `nil` |
| `Iterations` | [`int`](#doc_int) | The maximum number of boosting iterations to be run (0 will run until convergence.) | `1000` |
| `Labels` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | Labels for the training set. | `mat.NewDense(1, 1, nil)` |
| `Test` | [`*mat.Dense`](#doc_a__mat_Dense) | Test dataset. | `mat.NewDense(1, 1, nil)` |
| `Tolerance` | [`float64`](#doc_float64) | The tolerance for change in values of the weighted error during training. | `1e-10` |
| `Training` | [`*mat.Dense`](#doc_a__mat_Dense) | Dataset for training AdaBoost. | `mat.NewDense(1, 1, nil)` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |
| `WeakLearner` | [`string`](#doc_string) | The type of weak learner to use: 'decision_stump', or 'perceptron'. | `"decision_stump"` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `OutputModel` | [`adaBoostModel`](#doc_model) | Output trained AdaBoost model. | 
| `Predictions` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | Predicted labels for the test set. | 
| `Probabilities` | [`*mat.Dense`](#doc_a__mat_Dense) | Predicted class probabilities for each point in the test set. | 

### Detailed documentation
{: #adaboost_detailed-documentation }

This program implements the AdaBoost (or Adaptive Boosting) algorithm. The variant of AdaBoost implemented here is AdaBoost.MH. It uses a weak learner, either decision stumps or perceptrons, and over many iterations, creates a strong learner that is a weighted ensemble of weak learners. It runs these iterations until a tolerance value is crossed for change in the value of the weighted training error.

For more information about the algorithm, see the paper "Improved Boosting Algorithms Using Confidence-Rated Predictions", by R.E. Schapire and Y. Singer.

This program allows training of an AdaBoost model, and then application of that model to a test dataset.  To train a model, a dataset must be passed with the `Training` option.  Labels can be given with the `Labels` option; if no labels are specified, the labels will be assumed to be the last column of the input dataset.  Alternately, an AdaBoost model may be loaded with the `InputModel` option.

Once a model is trained or loaded, it may be used to provide class predictions for a given test dataset.  A test dataset may be specified with the `Test` parameter.  The predicted classes for each point in the test dataset are output to the `Predictions` output parameter.  The AdaBoost model itself is output to the `OutputModel` output parameter.

### Example
For example, to run AdaBoost on an input dataset `data` with labels `labels`and perceptrons as the weak learner type, storing the trained model in `model`, one could use the following command: 

```go
// Initialize optional parameters for Adaboost().
param := mlpack.AdaboostOptions()
param.Training = data
param.Labels = labels
param.WeakLearner = "perceptron"

model, _, _ := mlpack.Adaboost(param)
```

Similarly, an already-trained model in `model` can be used to provide class predictions from test data `test_data` and store the output in `predictions` with the following command: 

```go
// Initialize optional parameters for Adaboost().
param := mlpack.AdaboostOptions()
param.InputModel = &model
param.Test = test_data

_, predictions, _ := mlpack.Adaboost(param)
```

### See also

 - [AdaBoost on Wikipedia](https://en.wikipedia.org/wiki/AdaBoost)
 - [Improved boosting algorithms using confidence-rated predictions (pdf)](http://rob.schapire.net/papers/SchapireSi98.pdf)
 - [Perceptron](#perceptron)
 - [Decision Trees](#decision_tree)
 - [AdaBoost C++ class documentation](../../user/methods/adaboost.md)

## LinearRegression()
{: #linear_regression }

#### Simple Linear Regression and Prediction
{: #linear_regression_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for LinearRegression().
param := mlpack.LinearRegressionOptions()
param.InputModel = nil
param.Lambda = 0
param.Test = mat.NewDense(1, 1, nil)
param.Training = mat.NewDense(1, 1, nil)
param.TrainingResponses = mat.NewDense(1, 1, nil)
param.Verbose = false

output_model, output_predictions := mlpack.LinearRegression(param)
```

An implementation of simple linear regression and ridge regression using ordinary least squares.  Given a dataset and responses, a model can be trained and saved for later use, or a pre-trained model can be used to output regression predictions for a test set. [Detailed documentation](#linear_regression_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `InputModel` | [`linearRegression`](#doc_model) | Existing LinearRegression model to use. | `nil` |
| `Lambda` | [`float64`](#doc_float64) | Tikhonov regularization for ridge regression.  If 0, the method reduces to linear regression. | `0` |
| `Test` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix containing X' (test regressors). | `mat.NewDense(1, 1, nil)` |
| `Training` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix containing training set X (regressors). | `mat.NewDense(1, 1, nil)` |
| `TrainingResponses` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | Optional vector containing y (responses). If not given, the responses are assumed to be the last row of the input file. | `mat.NewDense(1, 1, nil)` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `OutputModel` | [`linearRegression`](#doc_model) | Output LinearRegression model. | 
| `OutputPredictions` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | If --test_file is specified, this matrix is where the predicted responses will be saved. | 

### Detailed documentation
{: #linear_regression_detailed-documentation }

An implementation of simple linear regression and simple ridge regression using ordinary least squares. This solves the problem

  y = X * b + e

where X (specified by `Training`) and y (specified either as the last column of the input matrix `Training` or via the `TrainingResponses` parameter) are known and b is the desired variable.  If the covariance matrix (X'X) is not invertible, or if the solution is overdetermined, then specify a Tikhonov regularization constant (with `Lambda`) greater than 0, which will regularize the covariance matrix to make it invertible.  The calculated b may be saved with the `OutputPredictions` output parameter.

Optionally, the calculated value of b is used to predict the responses for another matrix X' (specified by the `Test` parameter):

   y' = X' * b

and the predicted responses y' may be saved with the `OutputPredictions` output parameter.  This type of regression is related to least-angle regression, which mlpack implements as the 'lars' program.

### Example
For example, to run a linear regression on the dataset `X` with responses `y`, saving the trained model to `lr_model`, the following command could be used:

```go
// Initialize optional parameters for LinearRegression().
param := mlpack.LinearRegressionOptions()
param.Training = X
param.TrainingResponses = y

lr_model, _ := mlpack.LinearRegression(param)
```

Then, to use `lr_model` to predict responses for a test set `X_test`, saving the predictions to `X_test_responses`, the following command could be used:

```go
// Initialize optional parameters for LinearRegression().
param := mlpack.LinearRegressionOptions()
param.InputModel = &lr_model
param.Test = X_test

_, X_test_responses := mlpack.LinearRegression(param)
```

### See also

 - [Lars()](#lars)
 - [Linear regression on Wikipedia](https://en.wikipedia.org/wiki/Linear_regression)
 - [LinearRegression C++ class documentation](../../user/methods/linear_regression.md)

## ImageConverter()
{: #image_converter }

#### Image Converter
{: #image_converter_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for ImageConverter().
param := mlpack.ImageConverterOptions()
param.Channels = 0
param.Dataset = mat.NewDense(1, 1, nil)
param.Height = 0
param.Quality = 90
param.Save = false
param.Verbose = false
param.Width = 0

output := mlpack.ImageConverter(input, param)
```

A utility to load an image or set of images into a single dataset that can then be used by other mlpack methods and utilities. This can also unpack an image dataset into individual files, for instance after mlpack methods have been used. [Detailed documentation](#image_converter_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `Channels` | [`int`](#doc_int) | Number of channels in the image. | `0` |
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `Dataset` | [`*mat.Dense`](#doc_a__mat_Dense) | Input matrix to save as images. | `mat.NewDense(1, 1, nil)` |
| `Height` | [`int`](#doc_int) | Height of the images. | `0` |
| `input` | [`array of strings`](#doc_array_of_strings) | Image filenames which have to be loaded/saved. | `**--**` |
| `Quality` | [`int`](#doc_int) | Compression of the image if saved as jpg (0-100). | `90` |
| `Save` | [`bool`](#doc_bool) | Save a dataset as images. | `false` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |
| `Width` | [`int`](#doc_int) | Width of the image. | `0` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Output` | [`*mat.Dense`](#doc_a__mat_Dense) | Matrix to save images data to, Onlyneeded if you are specifying 'save' option. | 

### Detailed documentation
{: #image_converter_detailed-documentation }

This utility takes an image or an array of images and loads them to a matrix. You can optionally specify the height `Height` width `Width` and channel `Channels` of the images that needs to be loaded; otherwise, these parameters will be automatically detected from the image.
There are other options too, that can be specified such as `Quality`.

You can also provide a dataset and save them as images using `Dataset` and `Save` as an parameter.

### Example
 An example to load an image : 

```go
// Initialize optional parameters for ImageConverter().
param := mlpack.ImageConverterOptions()
param.Height = 256
param.Width = 256
param.Channels = 3

Y := mlpack.ImageConverter(X, param)
```

 An example to save an image is :

```go
// Initialize optional parameters for ImageConverter().
param := mlpack.ImageConverterOptions()
param.Height = 256
param.Width = 256
param.Channels = 3
param.Dataset = Y
param.Save = true

_ := mlpack.ImageConverter(X, param)
```

### See also

 - [PreprocessBinarize()](#preprocess_binarize)
 - [PreprocessDescribe()](#preprocess_describe)

