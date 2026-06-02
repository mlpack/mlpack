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

## DecisionTreeClassify()
{: #decision_tree_classify }

#### Decision tree Prediction
{: #decision_tree_classify_descr }

```go
import (
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
)

// Initialize optional parameters for DecisionTreeClassify().
param := mlpack.DecisionTreeClassifyOptions()
param.TestLabels = mat.NewDense(1, 1, nil)
param.Verbose = false

predictions := mlpack.DecisionTreeClassify(inputModel, test, param)
```

Class predictions from train decision tree model. [Detailed documentation](#decision_tree_classify_detailed-documentation).



### Input options
There are two types of input options: required options, which are passed directly to the function call, and optional options, which are passed via an initialized struct, which allows keyword access to each of the options.

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `CheckInputMatrices` | [`bool`](#doc_bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `inputModel` | [`decisionTreeModel`](#doc_model) | Pre-trained decision tree, to be used with test points. | `**--**` |
| `test` | [`matrixWithInfo`](#doc_matrixWithInfo) | Testing dataset (may contain categorical variables). | `**--**` |
| `TestLabels` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | Test point labels, if accuracy calculation is desired. | `mat.NewDense(1, 1, nil)` |
| `Verbose` | [`bool`](#doc_bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Output options are returned via Go's support for multiple return values, in the order listed below.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `Predictions` | [`*mat.Dense (1d)`](#doc_a__mat_Dense__1d_) | Class predictions for each test point. | 

### Detailed documentation
{: #decision_tree_classify_detailed-documentation }



### Example
