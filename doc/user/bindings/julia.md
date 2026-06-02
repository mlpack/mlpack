# mlpack Julia binding documentation

## mlpack overview

mlpack is an intuitive, fast, and flexible header-only C++ machine learning library with bindings to other languages.  It aims to provide fast, lightweight implementations of both common and cutting-edge machine learning algorithms.

This reference page details mlpack's bindings to Julia.

Further useful mlpack documentation links are given below.

 - [mlpack homepage](https://www.mlpack.org/)
 - [mlpack on Github](https://github.com/mlpack/mlpack)
 - [mlpack main documentation page](https://www.mlpack.org/doc/index.html)

See also the quickstart guide for Julia:

 - [Julia Quickstart](../../quickstart/julia.md)

## Data Formats

<div id="data-formats-div" markdown="1">
mlpack bindings for Julia take and return a restricted set of types, for simplicity.  These include primitive types, matrix/vector types, categorical matrix types, and model types. Each type is detailed below.

 - `Int`{: #doc_Int}: An integer (i.e., `1`).
 - `Float64`{: #doc_Float64 }: A floating-point number (i.e., `0.5`).
 - `Bool`{: #doc_Bool }: A boolean flag option (`true` or `false`).
 - `String`{: #doc_String }: A character string (i.e., `"hello"`).
 - `Array{Int, 1}`{: #doc_Array_Int__1_ }: A vector of integers; i.e., `[0, 1, 2]`.
 - `Array{String, 1}`{: #doc_Array_String__1_ }: A vector of strings; i.e., `["hello", "goodbye"]`.
 - `Float64 matrix-like`{: #doc_Float64_matrix_like }: A 2-d matrix-like containing `Float64` data (could be an `Array{Float64, 2}` or a `DataFrame` or anything convertible to an `Array{Float64, 2}`).  It is expected that each row of the matrix corresponds to a data point, unless `points_are_rows` is set to `false` when calling mlpack bindings.
 - `Int matrix-like`{: #doc_Int_matrix_like }: A 2-d matrix-like containing `Int` data (elements should be greater than or equal to 0).  Could be an `Array{Int, 2}` or a `DataFrame` or anything convertible to an `Array{Int, 2}`.  It is expected that each row of the matrix corresponds to a data point, unless `points_are_rows` is set to `false` when calling mlpack bindings.
 - `Float64 vector-like`{: #doc_Float64_vector_like }: A 1-d vector-like containing `Float64` data (could be an `Array{Float64, 1}`, an `Array{Float64, 2}` with one dimension of size 1, or anything convertible to `Array{Float64, 1}`.
 - `Int vector-like`{: #doc_Int_vector_like }: A 1-d vector-like containing `Int` data (elements should be greater than or equal to 0).  Could be an `Array{Int, 1}`, an `Array{Int, 2}` with one dimension of size 1, or anything convertible to `Array{Int, 1}`.
 - `Tuple{Array{Bool, 1}, Array{Float64, 2}}`{: #doc_Tuple_Array_Bool__1___Array_Float64__2__ }: A 2-d array containing `Float64` data along with a boolean array indicating which dimensions are categorical (represented by `true`) and which are numeric (represented by `false`).  The number of elements in the boolean array should be the same as the dimensionality of the data matrix.  Categorical dimensions should take integer values between 1 and the number of categories.  It is expected that each row of the matrix corresponds to a single data point, unless `points_are_rows` is set to `false` when calling mlpack bindings.
 - `<Model> (mlpack model)`{: #doc_model }: An mlpack model pointer.  `<Model>` refers to the type of model that is being stored, so, e.g., for `CF()`, the type will be `CFModel`. This type holds a pointer to C++ memory containing the mlpack model.  Note that this means the mlpack model itself cannot be easily inspected in Julia.  However, the pointer can be passed to subsequent calls to mlpack functions, and can be serialized and deserialized via either the `Serialization` package, or the `mlpack.serialize_bin()` and `mlpack.deserialize_bin()` functions.
</div>


## approx_kfn()
{: #approx_kfn }

#### Approximate furthest neighbor search
{: #approx_kfn_descr }

```julia
julia> using mlpack: approx_kfn
julia> distances, neighbors, output_model = approx_kfn( ;
          algorithm="ds", calculate_error=false, exact_distances=zeros(0, 0),
          input_model=nothing, k=0, num_projections=5, num_tables=5,
          query=zeros(0, 0), reference=zeros(0, 0), verbose=false)
```

An implementation of two strategies for furthest neighbor search.  This can be used to compute the furthest neighbor of query point(s) from a set of points; furthest neighbor models can be saved and reused with future query point(s). [Detailed documentation](#approx_kfn_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `algorithm` | [`String`](#doc_String) | Algorithm to use: 'ds' or 'qdafn'. | `"ds"` |
| `calculate_error` | [`Bool`](#doc_Bool) | If set, calculate the average distance error for the first furthest neighbor only. | `false` |
| `check_input_matrices` | [`Bool`](#doc_Bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `exact_distances` | [`Float64 matrix-like`](#doc_Float64_matrix_like) | Matrix containing exact distances to furthest neighbors; this can be used to avoid explicit calculation when --calculate_error is set. | `zeros(0, 0)` |
| `input_model` | [`ApproxKFNModel`](#doc_model) | File containing input model. | `nothing` |
| `k` | [`Int`](#doc_Int) | Number of furthest neighbors to search for. | `0` |
| `num_projections` | [`Int`](#doc_Int) | Number of projections to use in each hash table. | `5` |
| `num_tables` | [`Int`](#doc_Int) | Number of hash tables to use. | `5` |
| `query` | [`Float64 matrix-like`](#doc_Float64_matrix_like) | Matrix containing query points. | `zeros(0, 0)` |
| `reference` | [`Float64 matrix-like`](#doc_Float64_matrix_like) | Matrix containing the reference dataset. | `zeros(0, 0)` |
| `verbose` | [`Bool`](#doc_Bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Results are returned as a tuple, and can be unpacked directly into return values or stored directly as a tuple; undesired results can be ignored with the _ keyword.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `distances` | [`Float64 matrix-like`](#doc_Float64_matrix_like) | Matrix to save furthest neighbor distances to. | 
| `neighbors` | [`Int matrix-like`](#doc_Int_matrix_like) | Matrix to save neighbor indices to. | 
| `output_model` | [`ApproxKFNModel`](#doc_model) | File to save output model to. | 

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
For example, to find the 5 approximate furthest neighbors with ``reference_set`` as the reference set and ``query_set`` as the query set using DrusillaSelect, storing the furthest neighbor indices to ``neighbors`` and the furthest neighbor distances to ``distances``, one could call

```julia
julia> using CSV
julia> query_set = CSV.read("query_set.csv")
julia> reference_set = CSV.read("reference_set.csv")
julia> distances, neighbors, _ = approx_kfn(algorithm="ds", k=5,
            query=query_set, reference=reference_set)
```

and to perform approximate all-furthest-neighbors search with k=1 on the set ``data`` storing only the furthest neighbor distances to ``distances``, one could call

```julia
julia> using CSV
julia> reference_set = CSV.read("reference_set.csv")
julia> distances, _, _ = approx_kfn(k=1, reference=reference_set)
```

A trained model can be re-used.  If a model has been previously saved to ``model``, then we may find 3 approximate furthest neighbors on a query set ``new_query_set`` using that model and store the furthest neighbor indices into ``neighbors`` by calling

```julia
julia> using CSV
julia> new_query_set = CSV.read("new_query_set.csv")
julia> _, neighbors, _ = approx_kfn(input_model=model, k=3,
            query=new_query_set)
```

### See also

 - [k-furthest-neighbor search](#kfn)
 - [k-nearest-neighbor search](#knn)
 - [Fast approximate furthest neighbors with data-dependent candidate selection (pdf)](http://ratml.org/pub/pdf/2016fast.pdf)
 - [Approximate furthest neighbor in high dimensions (pdf)](https://www.rasmuspagh.net/papers/approx-furthest-neighbor-SISAP15.pdf)
 - [QDAFN class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/approx_kfn/qdafn.hpp)
 - [DrusillaSelect class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/approx_kfn/drusilla_select.hpp)

## cf()
{: #cf }

#### Collaborative Filtering
{: #cf_descr }

```julia
julia> using mlpack: cf
julia> output, output_model = cf( ;
                                 algorithm="NMF",
                                 all_user_recommendations=false,
                                 input_model=nothing, interpolation="average",
                                 iteration_only_termination=false,
                                 max_iterations=1000, min_residue=1e-05,
                                 neighbor_search="euclidean", neighborhood=5,
                                 normalization="none", query=zeros(Int, 0, 0),
                                 rank=0, recommendations=5, seed=0,
                                 test=zeros(0, 0), training=zeros(0, 0),
                                 verbose=false)
```

An implementation of several collaborative filtering (CF) techniques for recommender systems.  This can be used to train a new CF model, or use an existing CF model to compute recommendations. [Detailed documentation](#cf_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `algorithm` | [`String`](#doc_String) | Algorithm used for matrix factorization. | `"NMF"` |
| `all_user_recommendations` | [`Bool`](#doc_Bool) | Generate recommendations for all users. | `false` |
| `check_input_matrices` | [`Bool`](#doc_Bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `input_model` | [`CFModel`](#doc_model) | Trained CF model to load. | `nothing` |
| `interpolation` | [`String`](#doc_String) | Algorithm used for weight interpolation. | `"average"` |
| `iteration_only_termination` | [`Bool`](#doc_Bool) | Terminate only when the maximum number of iterations is reached. | `false` |
| `max_iterations` | [`Int`](#doc_Int) | Maximum number of iterations. If set to zero, there is no limit on the number of iterations. | `1000` |
| `min_residue` | [`Float64`](#doc_Float64) | Residue required to terminate the factorization (lower values generally mean better fits). | `1e-05` |
| `neighbor_search` | [`String`](#doc_String) | Algorithm used for neighbor search. | `"euclidean"` |
| `neighborhood` | [`Int`](#doc_Int) | Size of the neighborhood of similar users to consider for each query user. | `5` |
| `normalization` | [`String`](#doc_String) | Normalization performed on the ratings. | `"none"` |
| `query` | [`Int matrix-like`](#doc_Int_matrix_like) | List of query users for which recommendations should be generated. | `zeros(Int, 0, 0)` |
| `rank` | [`Int`](#doc_Int) | Rank of decomposed matrices (if 0, a heuristic is used to estimate the rank). | `0` |
| `recommendations` | [`Int`](#doc_Int) | Number of recommendations to generate for each query user. | `5` |
| `seed` | [`Int`](#doc_Int) | Set the random seed (0 uses std::time(NULL)). | `0` |
| `test` | [`Float64 matrix-like`](#doc_Float64_matrix_like) | Test set to calculate RMSE on. | `zeros(0, 0)` |
| `training` | [`Float64 matrix-like`](#doc_Float64_matrix_like) | Input dataset to perform CF on. | `zeros(0, 0)` |
| `verbose` | [`Bool`](#doc_Bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Results are returned as a tuple, and can be unpacked directly into return values or stored directly as a tuple; undesired results can be ignored with the _ keyword.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output` | [`Int matrix-like`](#doc_Int_matrix_like) | Matrix that will store output recommendations. | 
| `output_model` | [`CFModel`](#doc_model) | Output for trained CF model. | 

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
To train a CF model on a dataset ``training_set`` using NMF for decomposition and saving the trained model to ``model``, one could call: 

```julia
julia> using CSV
julia> training_set = CSV.read("training_set.csv")
julia> _, model = cf(algorithm="NMF", training=training_set)
```

Then, to use this model to generate recommendations for the list of users in the query set ``users``, storing 5 recommendations in ``recommendations``, one could call 

```julia
julia> using CSV
julia> users = CSV.read("users.csv"; type=Int)
julia> recommendations, _ = cf(input_model=model, query=users,
            recommendations=5)
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

```julia
julia> using mlpack: dbscan
julia> assignments, centroids = dbscan(input; epsilon=1.0, min_size=5,
          naive=false, selection_type="ordered", single_mode=false,
          tree_type="kd", verbose=false)
```

An implementation of DBSCAN clustering.  Given a dataset, this can compute and return a clustering of that dataset. [Detailed documentation](#dbscan_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`Bool`](#doc_Bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `epsilon` | [`Float64`](#doc_Float64) | Radius of each range search. | `1.0` |
| `input` | [`Float64 matrix-like`](#doc_Float64_matrix_like) | Input dataset to cluster. | `**--**` |
| `min_size` | [`Int`](#doc_Int) | Minimum number of points for a cluster. | `5` |
| `naive` | [`Bool`](#doc_Bool) | If set, brute-force range search (not tree-based) will be used. | `false` |
| `selection_type` | [`String`](#doc_String) | If using point selection policy, the type of selection to use ('ordered', 'random'). | `"ordered"` |
| `single_mode` | [`Bool`](#doc_Bool) | If set, single-tree range search (not dual-tree) will be used. | `false` |
| `tree_type` | [`String`](#doc_String) | If using single-tree or dual-tree search, the type of tree to use ('kd', 'r', 'r-star', 'x', 'hilbert-r', 'r-plus', 'r-plus-plus', 'cover', 'ball'). | `"kd"` |
| `verbose` | [`Bool`](#doc_Bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Results are returned as a tuple, and can be unpacked directly into return values or stored directly as a tuple; undesired results can be ignored with the _ keyword.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `assignments` | [`Int vector-like`](#doc_Int_vector_like) | Output matrix for assignments of each point. | 
| `centroids` | [`Float64 matrix-like`](#doc_Float64_matrix_like) | Matrix to save output centroids to. | 

### Detailed documentation
{: #dbscan_detailed-documentation }

This program implements the DBSCAN algorithm for clustering using accelerated tree-based range search.  The type of tree that is used may be parameterized, or brute-force range search may also be used.

The input dataset to be clustered may be specified with the `input` parameter; the radius of each range search may be specified with the `epsilon` parameters, and the minimum number of points in a cluster may be specified with the `min_size` parameter.

The `assignments` and `centroids` output parameters may be used to save the output of the clustering. `assignments` contains the cluster assignments of each point, and `centroids` contains the centroids of each cluster.

The range search may be controlled with the `tree_type`, `single_mode`, and `naive` parameters.  `tree_type` can control the type of tree used for range search; this can take a variety of values: 'kd', 'r', 'r-star', 'x', 'hilbert-r', 'r-plus', 'r-plus-plus', 'cover', 'ball'. The `single_mode` parameter will force single-tree search (as opposed to the default dual-tree search), and '`naive` will force brute-force range search.

### Example
An example usage to run DBSCAN on the dataset in ``input`` with a radius of 0.5 and a minimum cluster size of 5 is given below:

```julia
julia> using CSV
julia> input = CSV.read("input.csv")
julia> _, _ = dbscan(input; epsilon=0.5, min_size=5)
```

### See also

 - [DBSCAN on Wikipedia](https://en.wikipedia.org/wiki/DBSCAN)
 - [A density-based algorithm for discovering clusters in large spatial databases with noise (pdf)](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf)
 - [DBSCAN class documentation](https://github.com/mlpack/mlpack/blob/master/src/mlpack/methods/dbscan/dbscan.hpp)

## decision_tree()
{: #decision_tree }

#### Decision tree
{: #decision_tree_descr }

```julia
julia> using mlpack: decision_tree
julia> output_model, predictions, probabilities = decision_tree( ;
          input_model=nothing, labels=Int[], maximum_depth=0,
          minimum_gain_split=1e-07, minimum_leaf_size=20,
          print_training_accuracy=false, test=zeros(0, 0), test_labels=Int[],
          training=zeros(0, 0), verbose=false, weights=zeros(0, 0))
```

An implementation of an ID3-style decision tree for classification, which supports categorical data.  Given labeled data with numeric or categorical features, a decision tree can be trained and saved; or, an existing decision tree can be used for classification on new points. [Detailed documentation](#decision_tree_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`Bool`](#doc_Bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `input_model` | [`DecisionTreeModel`](#doc_model) | Pre-trained decision tree, to be used with test points. | `nothing` |
| `labels` | [`Int vector-like`](#doc_Int_vector_like) | Training labels. | `Int[]` |
| `maximum_depth` | [`Int`](#doc_Int) | Maximum depth of the tree (0 means no limit). | `0` |
| `minimum_gain_split` | [`Float64`](#doc_Float64) | Minimum gain for node splitting. | `1e-07` |
| `minimum_leaf_size` | [`Int`](#doc_Int) | Minimum number of points in a leaf. | `20` |
| `print_training_accuracy` | [`Bool`](#doc_Bool) | Print the training accuracy. | `false` |
| `test` | [`Tuple{Array{Bool, 1}, Array{Float64, 2}}`](#doc_Tuple_Array_Bool__1___Array_Float64__2__) | Testing dataset (may be categorical). | `zeros(0, 0)` |
| `test_labels` | [`Int vector-like`](#doc_Int_vector_like) | Test point labels, if accuracy calculation is desired. | `Int[]` |
| `training` | [`Tuple{Array{Bool, 1}, Array{Float64, 2}}`](#doc_Tuple_Array_Bool__1___Array_Float64__2__) | Training dataset (may be categorical). | `zeros(0, 0)` |
| `verbose` | [`Bool`](#doc_Bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |
| `weights` | [`Float64 matrix-like`](#doc_Float64_matrix_like) | The weight of labels | `zeros(0, 0)` |

### Output options

Results are returned as a tuple, and can be unpacked directly into return values or stored directly as a tuple; undesired results can be ignored with the _ keyword.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `output_model` | [`DecisionTreeModel`](#doc_model) | Output for trained decision tree. | 
| `predictions` | [`Int vector-like`](#doc_Int_vector_like) | Class predictions for each test point. | 
| `probabilities` | [`Float64 matrix-like`](#doc_Float64_matrix_like) | Class probabilities for each test point. | 

### Detailed documentation
{: #decision_tree_detailed-documentation }

Train and evaluate using a decision tree.  Given a dataset containing numeric or categorical features, and associated labels for each point in the dataset, this program can train a decision tree on that data.

The training set and associated labels are specified with the `training` and `labels` parameters, respectively.  The labels should be in the range `[0, num_classes - 1]`. Optionally, if `labels` is not specified, the labels are assumed to be the last dimension of the training dataset.

When a model is trained, the `output_model` output parameter may be used to save the trained model.  A model may be loaded for predictions with the `input_model` parameter.  The `input_model` parameter may not be specified when the `training` parameter is specified.  The `minimum_leaf_size` parameter specifies the minimum number of training points that must fall into each leaf for it to be split.  The `minimum_gain_split` parameter specifies the minimum gain that is needed for the node to split.  The `maximum_depth` parameter specifies the maximum depth of the tree.  If `print_training_accuracy` is specified, the training accuracy will be printed.

Test data may be specified with the `test` parameter, and if performance numbers are desired for that test set, labels may be specified with the `test_labels` parameter.  Predictions for each test point may be saved via the `predictions` output parameter.  Class probabilities for each prediction may be saved with the `probabilities` output parameter.

### Example
For example, to train a decision tree with a minimum leaf size of 20 on the dataset contained in ``data`` with labels ``labels``, saving the output model to ``tree`` and printing the training error, one could call

```julia
julia> using CSV
julia> data = CSV.read("data.csv")
julia> labels = CSV.read("labels.csv"; type=Int)
julia> tree, _, _ = decision_tree(labels=labels,
            minimum_gain_split=0.001, minimum_leaf_size=20,
            print_training_accuracy=1, training=data)
```

Then, to use that model to classify points in ``test_set`` and print the test error given the labels ``test_labels`` using that model, while saving the predictions for each point to ``predictions``, one could call 

```julia
julia> using CSV
julia> test_set = CSV.read("test_set.csv")
julia> test_labels = CSV.read("test_labels.csv"; type=Int)
julia> _, predictions, _ = decision_tree(input_model=tree,
            test=test_set, test_labels=test_labels)
```

### See also

 - [Random forest](#random_forest)
 - [Decision trees on Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning)
 - [Induction of Decision Trees (pdf)](https://www.hunch.net/~coms-4771/quinlan.pdf)
 - [DecisionTree C++ class documentation](../../user/methods/decision_tree.md)

## decision_tree_classify()
{: #decision_tree_classify }

#### Decision tree Prediction
{: #decision_tree_classify_descr }

```julia
julia> using mlpack: decision_tree_classify
julia> predictions = decision_tree_classify(input_model, test;
          test_labels=Int[], verbose=false)
```

Class predictions from train decision tree model. [Detailed documentation](#decision_tree_classify_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `check_input_matrices` | [`Bool`](#doc_Bool) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. | `false` |
| `input_model` | [`DecisionTreeModel`](#doc_model) | Pre-trained decision tree, to be used with test points. | `**--**` |
| `test` | [`Tuple{Array{Bool, 1}, Array{Float64, 2}}`](#doc_Tuple_Array_Bool__1___Array_Float64__2__) | Testing dataset (may contain categorical variables). | `**--**` |
| `test_labels` | [`Int vector-like`](#doc_Int_vector_like) | Test point labels, if accuracy calculation is desired. | `Int[]` |
| `verbose` | [`Bool`](#doc_Bool) | Display informational messages and the full list of parameters and timers at the end of execution. | `false` |

### Output options

Results are returned as a tuple, and can be unpacked directly into return values or stored directly as a tuple; undesired results can be ignored with the _ keyword.

| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `predictions` | [`Int vector-like`](#doc_Int_vector_like) | Class predictions for each test point. | 

### Detailed documentation
{: #decision_tree_classify_detailed-documentation }



### Example
