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

## mlpack_decision_tree_train
{: #decision_tree_train }

#### Decision tree training
{: #decision_tree_train_descr }

```bash
$ mlpack_decision_tree_train [--help] [--info <string>] [--labels_file
        <string>] [--maximum_depth 0] [--minimum_gain_split 1e-07]
        [--minimum_leaf_size 20] [--print_training_accuracy] --training_file
        <string> [--verbose] [--version] [--weights_file <string>]
        [--output_model_file <string>]
```

Training ID3-style decision tree model. [Detailed documentation](#decision_tree_train_detailed-documentation).



### Input options

| ***name*** | ***type*** | ***description*** | ***default*** |
|------------|------------|-------------------|---------------|
| `--check_input_matrices` | [`flag`](#doc_flag) | If specified, the input matrix is checked for NaN and inf values; an exception is thrown if any are found. |  |
| `--help (-h)` | [`flag`](#doc_flag) | Default help info.  <span class="special">Only exists in CLI binding.</span> |  |
| `--info` | [`string`](#doc_string) | Print help on a specific option.  <span class="special">Only exists in CLI binding.</span> | `''` |
| `--labels_file (-l)` | [`1-d index matrix file`](#doc_a_1_d_index_matrix_file) | Training labels. | `''` |
| `--maximum_depth (-D)` | [`int`](#doc_int) | Maximum depth of the tree (0 means no limit). | `0` |
| `--minimum_gain_split (-g)` | [`double`](#doc_double) | Minimum gain for node splitting. | `1e-07` |
| `--minimum_leaf_size (-n)` | [`int`](#doc_int) | Minimum number of points in a leaf. | `20` |
| `--print_training_accuracy (-a)` | [`flag`](#doc_flag) | Print the training accuracy. |  |
| `--training_file (-t)` | [`2-d categorical matrix file`](#doc_a_2_d_categorical_matrix_file) | Training dataset (may contain categorical variables). | `**--**` |
| `--verbose (-v)` | [`flag`](#doc_flag) | Display informational messages and the full list of parameters and timers at the end of execution. |  |
| `--version (-V)` | [`flag`](#doc_flag) | Display the version of mlpack.  <span class="special">Only exists in CLI binding.</span> |  |
| `--weights_file (-w)` | [`2-d matrix file`](#doc_a_2_d_matrix_file) | The weight of labels | `''` |

### Output options


| ***name*** | ***type*** | ***description*** |
|------------|------------|-------------------|
| `--output_model_file (-M)` | [`DecisionTreeModel file`](#doc_model) | Output for trained decision tree. | 

### Detailed documentation
{: #decision_tree_train_detailed-documentation }

Train using a decision tree.  Given a dataset containing numeric or categorical features, and associated labels for each point in the dataset, this program can train a decision tree on that data.

The training set and associated labels are specified with the `--training_file (-t)` and `--labels_file (-l)` parameters, respectively.  The labels should be in the range `[0, num_classes - 1]`. Optionally, if `--labels_file (-l)` is not specified, the labels are assumed to be the last dimension of the training dataset.

The trained model is returned, and can then be used for prediction. The `--minimum_leaf_size (-n)` parameter specifies the minimum number of training points that must fall into each leaf for it to be split.  The `--minimum_gain_split (-g)` parameter specifies the minimum gain that is needed for the node to split.  The `--maximum_depth (-D)` parameter specifies the maximum depth of the tree.  If `--print_training_accuracy (-a)` is specified, the training accuracy will be printed.

### Example
