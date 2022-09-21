# Approximate furthest neighbor search tutorial

mlpack implements multiple strategies for approximate furthest neighbor
search in its `mlpack_approx_kfn` and `mlpack_kfn` command-line programs (each
program corresponds to different techniques).  This tutorial discusses what
problems these algorithms solve and how to use each of the techniques that
mlpack implements.

Note that these functions are available as bindings to other languages too, and
all the examples here can be adapted accordingly.

mlpack implements five approximate furthest neighbor search algorithms:

 - brute-force search (in `mlpack_kfn`)
 - single-tree search (in `mlpack_kfn`)
 - dual-tree search (in `mlpack_kfn`)
 - query-dependent approximate furthest neighbor (QDAFN) (in `mlpack_approx_kfn`)
 - DrusillaSelect (in `mlpack_approx_kfn`)

These methods are described in the following papers:

```
@inproceedings{curtin2013tree,
  title={Tree-Independent Dual-Tree Algorithms},
  author={Curtin, Ryan R. and March, William B. and Ram, Parikshit and Anderson,
      David V. and Gray, Alexander G. and Isbell Jr., Charles L.},
  booktitle={Proceedings of The 30th International Conference on Machine
      Learning (ICML '13)},
  pages={1435--1443},
  year={2013}
}
```

```
@incollection{pagh2015approximate,
  title={Approximate furthest neighbor in high dimensions},
  author={Pagh, Rasmus and Silvestri, Francesco and Sivertsen, Johan and Skala,
      Matthew},
  booktitle={Similarity Search and Applications},
  pages={3--14},
  year={2015},
  publisher={Springer}
}
```

```
@incollection{curtin2016fast,
  title={Fast approximate furthest neighbors with data-dependent candidate
      selection},
  author={Curtin, Ryan R., and Gardner, Andrew B.},
  booktitle={Similarity Search and Applications},
  pages={221--235},
  year={2016},
  publisher={Springer}
}
```

```
@article{curtin2018exploiting,
  title={Exploiting the structure of furthest neighbor search for fast
      approximate results},
  author={Curtin, Ryan R., and Echauz, Javier, and Gardner, Andrew B.},
  journal={Information Systems},
  year={2018},
  publisher={Elsevier}
}
```

The problem of furthest neighbor search is simple, and is the opposite of the
much-more-studied nearest neighbor search problem.  Given a set of reference
points `R` (the set in which we are searching), and a set of query points `Q`
(the set of points for which we want the furthest neighbor), our goal is to
return the `k` furthest neighbors for each query point in `Q`:

```
k-argmax_{p_r in R} d(p_q, p_r).
```

In order to solve this problem, mlpack provides a number of interfaces.

 - two simple command-line executables to calculate approximate furthest
   neighbors
 - a simple C++ class for QDAFN"
 - a simple C++ class for DrusillaSelect
 - a simple C++ class for tree-based and brute-force search

## Which algorithm should be used?

There are three algorithms for furthest neighbor search that mlpack
implements, and each is suited to a different setting.  Below is some basic
guidance on what should be used.  Note that the question of "which algorithm
should be used" is a very difficult question to answer, so the guidance below is
just that---guidance---and may not be right for a particular problem.

 - `DrusillaSelect` is very fast and will perform extremely well for datasets
   with outliers or datasets with structure (like low-dimensional datasets
   embedded in high dimensions)
 - `QDAFN` is a random approach and therefore should be well-suited for
   datasets with little to no structure
 - The tree-based approaches (the `KFN` class and the `mlpack_kfn` program) is
   best suited for low-dimensional datasets, and is most effective when very
   small levels of approximation are desired, or when exact results are desired.
 - Dual-tree search is most useful when the query set is large and structured
   (like for all-furthest-neighbor search).
 - Single-tree search is more useful when the query set is small.

## Command-line `mlpack_approx_kfn` and `mlpack_kfn`

mlpack provides two command-line programs to solve approximate furthest neighbor
search:

 - `mlpack_approx_kfn`, for the QDAFN and DrusillaSelect approaches
 - `mlpack_kfn`, for exact and approximate tree-based approaches

These two programs allow a large number of algorithms to be used to find
approximate furthest neighbors.  Note that the `mlpack_kfn` program is also
documented in the [KNN tutorial](knn.md) page, as it shares options with the
`mlpack_knn` program.

Below are several examples of how the `mlpack_approx_kfn` and `mlpack_kfn`
programs might be used.  The first examples focus on the `mlpack_approx_kfn`
program, and the last few show how `mlpack_kfn` can be used to produce
approximate results.

### Calculate 5 furthest neighbors with default options

Here we have a query dataset `queries.csv` and a reference dataset `refs.csv`
and we wish to find the 5 furthest neighbors of every query point in the
reference dataset.  We may do that with the `mlpack_approx_kfn` algorithm,
using the default of the `DrusillaSelect` algorithm with default parameters.

```sh
$ mlpack_approx_kfn -q queries.csv -r refs.csv -v -k 5 -n n.csv -d d.csv
[INFO ] Loading 'refs.csv' as CSV data.  Size is 3 x 1000.
[INFO ] Building DrusillaSelect model...
[INFO ] Model built.
[INFO ] Loading 'queries.csv' as CSV data.  Size is 3 x 1000.
[INFO ] Searching for 5 furthest neighbors with DrusillaSelect...
[INFO ] Search complete.
[INFO ] Saving CSV data to 'n.csv'.
[INFO ] Saving CSV data to 'd.csv'.
[INFO ]
[INFO ] Execution parameters:
[INFO ]   algorithm: ds
[INFO ]   calculate_error: false
[INFO ]   distances_file: d.csv
[INFO ]   exact_distances_file: ""
[INFO ]   help: false
[INFO ]   info: ""
[INFO ]   input_model_file: ""
[INFO ]   k: 5
[INFO ]   neighbors_file: n.csv
[INFO ]   num_projections: 5
[INFO ]   num_tables: 5
[INFO ]   output_model_file: ""
[INFO ]   query_file: queries.csv
[INFO ]   reference_file: refs.csv
[INFO ]   verbose: true
[INFO ]   version: false
[INFO ]
[INFO ] Program timers:
[INFO ]   drusilla_select_construct: 0.000342s
[INFO ]   drusilla_select_search: 0.000780s
[INFO ]   loading_data: 0.010689s
[INFO ]   saving_data: 0.005585s
[INFO ]   total_time: 0.018592s
```

Convenient timers for parts of the program operation are printed.  The results,
saved in `n.csv` and `d.csv`, indicate the furthest neighbors and distances for
each query point.  The row of the output file indicates the query point that the
results are for.  The neighbors are listed from furthest to nearest; so, the 4th
element in the 3rd row of `d.csv` indicates the distance between the 3rd query
point in `queries.csv` and its approximate 4th furthest neighbor.  Similarly,
the same element in `n.csv` indicates the index of the approximate 4th furthest
neighbor (with respect to `refs.csv`).

### Specifying algorithm parameters for `DrusillaSelect`

The `-p` (`--num_projections`) and `-t` (`--num_tables`) parameters affect the
running of the `DrusillaSelect` algorithm and the QDAFN algorithm.
Specifically, larger values for each of these parameters will search more
possible candidate furthest neighbors and produce better results (at the cost of
runtime).  More details on how each of these parameters works is available in
the original papers, the mlpack source, or the documentation given by `--help`.

In the example below, we run `DrusillaSelect` to find 4 furthest neighbors using
10 tables and 2 points in each table.  In this case we have chosen to omit the
`-n n.csv` option, meaning that only the output candidate distances will be
written to `d.csv`.

```sh
$ mlpack_approx_kfn -q queries.csv -r refs.csv -v -k 4 -n n.csv -d d.csv -t 10 -p 2
[INFO ] Loading 'refs.csv' as CSV data.  Size is 3 x 1000.
[INFO ] Building DrusillaSelect model...
[INFO ] Model built.
[INFO ] Loading 'queries.csv' as CSV data.  Size is 3 x 1000.
[INFO ] Searching for 4 furthest neighbors with DrusillaSelect...
[INFO ] Search complete.
[INFO ] Saving CSV data to 'n.csv'.
[INFO ] Saving CSV data to 'd.csv'.
[INFO ]
[INFO ] Execution parameters:
[INFO ]   algorithm: ds
[INFO ]   calculate_error: false
[INFO ]   distances_file: d.csv
[INFO ]   exact_distances_file: ""
[INFO ]   help: false
[INFO ]   info: ""
[INFO ]   input_model_file: ""
[INFO ]   k: 4
[INFO ]   neighbors_file: n.csv
[INFO ]   num_projections: 2
[INFO ]   num_tables: 10
[INFO ]   output_model_file: ""
[INFO ]   query_file: queries.csv
[INFO ]   reference_file: refs.csv
[INFO ]   verbose: true
[INFO ]   version: false
[INFO ]
[INFO ] Program timers:
[INFO ]   drusilla_select_construct: 0.000645s
[INFO ]   drusilla_select_search: 0.000551s
[INFO ]   loading_data: 0.008518s
[INFO ]   saving_data: 0.003734s
[INFO ]   total_time: 0.014019s
```

### Using QDAFN instead of `DrusillaSelect`

The algorithm to be used for approximate furthest neighbor search can be
specified with the `--algorithm` (`-a`) option to the `mlpack_approx_kfn`
program.  Below, we use the QDAFN algorithm instead of the default.  We leave
the `-p` and `-t` options at their defaults---even though QDAFN often requires
more tables and points to get the same quality of results.

```sh
$ mlpack_approx_kfn -q queries.csv -r refs.csv -v -k 3 -n n.csv -d d.csv -a qdafn
[INFO ] Loading 'refs.csv' as CSV data.  Size is 3 x 1000.
[INFO ] Building QDAFN model...
[INFO ] Model built.
[INFO ] Loading 'queries.csv' as CSV data.  Size is 3 x 1000.
[INFO ] Searching for 3 furthest neighbors with QDAFN...
[INFO ] Search complete.
[INFO ] Saving CSV data to 'n.csv'.
[INFO ] Saving CSV data to 'd.csv'.
[INFO ]
[INFO ] Execution parameters:
[INFO ]   algorithm: qdafn
[INFO ]   calculate_error: false
[INFO ]   distances_file: d.csv
[INFO ]   exact_distances_file: ""
[INFO ]   help: false
[INFO ]   info: ""
[INFO ]   input_model_file: ""
[INFO ]   k: 3
[INFO ]   neighbors_file: n.csv
[INFO ]   num_projections: 5
[INFO ]   num_tables: 5
[INFO ]   output_model_file: ""
[INFO ]   query_file: queries.csv
[INFO ]   reference_file: refs.csv
[INFO ]   verbose: true
[INFO ]   version: false
[INFO ]
[INFO ] Program timers:
[INFO ]   loading_data: 0.008380s
[INFO ]   qdafn_construct: 0.003399s
[INFO ]   qdafn_search: 0.000886s
[INFO ]   saving_data: 0.002253s
[INFO ]   total_time: 0.015465s
```

### Printing results quality with exact distances

The `mlpack_approx_kfn` program can calculate the quality of the results if the
`--calculate_error` (`-e`) flag is specified.  Below we use the program with its
default parameters and calculate the error, which is displayed in the output.
The error is only calculated for the furthest neighbor, not all k; therefore, in
this example we have set `-k` to `1`.

```sh
$ mlpack_approx_kfn -q queries.csv -r refs.csv -v -k 1 -e -q -n n.csv
[INFO ] Loading 'refs.csv' as CSV data.  Size is 3 x 1000.
[INFO ] Building DrusillaSelect model...
[INFO ] Model built.
[INFO ] Loading 'queries.csv' as CSV data.  Size is 3 x 1000.
[INFO ] Searching for 1 furthest neighbors with DrusillaSelect...
[INFO ] Search complete.
[INFO ] Calculating exact distances...
[INFO ] 28891 node combinations were scored.
[INFO ] 37735 base cases were calculated.
[INFO ] Calculation complete.
[INFO ] Average error: 1.08417.
[INFO ] Maximum error: 1.28712.
[INFO ] Minimum error: 1.
[INFO ]
[INFO ] Execution parameters:
[INFO ]   algorithm: ds
[INFO ]   calculate_error: true
[INFO ]   distances_file: ""
[INFO ]   exact_distances_file: ""
[INFO ]   help: false
[INFO ]   info: ""
[INFO ]   input_model_file: ""
[INFO ]   k: 3
[INFO ]   neighbors_file: ""
[INFO ]   num_projections: 5
[INFO ]   num_tables: 5
[INFO ]   output_model_file: ""
[INFO ]   query_file: queries.csv
[INFO ]   reference_file: refs.csv
[INFO ]   verbose: true
[INFO ]   version: false
[INFO ]
[INFO ] Program timers:
[INFO ]   computing_neighbors: 0.001476s
[INFO ]   drusilla_select_construct: 0.000309s
[INFO ]   drusilla_select_search: 0.000495s
[INFO ]   loading_data: 0.008462s
[INFO ]   total_time: 0.011670s
[INFO ]   tree_building: 0.000202s
```

Note that the output includes three lines indicating the error:

```sh
[INFO ] Average error: 1.08417.
[INFO ] Maximum error: 1.28712.
[INFO ] Minimum error: 1.
```

In this case, a minimum error of 1 indicates an exact result, and over the
entire query set the algorithm has returned a furthest neighbor candidate with
maximum error 1.28712.

### Using cached exact distances for quality results

However, for large datasets, calculating the error may take a long time, because
the exact furthest neighbors must be calculated.  Therefore, if the exact
furthest neighbor distances are already known, they may be passed in with the
`--exact_distances_file` (`-x`) option in order to avoid the calculation.  In
the example below, we assume `exact.csv` contains the exact furthest neighbor
distances.  We run the `qdafn` algorithm in this example.

Note that the `-e` option must be specified for the `-x` option have any effect.

```sh
$ mlpack_approx_kfn -q queries.csv -r refs.csv -k 1 -e -x exact.csv -n n.csv -v -a qdafn
[INFO ] Loading 'refs.csv' as CSV data.  Size is 3 x 1000.
[INFO ] Building QDAFN model...
[INFO ] Model built.
[INFO ] Loading 'queries.csv' as CSV data.  Size is 3 x 1000.
[INFO ] Searching for 1 furthest neighbors with QDAFN...
[INFO ] Search complete.
[INFO ] Loading 'exact.csv' as raw ASCII formatted data.  Size is 1 x 1000.
[INFO ] Average error: 1.06914.
[INFO ] Maximum error: 1.67407.
[INFO ] Minimum error: 1.
[INFO ] Saving CSV data to 'n.csv'.
[INFO ]
[INFO ] Execution parameters:
[INFO ]   algorithm: qdafn
[INFO ]   calculate_error: true
[INFO ]   distances_file: ""
[INFO ]   exact_distances_file: exact.csv
[INFO ]   help: false
[INFO ]   info: ""
[INFO ]   input_model_file: ""
[INFO ]   k: 1
[INFO ]   neighbors_file: n.csv
[INFO ]   num_projections: 5
[INFO ]   num_tables: 5
[INFO ]   output_model_file: ""
[INFO ]   query_file: queries.csv
[INFO ]   reference_file: refs.csv
[INFO ]   verbose: true
[INFO ]   version: false
[INFO ]
[INFO ] Program timers:
[INFO ]   loading_data: 0.010348s
[INFO ]   qdafn_construct: 0.000318s
[INFO ]   qdafn_search: 0.000793s
[INFO ]   saving_data: 0.000259s
[INFO ]   total_time: 0.012254s
```

### Using tree-based approximation with `mlpack_kfn`

The `mlpack_kfn` algorithm allows specifying a desired approximation level with
the `--epsilon` (`-e`) option.  The parameter must be greater than or equal to 0
and less than 1.  A setting of 0 indicates exact search.

The example below runs dual-tree furthest neighbor search (the default
algorithm) with the approximation parameter set to 0.5.

```sh
$ mlpack_kfn -q queries.csv -r refs.csv -v -k 3 -e 0.5 -n n.csv -d d.csv
[INFO ] Loading 'refs.csv' as CSV data.  Size is 3 x 1000.
[INFO ] Loaded reference data from 'refs.csv' (3x1000).
[INFO ] Building reference tree...
[INFO ] Tree built.
[INFO ] Loading 'queries.csv' as CSV data.  Size is 3 x 1000.
[INFO ] Loaded query data from 'queries.csv' (3x1000).
[INFO ] Searching for 3 neighbors with dual-tree kd-tree search...
[INFO ] 1611 node combinations were scored.
[INFO ] 13938 base cases were calculated.
[INFO ] 1611 node combinations were scored.
[INFO ] 13938 base cases were calculated.
[INFO ] Search complete.
[INFO ] Saving CSV data to 'n.csv'.
[INFO ] Saving CSV data to 'd.csv'.
[INFO ]
[INFO ] Execution parameters:
[INFO ]   algorithm: dual_tree
[INFO ]   distances_file: d.csv
[INFO ]   epsilon: 0.5
[INFO ]   help: false
[INFO ]   info: ""
[INFO ]   input_model_file: ""
[INFO ]   k: 3
[INFO ]   leaf_size: 20
[INFO ]   naive: false
[INFO ]   neighbors_file: n.csv
[INFO ]   output_model_file: ""
[INFO ]   percentage: 1
[INFO ]   query_file: queries.csv
[INFO ]   random_basis: false
[INFO ]   reference_file: refs.csv
[INFO ]   seed: 0
[INFO ]   single_mode: false
[INFO ]   tree_type: kd
[INFO ]   true_distances_file: ""
[INFO ]   true_neighbors_file: ""
[INFO ]   verbose: true
[INFO ]   version: false
[INFO ]
[INFO ] Program timers:
[INFO ]   computing_neighbors: 0.000442s
[INFO ]   loading_data: 0.008060s
[INFO ]   saving_data: 0.002850s
[INFO ]   total_time: 0.012667s
[INFO ]   tree_building: 0.000251s
```

Note that the format of the output files `d.csv` and `n.csv` are the same as
for `mlpack_approx_kfn`.

### Different algorithms with `mlpack_kfn`

The `mlpack_kfn` program offers a large number of different algorithms that can
be used.  The `--algorithm` (`-a`) parameter may be used to specify three main
different algorithm types: `naive` (brute-force search), `single_tree`
(single-tree search), `dual_tree` (dual-tree search, the default), and `greedy`
("defeatist" greedy search, which goes to one leaf node of the tree then
terminates).  The example below uses single-tree search to find approximate
neighbors with epsilon set to 0.1.

```sh
mlpack_kfn -q queries.csv -r refs.csv -v -k 3 -e 0.1 -n n.csv -d d.csv -a single_tree
[INFO ] Loading 'refs.csv' as CSV data.  Size is 3 x 1000.
[INFO ] Loaded reference data from 'refs.csv' (3x1000).
[INFO ] Building reference tree...
[INFO ] Tree built.
[INFO ] Loading 'queries.csv' as CSV data.  Size is 3 x 1000.
[INFO ] Loaded query data from 'queries.csv' (3x1000).
[INFO ] Searching for 3 neighbors with single-tree kd-tree search...
[INFO ] 13240 node combinations were scored.
[INFO ] 15924 base cases were calculated.
[INFO ] Search complete.
[INFO ] Saving CSV data to 'n.csv'.
[INFO ] Saving CSV data to 'd.csv'.
[INFO ]
[INFO ] Execution parameters:
[INFO ]   algorithm: single_tree
[INFO ]   distances_file: d.csv
[INFO ]   epsilon: 0.1
[INFO ]   help: false
[INFO ]   info: ""
[INFO ]   input_model_file: ""
[INFO ]   k: 3
[INFO ]   leaf_size: 20
[INFO ]   naive: false
[INFO ]   neighbors_file: n.csv
[INFO ]   output_model_file: ""
[INFO ]   percentage: 1
[INFO ]   query_file: queries.csv
[INFO ]   random_basis: false
[INFO ]   reference_file: refs.csv
[INFO ]   seed: 0
[INFO ]   single_mode: false
[INFO ]   tree_type: kd
[INFO ]   true_distances_file: ""
[INFO ]   true_neighbors_file: ""
[INFO ]   verbose: true
[INFO ]   version: false
[INFO ]
[INFO ] Program timers:
[INFO ]   computing_neighbors: 0.000850s
[INFO ]   loading_data: 0.007858s
[INFO ]   saving_data: 0.003445s
[INFO ]   total_time: 0.013084s
[INFO ]   tree_building: 0.000250s
```

## Saving a model for later use

The `mlpack_approx_kfn` and `mlpack_kfn` programs both allow models to be saved
and loaded for future use.  The `--output_model_file` (`-M`) option allows
specifying where to save a model, and the `--input_model_file` (`-m`) option
allows a model to be loaded instead of trained.  So, if you specify
`--input_model_file` then you do not need to specify `--reference_file` (`-r`),
`--num_projections` (`-p`), or `--num_tables` (`-t`).

The example below saves a model with 10 projections and 5 tables.  Note that
neither `--query_file` (`-q`) nor `-k` are specified; this run only builds the
model and saves it to `model.bin`.

```sh
$ mlpack_approx_kfn -r refs.csv -t 5 -p 10 -v -M model.bin
[INFO ] Loading 'refs.csv' as CSV data.  Size is 3 x 1000.
[INFO ] Building DrusillaSelect model...
[INFO ] Model built.
[INFO ]
[INFO ] Execution parameters:
[INFO ]   algorithm: ds
[INFO ]   calculate_error: false
[INFO ]   distances_file: ""
[INFO ]   exact_distances_file: ""
[INFO ]   help: false
[INFO ]   info: ""
[INFO ]   input_model_file: ""
[INFO ]   k: 0
[INFO ]   neighbors_file: ""
[INFO ]   num_projections: 10
[INFO ]   num_tables: 5
[INFO ]   output_model_file: model.bin
[INFO ]   query_file: ""
[INFO ]   reference_file: refs.csv
[INFO ]   verbose: true
[INFO ]   version: false
[INFO ]
[INFO ] Program timers:
[INFO ]   drusilla_select_construct: 0.000321s
[INFO ]   loading_data: 0.004700s
[INFO ]   total_time: 0.007320s
```

Now, with the model saved, we can run approximate furthest neighbor search on a
query set using the saved model:

```sh
$ mlpack_approx_kfn -m model.bin -q queries.csv -k 3 -d d.csv -n n.csv -v
[INFO ] Loading 'queries.csv' as CSV data.  Size is 3 x 1000.
[INFO ] Searching for 3 furthest neighbors with DrusillaSelect...
[INFO ] Search complete.
[INFO ] Saving CSV data to 'n.csv'.
[INFO ] Saving CSV data to 'd.csv'.
[INFO ]
[INFO ] Execution parameters:
[INFO ]   algorithm: ds
[INFO ]   calculate_error: false
[INFO ]   distances_file: d.csv
[INFO ]   exact_distances_file: ""
[INFO ]   help: false
[INFO ]   info: ""
[INFO ]   input_model_file: model.bin
[INFO ]   k: 3
[INFO ]   neighbors_file: n.csv
[INFO ]   num_projections: 5
[INFO ]   num_tables: 5
[INFO ]   output_model_file: ""
[INFO ]   query_file: queries.csv
[INFO ]   reference_file: ""
[INFO ]   verbose: true
[INFO ]   version: false
[INFO ]
[INFO ] Program timers:
[INFO ]   drusilla_select_search: 0.000878s
[INFO ]   loading_data: 0.004599s
[INFO ]   saving_data: 0.003006s
[INFO ]   total_time: 0.009234s
```

These options work in the same way for both the `mlpack_approx_kfn` and
`mlpack_kfn` programs.

### Final command-line program notes

Both the `mlpack_kfn` and `mlpack_approx_kfn` programs contain numerous options
not fully documented in these short examples.  You can run each program with the
`--help` (`-h`) option for more information.

## `DrusillaSelect` C++ class

mlpack provides a simple `DrusillaSelect` C++ class that can be used inside of
C++ programs to perform approximate furthest neighbor search.  The class has
only one template parameter---`MatType`---which specifies the type of matrix to
be use.  That means the class can be used with either dense data (of type
`arma::mat`) or sparse data (of type `arma::sp_mat`).

The following examples show simple usage of this class.

### Approximate furthest neighbors with defaults

The code below builds a `DrusillaSelect` model with default options on the
matrix `dataset`, then queries for the approximate furthest neighbor of every
point in the `queries` matrix.

```c++
#include <mlpack.hpp>

using namespace mlpack;

// The reference dataset.
extern arma::mat dataset;
// The query set.
extern arma::mat queries;

// Construct the model with defaults.
DrusillaSelect<> ds(dataset);

// Query the model, putting output into the following two matrices.
arma::mat distances;
arma::Mat<size_t> neighbors;
ds.Search(queries, 1, neighbors, distances);
```

At the end of this code, both the `distances` and `neighbors` matrices will have
number of columns equal to the number of columns in the `queries` matrix.  So,
each column of the `distances` and `neighbors` matrices are the distances or
neighbors of the corresponding column in the `queries` matrix.

### Custom numbers of tables and projections

The following example constructs a `DrusillaSelect` model with 10 tables and 5
projections.  Once that is done it performs the same task as the previous
example.

```c++
#include <mlpack.hpp>

using namespace mlpack;

// The reference dataset.
extern arma::mat dataset;
// The query set.
extern arma::mat queries;

// Construct the model with custom parameters.
DrusillaSelect<> ds(dataset, 10, 5);

// Query the model, putting output into the following two matrices.
arma::mat distances;
arma::Mat<size_t> neighbors;
ds.Search(queries, 1, neighbors, distances);
```

### Accessing the candidate set

The `DrusillaSelect` algorithm merely scans the reference set and extracts a
number of points that will be queried in a brute-force fashion when the
`Search()` method is called.  We can access this set with the `CandidateSet()`
method.  The code below prints the fifth point of the candidate set.

```c++
#include <mlpack.hpp>

using namespace mlpack;

// The reference dataset.
extern arma::mat dataset;

// Construct the model with custom parameters.
DrusillaSelect<> ds(dataset, 10, 5);

// Print the fifth point of the candidate set.
std::cout << ds.CandidateSet().col(4).t();
```

### Retraining on a new reference set

It is possible to retrain a `DrusillaSelect` model with new parameters or with a
new reference set.  This is functionally equivalent to creating a new model.
The example code below creates a first \c DrusillaSelect model using 3 tables
and 10 projections, and then retrains this with the same reference set using 10
tables and 3 projections.

```c++
#include <mlpack.hpp>

using namespace mlpack;

// The reference dataset.
extern arma::mat dataset;

// Construct the model with initial parameters.
DrusillaSelect<> ds(dataset, 3, 10);

// Now retrain with different parameters.
ds.Train(dataset, 10, 3);
```

### Running on sparse data

We can set the template parameter for `DrusillaSelect` to `arma::sp_mat` in
order to perform furthest neighbor search on sparse data.  This code below
creates a `DrusillaSelect` model using 4 tables and 6 projections with sparse
input data, then searches for 3 approximate furthest neighbors.

```c++
#include <mlpack.hpp>

using namespace mlpack;

// The reference dataset.
extern arma::sp_mat dataset;
// The query dataset.
extern arma::sp_mat querySet;

// Construct the model on sparse data.
DrusillaSelect<arma::sp_mat> ds(dataset, 4, 6);

// Search on query data.
arma::Mat<size_t> neighbors;
arma::mat distances;
ds.Search(querySet, 3, neighbors, distances);
```

## QDAFN C++ class

mlpack also provides a standalone simple `QDAFN` class for furthest neighbor
search.  The API for this class is virtually identical to the `DrusillaSelect`
class, and also has one template parameter to specify the type of matrix to be
used (dense or sparse or other).

The following subsections demonstrate usage of the `QDAFN` class in the same way
as the previous section's examples for `DrusillaSelect`.

### Approximate furthest neighbors with defaults

The code below builds a `QDAFN` model with default options on the matrix
`dataset`, then queries for the approximate furthest neighbor of every point in
the `queries` matrix.

```c++
#include <mlpack.hpp>

using namespace mlpack;

// The reference dataset.
extern arma::mat dataset;
// The query set.
extern arma::mat queries;

// Construct the model with defaults.
QDAFN<> qd(dataset);

// Query the model, putting output into the following two matrices.
arma::mat distances;
arma::Mat<size_t> neighbors;
qd.Search(queries, 1, neighbors, distances);
```

At the end of this code, both the `distances` and `neighbors` matrices will have
number of columns equal to the number of columns in the `queries` matrix.  So,
each column of the `distances` and `neighbors` matrices are the distances or
neighbors of the corresponding column in the `queries` matrix.

### Custom numbers of tables and projections

The following example constructs a `QDAFN` model with 15 tables and 30
projections.  Once that is done it performs the same task as the previous
example.

```c++
#include <mlpack.hpp>

using namespace mlpack;

// The reference dataset.
extern arma::mat dataset;
// The query set.
extern arma::mat queries;

// Construct the model with custom parameters.
QDAFN<> qdafn(dataset, 15, 30);

// Query the model, putting output into the following two matrices.
arma::mat distances;
arma::Mat<size_t> neighbors;
qdafn.Search(queries, 1, neighbors, distances);
```

### Accessing the candidate set

The `QDAFN` algorithm scans the reference set, extracting points that have been
projected onto random directions.  Each random direction corresponds to a single
table.  The `QDAFN` class stores these points as a vector of matrices, which can
be accessed with the `CandidateSet()` method.  The code below prints the fifth
point of the candidate set of the third table.

```c++
#include <mlpack.hpp>

using namespace mlpack;

// The reference dataset.
extern arma::mat dataset;

// Construct the model with custom parameters.
QDAFN<> qdafn(dataset, 10, 5);

// Print the fifth point of the candidate set.
std::cout << ds.CandidateSet(2).col(4).t();
```

### Retraining on a new reference set

It is possible to retrain a `QDAFN` model with new parameters or with a new
reference set.  This is functionally equivalent to creating a new model.  The
example code below creates a first `QDAFN` model using 10 tables and 40
projections, and then retrains this with the same reference set using 15 tables
and 25 projections.

```c++
#include <mlpack.hpp>

using namespace mlpack;

// The reference dataset.
extern arma::mat dataset;

// Construct the model with initial parameters.
QDAFN<> qdafn(dataset, 3, 10);

// Now retrain with different parameters.
qdafn.Train(dataset, 10, 3);
```

### Running on sparse data

We can set the template parameter for `QDAFN` to `arma::sp_mat` in order to
perform furthest neighbor search on sparse data.  This code below creates a
`QDAFN` model using 20 tables and 60 projections with sparse input data, then
searches for 3 approximate furthest neighbors.

```c++
#include <mlpack.hpp>

using namespace mlpack;

// The reference dataset.
extern arma::sp_mat dataset;
// The query dataset.
extern arma::sp_mat querySet;

// Construct the model on sparse data.
QDAFN<arma::sp_mat> qdafn(dataset, 20, 60);

// Search on query data.
arma::Mat<size_t> neighbors;
arma::mat distances;
qdafn.Search(querySet, 3, neighbors, distances);
```

## KFN C++ class

The extensive `NeighborSearch` class also provides a way to search for
approximate furthest neighbors using a different, tree-based technique.  For
full documentation on this class, see the [NeighborSearch
tutorial](nstutorial.md).  The `KFN` class is a convenient typedef of the
`NeighborSearch` class that can be used to perform the furthest neighbors task
with `kd`-trees.

In the following subsections, the `KFN` class is used in short code examples.

### Simple furthest neighbors example

The `KFN` class has construction semantics similar to `DrusillaSelect` and
`QDAFN`.  The example below constructs a `KFN` object (which will build the
tree on the reference set), but note that the third parameter to the constructor
allows us to specify our desired level of approximation.  In this example we
choose `epsilon = 0.05`.  Then, the code searches for 3 approximate furthest
neighbors.

```c++
#include <mlpack.hpp>

using namespace mlpack;

// The reference dataset.
extern arma::mat dataset;
// The query set.
extern arma::mat querySet;

// Construct the object, performing the default dual-tree search with
// approximation level epsilon = 0.05.
KFN kfn(dataset, KFN::DUAL_TREE_MODE, 0.05);

// Search for approximate furthest neighbors.
arma::Mat<size_t> neighbors;
arma::mat distances;
kfn.Search(querySet, 3, neighbors, distances);
```

### Retraining on a new reference set

Like the `QDAFN` and `DrusillaSelect` classes, the `KFN` class is capable of
retraining on a new reference set.  The code below demonstrates this.

```c++
#include <mlpack.hpp>

using namespace mlpack;

// The original reference set we train on.
extern arma::mat dataset;
// The new reference set we retrain on.
extern arma::mat newDataset;

// Construct the object with approximation level 0.1.
KFN kfn(dataset, DUAL_TREE_MODE, 0.1);

// Retrain on the new reference set.
kfn.Train(newDataset);
```

### Searching in single-tree mode

The particular mode to be used in search can be specified in the constructor.
In this example, we use single-tree search (as opposed to the default of
dual-tree search).

```c++
#include <mlpack.hpp>

using namespace mlpack;

// The reference set.
extern arma::mat dataset;
// The query set.
extern arma::mat querySet;

// Construct the object with approximation level 0.25 and in single tree search
// mode.
KFN kfn(dataset, SINGLE_TREE_MODE, 0.25);

// Search for 5 approximate furthest neighbors.
arma::Mat<size_t> neighbors;
arma::mat distances;
kfn.Search(querySet, 5, neighbors, distances);
```

### Searching in brute-force mode

If desired, brute-force search ("naive search") can be used to find the furthest
neighbors; however, the result will not be approximate---it will be exact (since
every possibility will be considered).  The code below performs exact furthest
neighbor search by using the `KFN` class in brute-force mode.

```c++
#include <mlpack.hpp>

using namespace mlpack;

// The reference set.
extern arma::mat dataset;
// The query set.
extern arma::mat querySet;

// Construct the object in brute-force mode.  We can leave the approximation
// parameter to its default (0) since brute-force will provide exact results.
KFN kfn(dataset, NAIVE_MODE);

// Perform the search for 2 furthest neighbors.
arma::Mat<size_t> neighbors;
arma::mat distances;
kfn.Search(querySet, 2, neighbors, distances);
```

## Further documentation

For further documentation on the approximate furthest neighbor facilities
offered by mlpack, see also [the NeighborSearch tutorial](nstutorial.md).  Also,
each class (`QDAFN`, `DrusillaSelect`, `NeighborSelect`) are well-documented,
and more details can be found in the source code documentation.
