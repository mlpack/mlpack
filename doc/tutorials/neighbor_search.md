# NeighborSearch tutorial (k-nearest-neighbors)

Nearest-neighbors search is a common machine learning task.  In this setting, we
have a *query* and a *reference* dataset.  For each point in the *query*
dataset, we wish to know the `k` points in the *reference* dataset which are
closest to the given query point.

Alternately, if the query and reference datasets are the same, the problem can
be stated more simply: for each point in the dataset, we wish to know the `k`
nearest points to that point.

mlpack provides:

 - a simple command-line executable to run nearest-neighbors search (and
   furthest-neighbors search)
 - a simple C++ interface to perform nearest-neighbors search (and
   furthest-neighbors search)
 - a generic, extensible, and powerful C++ class (`NeighborSearch`) for complex
   usage

## Command-line `mlpack_knn`

The simplest way to perform nearest-neighbors search in mlpack is to use the
`mlpack_knn` executable.  *(Note that mlpack also provides bindings to other
languages, so, e.g., the `knn()` function is available in Python and Julia and
has the same options.  So, any example here can be readily adapted to another
language that mlpack provides bindings for.)*

The `mlpack_knn` program will perform nearest-neighbors search and place the
resultant neighbors into one file and the resultant distances into another.  The
output files are organized such that the first row corresponds to the nearest
neighbors of the first query point, with the first column corresponding to the
nearest neighbor, and so forth.

Below are several examples of simple usage (and the resultant output).  The `-v`
option is used so that output is given.  Further documentation on each
individual option can be found by typing

```sh
$ mlpack_knn --help
```

### One dataset, 5 nearest neighbors

```sh
$ mlpack_knn -r dataset.csv -n neighbors_out.csv -d distances_out.csv -k 5 -v
[INFO ] Loading 'dataset.csv' as CSV data.  Size is 3 x 1000.
[INFO ] Loaded reference data from 'dataset.csv' (3 x 1000).
[INFO ] Building reference tree...
[INFO ] Tree built.
[INFO ] Searching for 5 nearest neighbors with dual-tree kd-tree search...
[INFO ] 18412 node combinations were scored.
[INFO ] 54543 base cases were calculated.
[INFO ] Search complete.
[INFO ] Saving CSV data to 'neighbors_out.csv'.
[INFO ] Saving CSV data to 'distances_out.csv'.
[INFO ]
[INFO ] Execution parameters:
[INFO ]   distances_file: distances_out.csv
[INFO ]   help: false
[INFO ]   info: ""
[INFO ]   input_model_file: ""
[INFO ]   k: 5
[INFO ]   leaf_size: 20
[INFO ]   naive: false
[INFO ]   neighbors_file: neighbors_out.csv
[INFO ]   output_model_file: ""
[INFO ]   query_file: ""
[INFO ]   random_basis: false
[INFO ]   reference_file: dataset.csv
[INFO ]   seed: 0
[INFO ]   single_mode: false
[INFO ]   tree_type: kd
[INFO ]   verbose: true
[INFO ]   version: false
[INFO ]
[INFO ] Program timers:
[INFO ]   computing_neighbors: 0.108968s
[INFO ]   loading_data: 0.006495s
[INFO ]   saving_data: 0.003843s
[INFO ]   total_time: 0.126036s
[INFO ]   tree_building: 0.003442s
```

Convenient program timers are given for different parts of the calculation at
the bottom of the output, as well as the parameters the simulation was run with.
Now, if we look at the output files:

```sh
$ head neighbors_out.csv
862,344,224,43,885
703,499,805,639,450
867,472,972,380,601
397,319,277,443,323
840,827,865,38,438
732,876,751,492,616
563,222,569,985,940
361,97,928,437,79
547,695,419,961,716
982,113,689,843,634

$ head distances_out.csv
5.986076164057e-02,7.664920518084e-02,1.116050961847e-01,1.155595474371e-01,1.169810085522e-01
7.532635022982e-02,1.012564715841e-01,1.127846944644e-01,1.209584396720e-01,1.216543647014e-01
7.659571546879e-02,1.014588981948e-01,1.025114621511e-01,1.128082429187e-01,1.131659758673e-01
2.079405647909e-02,4.710724516732e-02,7.597622408419e-02,9.171977778898e-02,1.037033340864e-01
7.082206779700e-02,9.002355499742e-02,1.044181406406e-01,1.093149568834e-01,1.139700558608e-01
5.688056488896e-02,9.478072514474e-02,1.085637706630e-01,1.114177921451e-01,1.139370265105e-01
7.882260880455e-02,9.454474078041e-02,9.724494179950e-02,1.023829575445e-01,1.066927013814e-01
7.005321598247e-02,9.131417221561e-02,9.498248889074e-02,9.897964162308e-02,1.121202216165e-01
5.295654132754e-02,5.509877761894e-02,8.108227366619e-02,9.785461174861e-02,1.043968140367e-01
3.992859920333e-02,4.471418646159e-02,7.346053904990e-02,9.181982339584e-02,9.843075910782e-02
```

So, the nearest neighbor to point 0 is point 862, with a distance of
`5.986076164057e-02`.  The second nearest neighbor to point 0 is point 344, with
a distance of `7.664920518084e-02`.  The third nearest neighbor to point 5 is
point 751, with a distance of `1.085637706630e-01`.

### Query and reference dataset, 10 nearest neighbors

```sh
$ mlpack_knn -q query_dataset.csv -r reference_dataset.csv \
> -n neighbors_out.csv -d distances_out.csv -k 10 -v
[INFO ] Loading 'reference_dataset.csv' as CSV data.  Size is 3 x 1000.
[INFO ] Loaded reference data from 'reference_dataset.csv' (3 x 1000).
[INFO ] Building reference tree...
[INFO ] Tree built.
[INFO ] Loading 'query_dataset.csv' as CSV data.  Size is 3 x 50.
[INFO ] Loaded query data from 'query_dataset.csv' (3x50).
[INFO ] Searching for 10 nearest neighbors with dual-tree kd-tree search...
[INFO ] Building query tree...
[INFO ] Tree built.
[INFO ] Search complete.
[INFO ] Saving CSV data to 'neighbors_out.csv'.
[INFO ] Saving CSV data to 'distances_out.csv'.
[INFO ]
[INFO ] Execution parameters:
[INFO ]   distances_file: distances_out.csv
[INFO ]   help: false
[INFO ]   info: ""
[INFO ]   input_model_file: ""
[INFO ]   k: 10
[INFO ]   leaf_size: 20
[INFO ]   naive: false
[INFO ]   neighbors_file: neighbors_out.csv
[INFO ]   output_model_file: ""
[INFO ]   query_file: query_dataset.csv
[INFO ]   random_basis: false
[INFO ]   reference_file: reference_dataset.csv
[INFO ]   seed: 0
[INFO ]   single_mode: false
[INFO ]   tree_type: kd
[INFO ]   verbose: true
[INFO ]   version: false
[INFO ]
[INFO ] Program timers:
[INFO ]   computing_neighbors: 0.022589s
[INFO ]   loading_data: 0.003572s
[INFO ]   saving_data: 0.000755s
[INFO ]   total_time: 0.032197s
[INFO ]   tree_building: 0.002590s
```

### One dataset, 3 nearest neighbors, leaf size of 15 points

```sh
$ mlpack_knn -r dataset.csv -n neighbors_out.csv -d distances_out.csv -k 3 -l 15 -v
[INFO ] Loading 'dataset.csv' as CSV data.  Size is 3 x 1000.
[INFO ] Loaded reference data from 'dataset.csv' (3 x 1000).
[INFO ] Building reference tree...
[INFO ] Tree built.
[INFO ] Searching for 3 nearest neighbors with dual-tree kd-tree search...
[INFO ] 19692 node combinations were scored.
[INFO ] 36263 base cases were calculated.
[INFO ] Search complete.
[INFO ] Saving CSV data to 'neighbors_out.csv'.
[INFO ] Saving CSV data to 'distances_out.csv'.
[INFO ]
[INFO ] Execution parameters:
[INFO ]   distances_file: distances_out.csv
[INFO ]   help: false
[INFO ]   info: ""
[INFO ]   input_model_file: ""
[INFO ]   k: 3
[INFO ]   leaf_size: 15
[INFO ]   naive: false
[INFO ]   neighbors_file: neighbors_out.csv
[INFO ]   output_model_file: ""
[INFO ]   query_file: ""
[INFO ]   random_basis: false
[INFO ]   reference_file: dataset.csv
[INFO ]   seed: 0
[INFO ]   single_mode: false
[INFO ]   tree_type: kd
[INFO ]   verbose: true
[INFO ]   version: false
[INFO ]
[INFO ] Program timers:
[INFO ]   computing_neighbors: 0.059020s
[INFO ]   loading_data: 0.002791s
[INFO ]   saving_data: 0.002369s
[INFO ]   total_time: 0.069277s
[INFO ]   tree_building: 0.002713s
```

Further documentation on options should be found by using the `--help` option.

## The `KNN` class

The `KNN` class is, specifically, a typedef of the more extensible
`NeighborSearch` class, querying for nearest neighbors using the Euclidean
distance.

```c++
typedef NeighborSearch<NearestNeighborSort, EuclideanDistance> KNN;
```

Using the `KNN` class is particularly simple; first, the object must be
constructed and given a dataset.  Then, the method is run, and two matrices are
returned: one which holds the indices of the nearest neighbors, and one which
holds the distances of the nearest neighbors.  These are of the same structure
as the output `--neighbors_file` and `--distances_file` for the command-line
program (see above).  A handful of examples of simple usage of the KNN class are
given below.

### 5 nearest neighbors on a single dataset

```c++
#include <mlpack.hpp>

using namespace mlpack;

// Our dataset matrix, which is column-major.
extern arma::mat data;

KNN a(data);

// The matrices we will store output in.
arma::Mat<size_t> resultingNeighbors;
arma::mat resultingDistances;

a.Search(5, resultingNeighbors, resultingDistances);
```

The output of the search is stored in `resultingNeighbors` and
`resultingDistances`.

### 10 nearest neighbors on a query and reference dataset

```c++
#include <mlpack.hpp>

using namespace mlpack;

// Our dataset matrices, which are column-major.
extern arma::mat queryData, referenceData;

KNN a(referenceData);

// The matrices we will store output in.
arma::Mat<size_t> resultingNeighbors;
arma::mat resultingDistances;

a.Search(queryData, 10, resultingNeighbors, resultingDistances);
```

### Naive (exhaustive) search for 6 nearest neighbors on one dataset

This example uses the `O(n^2)` naive search (not the tree-based search).

```c++
#include <mlpack.hpp>

using namespace mlpack;

// Our dataset matrix, which is column-major.
extern arma::mat dataset;

KNN a(dataset, true);

// The matrices we will store output in.
arma::Mat<size_t> resultingNeighbors;
arma::mat resultingDistances;

a.Search(6, resultingNeighbors, resultingDistances);
```

Needless to say, naive search can be very slow...

## The extensible `NeighborSearch` class

The `NeighborSearch` class is very extensible, having the following template
arguments:

```c++
template<
  typename SortPolicy = NearestNeighborSort,
  typename MetricType = EuclideanDistance,
  typename MatType = arma::mat,
  template<typename TreeMetricType,
           typename TreeStatType,
           typename TreeMatType> class TreeType = KDTree,
  template<typename RuleType> class TraversalType =
      TreeType<MetricType, NeighborSearchStat<SortPolicy>,
               MatType>::template DualTreeTraverser>
>
class NeighborSearch;
```

By choosing different components for each of these template classes, a very
arbitrary neighbor searching object can be constructed.  Note that each of these
template parameters have defaults, so it is not necessary to specify each one.

### `SortPolicy` policy class

The `SortPolicy` template parameter allows specification of how the
NeighborSearch object will decide which points are to be searched for.  The
`NearestNeighborSort` class is a well-documented example.  A custom `SortPolicy`
class must implement the same methods which `NearestNeighborSort` does:

```c++
static size_t SortDistance(const arma::vec& list, double newDistance);

static bool IsBetter(const double value, const double ref);

template<typename TreeType>
static double BestNodeToNodeDistance(const TreeType* queryNode,
                                     const TreeType* referenceNode);

template<typename TreeType>
static double BestPointToNodeDistance(const arma::vec& queryPoint,
                                      const TreeType* referenceNode);

static const double WorstDistance();

static const double BestDistance();
```

The `FurthestNeighborSort` class is another implementation, which is used to
create the `KFN` typedef class, which finds the furthest neighbors, as opposed
to the nearest neighbors.

## `MetricType` policy class

The `MetricType` policy class allows the neighbor search to take place in any
arbitrary metric space.  The `LMetric` class is a good example implementation.
A `MetricType` class must provide the following functions:

```c++
// Empty constructor is required.
MetricType();

// Compute the distance between two points.
template<typename VecType>
double Evaluate(const VecType& a, const VecType& b);
```

Internally, the `NeighborSearch` class keeps an instantiated `MetricType` class
(which can be given in the constructor).   This is useful for a metric like the
Mahalanobis distance (`MahalanobisDistance`), which must store state (the
covariance matrix).  Therefore, you can write a non-static MetricType class and
use it seamlessly with `NeighborSearch`.

For more information on the `MetricType` policy, see the [documentation for
`MetricType`s](../developer/metrics.md).

### `MatType` policy class

The `MatType` template parameter specifies the type of data matrix used.  This
type must implement the same operations as an Armadillo matrix, and so standard
choices are `arma::mat` and `arma::sp_mat`.

### `TreeType` policy class

The NeighborSearch class allows great extensibility in the selection of the type
of tree used for search.  This type must follow the typical mlpack TreeType
policy, documented [here](../developer/trees.md).

Typical choices might include `KDTree`, `BallTree`, `StandardCoverTree`,
`RTree`, or `RStarTree`.  It is easily possible to make your own tree type for
use with NeighborSearch; consult the [TreeType
documentation](../developer/trees.md) for more details.

An example of using the `NeighborSearch` class with a ball tree is given below.

```c++
// Construct a NeighborSearch object with ball bounds.
NeighborSearch<
    NearestNeighborSort,
    EuclideanDistance,
    arma::mat,
    BallTree
> neighborSearch(dataset);
```

### `TraverserType` policy class

The last template parameter the `NeighborSearch` class offers is the
`TraverserType` class.  The `TraverserType` class holds the strategy used to
traverse the trees in either single-tree or dual-tree search mode.  By default,
it is set to use the default traverser of the given `TreeType` (which is the
member `TreeType::DualTreeTraverser`).

This class must implement the following two methods:

```c++
// Instantiate with a given RuleType.
TraverserType(RuleType& rule);

// Traverse with two trees.
void Traverse(TreeType& queryNode, TreeType& referenceNode);
```

The `RuleType` class provides the following functions for use in the traverser:

```c++
// Evaluate the base case between two points.
double BaseCase(const size_t queryIndex, const size_t referenceIndex);

// Score the two nodes to see if they can be pruned, returning DBL_MAX if they
// can be pruned.
double Score(TreeType& queryNode, TreeType& referenceNode);
```

Note also that any traverser given must satisfy the definition of a pruning
dual-tree traversal given in the paper "Tree-independent dual-tree algorithms".

## Further documentation

For further documentation on the NeighborSearch class, consult the comments in
the source code, found in `mlpack/methods/neighbor_search/`.
