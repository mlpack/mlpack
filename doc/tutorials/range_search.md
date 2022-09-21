# RangeSearch tutorial (`mlpack_range_search`)

Range search is a simple machine learning task which aims to find all the
neighbors of a point that fall into a certain range of distances.  In this
setting, we have a *query* and a *reference* dataset.  Given a certain range,
for each point in the *query* dataset, we wish to know all points in the \b
reference dataset which have distances within that given range to the given
query point.

Alternately, if the query and reference datasets are the same, the problem can
be stated more simply: for each point in the dataset, we wish to know all points
which have distance in the given range to that point.

mlpack provides:

 - a simple command-line executable to run range search
 - a simple C++ interface to perform range search
 - a generic, extensible, and powerful C++ class (`RangeSearch`) for complex
   usage

## The `mlpack_range_search` command-line executable

mlpack provides a command-line program, `mlpack_range_search`, which can be used
to perform range searches quickly and simply.  *(Note that unlike other
bindings, a range search binding is not currently available in other languages
that mlpack provides bindings to.)*

The `mlpack_range_search` program will perform the range search and place the
resulting neighbor index list into one file and their corresponding distances
into another file.  These files are organized such that the first row
corresponds to the neighbors (or distances) of the first query point, and the
second row corresponds to the neighbors (or distances) of the second query
point, and so forth.  The neighbors of a specific point are not arranged in any
specific order.

Because a range search may return different numbers of points (including zero),
the output file is technically not a valid CSV and may not be loadable by other
programs.  Therefore, if you need the results in a certain format, it may be
better to use the C++ interface to manually export the data in the preferred
format.

Below are several examples of simple usage (and the resultant output).  The `-v`
option is used so that output is given.  Further documentation on each
individual option can be found by typing

```sh
$ mlpack_range_search --help
```

### One dataset, points with distance <= 0.01

```sh
$ mlpack_range_search -r dataset.csv -n neighbors_out.csv -d distances_out.csv \
> -U 0.076 -v
[INFO ] Loading 'dataset.csv' as CSV data.  Size is 3 x 1000.
[INFO ] Loaded reference data from 'dataset.csv' (3x1000).
[INFO ] Building reference tree...
[INFO ] Tree built.
[INFO ] Search for points in the range [0, 0.076] with dual-tree kd-tree
search...
[INFO ] Search complete.
[INFO ]
[INFO ] Execution parameters:
[INFO ]   distances_file: distances_out.csv
[INFO ]   help: false
[INFO ]   info: ""
[INFO ]   input_model_file: ""
[INFO ]   leaf_size: 20
[INFO ]   max: 0.01
[INFO ]   min: 0
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
[INFO ]   loading_data: 0.005201s
[INFO ]   range_search/computing_neighbors: 0.017110s
[INFO ]   total_time: 0.033313s
[INFO ]   tree_building: 0.002500s
```

Convenient program timers are given for different parts of the calculation at
the bottom of the output, as well as the parameters the simulation was run with.
Now, if we look at the output files:

```sh
$ head neighbors_out.csv
862
703

397, 277, 319
840
732

361
547, 695
113, 982, 689

$ head distances_out.csv
0.0598608
0.0753264

0.0207941, 0.0759762, 0.0471072
0.0708221
0.0568806

0.0700532
0.0529565, 0.0550988
0.0447142, 0.0399286, 0.0734605
```

We can see that only point 862 is within distance 0.076 of point 0.  We can
also see that point 2 has no points within a distance of 0.076---that line is
empty.

### Query and reference dataset, range `[1.0, 1.5]`

```sh
$ mlpack_range_search -q query_dataset.csv -r reference_dataset.csv -n \
> neighbors_out.csv -d distances_out.csv -L 1.0 -U 1.5 -v
[INFO ] Loading 'reference_dataset.csv' as CSV data.  Size is 3 x 1000.
[INFO ] Loaded reference data from 'reference_dataset.csv' (3x1000).
[INFO ] Building reference tree...
[INFO ] Tree built.
[INFO ] Loading 'query_dataset.csv' as CSV data.  Size is 3 x 50.
[INFO ] Loaded query data from 'query_dataset.csv' (3x50).
[INFO ] Search for points in the range [1, 1.5] with dual-tree kd-tree search...
[INFO ] Building query tree...
[INFO ] Tree built.
[INFO ] Search complete.
[INFO ]
[INFO ] Execution parameters:
[INFO ]   distances_file: distances_out.csv
[INFO ]   help: false
[INFO ]   info: ""
[INFO ]   input_model_file: ""
[INFO ]   leaf_size: 20
[INFO ]   max: 1.5
[INFO ]   min: 1
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
[INFO ]   loading_data: 0.006199s
[INFO ]   range_search/computing_neighbors: 0.024427s
[INFO ]   total_time: 0.045403s
[INFO ]   tree_building: 0.003979s
```

### One dataset, range `[0.7, 0.8]`, leaf size of 15 points

The mlpack implementation of range search is a dual-tree algorithm; when
`kd`-trees are used, the leaf size of the tree can be changed.  Depending on the
characteristics of the dataset, a larger or smaller leaf size can provide faster
computation.  The leaf size is modifiable through the command-line interface, as
shown below.

```sh
$ mlpack_range_search -r dataset.csv -n neighbors_out.csv -d distances_out.csv \
> -L 0.7 -U 0.8 -l 15 -v
[INFO ] Loading 'dataset.csv' as CSV data.  Size is 3 x 1000.
[INFO ] Loaded reference data from 'dataset.csv' (3x1000).
[INFO ] Building reference tree...
[INFO ] Tree built.
[INFO ] Search for points in the range [0.7, 0.8] with dual-tree kd-tree
search...
[INFO ] Search complete.
[INFO ]
[INFO ] Execution parameters:
[INFO ]   distances_file: distances_out.csv
[INFO ]   help: false
[INFO ]   info: ""
[INFO ]   input_model_file: ""
[INFO ]   leaf_size: 15
[INFO ]   max: 0.8
[INFO ]   min: 0.7
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
[INFO ]   loading_data: 0.006298s
[INFO ]   range_search/computing_neighbors: 0.411041s
[INFO ]   total_time: 0.539931s
[INFO ]   tree_building: 0.004695s
```

Further documentation on options should be found by using the `--help` option.

## The `RangeSearch` class

The `RangeSearch` class is an extensible template class which allows a high
level of flexibility.  However, all of the template arguments have default
parameters, allowing a user to simply use `RangeSearch<>` for simple usage
without worrying about the exact necessary template parameters.

The class bears many similarities to the [`NeighborSearch`](neighbor_search.md)
class; usage generally consists of calling the constructor with one or two
datasets, and then calling the `Search()` method to perform the actual range
search.

The `Search()` method stores the results in two vector-of-vector objects.  This
is necessary because each query point may have a different number of neighbors
in the specified distance range.  The structure of those two objects is very
similar to the output files `--neighbors_file` and `--distances_file` for the
command-line interface (see above).  A handful of examples of simple usage of
the `RangeSearch` class are given below.

### Distance less than `2.0` on a single dataset

```c++
#include <mlpack.hpp>

using namespace mlpack;

// Our dataset matrix, which is column-major.
extern arma::mat data;

RangeSearch<> a(data);

// The vector-of-vector objects we will store output in.
std::vector<std::vector<size_t> > resultingNeighbors;
std::vector<std::vector<double> > resultingDistances;

// The range we will use.
Range r(0.0, 2.0); // [0.0, 2.0].

a.Search(r, resultingNeighbors, resultingDistances);
```

The output of the search is stored in `resultingNeighbors` and
`resultingDistances`.

### Range `[3.0, 4.0]` on a query and reference dataset

```c++
#include <mlpack.hpp>

using namespace mlpack;

// Our dataset matrices, which are column-major.
extern arma::mat queryData, referenceData;

RangeSearch<> a(referenceData);

// The vector-of-vector objects we will store output in.
std::vector<std::vector<size_t> > resultingNeighbors;
std::vector<std::vector<double> > resultingDistances;

// The range we will use.
Range r(3.0, 4.0); // [3.0, 4.0].

a.Search(queryData, r, resultingNeighbors, resultingDistances);
```

### Naive (exhaustive) search for distance greater than `5.0` on one dataset

This example uses the `O(n^2)` naive search (not the tree-based search).

```c++
#include <mlpack.hpp>

using namespace mlpack;

// Our dataset matrix, which is column-major.
extern arma::mat dataset;

// The 'true' option indicates that we will use naive calculation.
RangeSearch<> a(dataset, true);

// The vector-of-vector objects we will store output in.
std::vector<std::vector<size_t> > resultingNeighbors;
std::vector<std::vector<double> > resultingDistances;

// The range we will use.  The upper bound is DBL_MAX.
Range r(5.0, DBL_MAX); // [5.0, inf).

a.Search(r, resultingNeighbors, resultingDistances);
```

Needless to say, naive search can be very slow...

## The extensible `RangeSearch` class

Similar to the [`NeighborSearch` class](neighbor_search.md), the `RangeSearch`
class is very extensible, having the following template arguments:

```c++
template<typename MetricType = EuclideanDistance,
         typename MatType = arma::mat,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType = KDTree>
class RangeSearch;
```

By choosing different components for each of these template classes, a very
arbitrary range searching object can be constructed.

### `MetricType` policy class

The `MetricType` policy class allows the range search to take place in any
arbitrary metric space.  The `LMetric` class is a good example implementation.
A `MetricType` class must provide the following functions:

```c++
// Empty constructor is required.
MetricType();

// Compute the distance between two points.
template<typename VecType>
double Evaluate(const VecType& a, const VecType& b);
```

Internally, the `RangeSearch` class keeps an instantiated `MetricType` class
(which can be given in the constructor).   This is useful for a metric like the
Mahalanobis distance (`MahalanobisDistance`), which must store state (the
covariance matrix).  Therefore, you can write a non-static `MetricType` class
and use it seamlessly with `RangeSearch`.

See also the [documentation for the `MetricType`
policy](../developer/metrics.md).

### `MatType` policy class

The `MatType` template parameter specifies the type of data matrix used.  This
type must implement the same operations as an Armadillo matrix, and so standard
choices are `arma::mat` and `arma::sp_mat`.

### `TreeType` policy class

The `RangeSearch` class also allows a custom tree to be used.  The `TreeType`
policy is also used elsewhere in mlpack and is documented more thoroughly
[here](../developer/trees.md).

Typical choices might include `KDTree` (the default), `BallTree`, `RTree`,
`RStarTree`, or `StandardCoverTree`.  Below is an example that uses the
`RangeSearch` class with an R-tree:

```c++
// Construct a RangeSearch object with ball bounds.
RangeSearch<EuclideanDistance, arma::mat, RTree> rangeSearch(dataset);
```

For further information on trees, including how to write your own tree for use
with `RangeSearch` and other mlpack methods, see the [TreeType policy
documentation](../developer/trees.md).

## Further documentation

For further documentation on the `RangeSearch` class, consult the documentation
in the source code, found in `mlpack/methods/range_search/`.
