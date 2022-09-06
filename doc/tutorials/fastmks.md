# Fast max-kernel search tutorial (FastMKS)

The FastMKS algorithm (fast exact max-kernel search) is a recent algorithm
proposed in the following papers:

```
@inproceedings{curtin2013fast,
  title={Fast Exact Max-Kernel Search},
  author={Curtin, Ryan R. and Ram, Parikshit and Gray, Alexander G.},
  booktitle={Proceedings of the 2013 SIAM International Conference on Data
      Mining (SDM '13)},
  year={2013},
  pages={1--9}
}

@article{curtin2014dual,
  author = {Curtin, Ryan R. and Ram, Parikshit},
  title = {Dual-tree fast exact max-kernel search},
  journal = {Statistical Analysis and Data Mining},
  volume = {7},
  number = {4},
  publisher = {Wiley Subscription Services, Inc., A Wiley Company},
  issn = {1932-1872},
  url = {http://dx.doi.org/10.1002/sam.11218},
  doi = {10.1002/sam.11218},
  pages = {229--253},
  year = {2014},
}
```

Given a set of query points `Q` and a set of reference points `R`, the FastMKS
algorithm is a fast dual-tree (or single-tree) algorithm which finds

```
argmax_{p_r in R} K(p_q, p_r)
```

for all points `p_q` in `Q` and for some Mercer kernel `K()`.  A Mercer kernel
is a kernel that is positive semidefinite; these are the classes of kernels that
can be used with the kernel trick.  In short, the positive semidefiniteness of a
Mercer kernel means that any kernel matrix (or Gram matrix) created on a dataset
must be positive semidefinite.

The FastMKS algorithm builds trees on the datasets `Q` and `R` in such a way
that explicit representation of the points in the kernel space is unnecessary,
by using cover trees (see `CoverTree`).  This allows the algorithm to be run,
for instance, on string kernels, where there is no sensible explicit
representation.  The mlpack implementation allows any type of tree that does not
require an explicit representation to be used.  For more details, see the paper.

At the time of this writing there is no other fast algorithm for exact
max-kernel search.  mlpack implements both single-tree and dual-tree fast
max-kernel search.

mlpack provides:

 - a simple command-line executable to run FastMKS
 - a C++ interface to run FastMKS

## Command-line FastMKS (`mlpack_fastmks`)

mlpack provides a command-line program, `mlpack_fastmks`, which is used to
perform FastMKS on a given query and reference dataset.  It supports numerous
different types of kernels:

 - `LinearKernel`
 - `PolynomialKernel`
 - `CosineDistance`
 - `GaussianKernel`
 - `EpanechnikovKernel`
 - `TriangularKernel`
 - `HyperbolicTangentKernel`

Note that when a shift-invariant kernel is used, the results will be the same as
nearest neighbor search, so [KNN](neighbor_search.md) may be a better option.  A
shift-invariant kernel is a kernel that depends only on the distance between the
two input points.  The `GaussianKernel`, `EpanechnikovKernel`, and
`TriangularKernel` are instances of shift-invariant kernels.  The paper contains
more details on this situation.  The `mlpack_fastmks` executable still provides
these kernels as options, though.

The following examples detail usage of the `mlpack_fastmks` program.  Note that
you can get documentation on all the possible parameters by typing:

```sh
$ mlpack_fastmks --help
```

### FastMKS with a linear kernel on one dataset

If only one dataset is specified (with `-r` or `--reference_file`), the
reference dataset is taken to be both the query and reference datasets.  The
example below finds the 4 maximum kernels of each point in `dataset.csv`, using
the default linear kernel.

```sh
$ mlpack_fastmks -r dataset.csv -k 4 -v -p products.csv -i indices.csv
```

When the operation completes, the values of the kernels are saved in
`products.csv` and the indices of the points which give the maximum kernels are
saved in `indices.csv`.

```sh
$ head indices.csv
762,910,863,890
762,910,426,568
910,762,863,426
762,910,863,426
863,910,614,762
762,863,910,614
762,910,488,568
762,910,863,426
910,762,863,426
863,762,910,614
```

```sh
$ head products.csv
1.6221652894e+00,1.5998743443e+00,1.5898890769e+00,1.5406789753e+00
1.3387953449e+00,1.3317349486e+00,1.2966613184e+00,1.2774493620e+00
1.6386110476e+00,1.6332029753e+00,1.5952629124e+00,1.5887195330e+00
1.0917545803e+00,1.0820878726e+00,1.0668992636e+00,1.0419838050e+00
1.2272441028e+00,1.2169643942e+00,1.2104597963e+00,1.2067780154e+00
1.5720962456e+00,1.5618504956e+00,1.5609069923e+00,1.5235605095e+00
1.3655478674e+00,1.3548593212e+00,1.3311547298e+00,1.3250728881e+00
2.0119149744e+00,2.0043668067e+00,1.9847289214e+00,1.9298280046e+00
1.1586923205e+00,1.1494586097e+00,1.1274872962e+00,1.1248172766e+00
4.4789820372e-01,4.4618539778e-01,4.4200024852e-01,4.3989721792e-01
```

We can see in this example that for point 0, the point with maximum kernel value
is point 762, with a kernel value of 1.622165.  For point 3, the point with
third largest kernel value is point 863, with a kernel value of 1.0669.

### FastMKS on a reference and query dataset

The query points may be different than the reference points.  To specify a
different query set, the `-q` (or `--query_file`) option is used, as in the
example below.

```sh
$ mlpack_fastmks -q query_set.csv -r reference_set.csv -k 5 -i indices.csv \
> -p products.csv
```

### FastMKS with a different kernel

The `mlpack_fastmks` program offers more than just the linear kernel.  Valid
options are `'linear'`, `'polynomial'`, `'cosine'`, `'gaussian'`,
`'epanechnikov'`, `'triangular'` and `'hyptan'` (the hyperbolic tangent kernel).
Note that the hyperbolic tangent kernel is provably not a Mercer kernel but is
positive semidefinite on most datasets and is commonly used as a kernel.  Note
also that the Gaussian kernel and other shift-invariant kernels give the same
results as nearest neighbor search (see [the tutorial](neighbor_search.md)).

The kernel to use is specified with the `-K` (or `--kernel`) option.  The
example below uses the cosine similarity as a kernel.

```sh
$ mlpack_fastmks -r dataset.csv -k 5 -K cosine -i indices.csv -p products.csv -v
```

### Using single-tree search or naive search

In some cases, it may be useful to not use the dual-tree FastMKS algorithm.
Instead you can specify the `--single` option, indicating that a tree should be
built only on the reference set, and then the queries should be processed in a
linear scan (instead of in a tree).  Alternately, the `-N` (or `--naive`) option
makes the program not build trees at all and instead use brute-force search to
find the solutions.

The example below uses single-tree search on two datasets with the linear
kernel.

```sh
$ mlpack_fastmks -q query_set.csv -r reference_set.csv --single -k 5 \
> -p products.csv -i indices.csv -K linear
```

The example below uses naive search on one dataset.

```sh
$ mlpack_fastmks -r reference_set.csv -k 5 -N -p products.csv -i indices.csv
```

### Parameters for alternate kernels

Many of the alternate kernel choices have parameters which can be chosen; these
are detailed in this section.

 - `-w` (`--bandwidth`): this sets the bandwidth of the kernel, and is
   applicable to the `'gaussian'`, `'epanechnikov'`, and `'triangular'` kernels.
   This is the "spread" of the kernel.

 - `-d` (`--degree`): this sets the degree of the polynomial kernel (the power
   to which the result is raised).  It is only applicable to the `'polynomial'`
   kernel.

 - `-o` (`--offset`): this sets the offset of the kernel, for the
   `'polynomial'` and `'hyptan'` kernel.  See the documentation for
   `PolynomialKernel` and `HyperbolicTangentKernel` for more information.

 - `-s` (`--scale`): this sets the scale of the kernel, and is only applicable
   to the `'hyptan'` kernel.  See the documentation for
   `HyperbolicTangentKernel` for more information.

### Saving a FastMKS model/tree

The `mlpack_fastmks` program also supports saving a model built on a reference
dataset (this model includes the tree, the kernel, and the search parameters).
The `--output_model_file` or `-M` option allows one to save these parameters to
disk for later usage.  An example is below:

```sh
$ mlpack_fastmks -r reference_set.csv -K cosine -M fastmks_model.xml
```

This example builds a tree on the dataset in `reference_set.csv` using the
cosine similarity kernel, and saves the resulting model to `fastmks_model.xml`.
This model may then be used in later calls to the `mlpack_fastmks` program.

### Loading a FastMKS model for further searches

Supposing that a FastMKS model has been saved with the `--output_model_file` or
`-M` parameter, that model can then be later loaded in subsequent calls to the
`mlpack_fastmks` program, using the `--input_model_file` or `-m` option.  For
instance, with a model saved in `fastmks_model.xml` and a query set in
`query_set.csv`, we can find 3 max-kernel candidates, saving to `indices.csv`
and `kernels.csv`:

```sh
$ mlpack_fastmks -m fastmks_model.xml -k 3 -i indices.csv -p kernels.csv
```

Loading a model as opposed to building a model is advantageous because the
reference tree is already built.  So, among other situations, this could be
useful in the setting where many different query sets (or many different values
of `k`) will be used.

Note that the kernel cannot be changed in a saved model without rebuilding the
model entirely.

## The `FastMKS` class

The `FastMKS<>` class offers a simple API for use within C++ applications, and
allows further flexibility in kernel choice and tree type choice.  However,
`FastMKS<>` has no default template parameter for the kernel type---that must be
manually specified.  Choices that mlpack provides include:

 - `LinearKernel`
 - `PolynomialKernel`
 - `CosineDistance`
 - `GaussianKernel`
 - `EpanechnikovKernel`
 - `TriangularKernel`
 - `HyperbolicTangentKernel`
 - `LaplacianKernel`
 - `PSpectrumStringKernel`

The following examples use kernels from that list.  Writing your own kernel is
detailed in the next section.  Remember that when you are using the C++
interface, the data matrices must be column-major.  See the [matrices
documentation](../user/matrices.md) for more information.

### `FastMKS` on one dataset

Given only a reference dataset, the following code will run FastMKS with k set
to 5.

```c++
#include <mlpack.hpp>

using namespace mlpack::fastmks;

// The reference dataset, which is column-major.
extern arma::mat data;

// This will initialize the FastMKS object with the linear kernel with default
// options: K(x, y) = x^T y.  The tree is built in the constructor.
FastMKS<LinearKernel> f(data);

// The results will be stored in these matrices.
arma::Mat<size_t> indices;
arma::mat products;

// Run FastMKS.
f.Search(5, indices, products);
```

### FastMKS with a query and reference dataset

In this setting we have both a query and reference dataset.  We search for 10
maximum kernels.

```
#include <mlpack.hpp>

using namespace mlpack::fastmks;
using namespace mlpack::kernel;

// The reference and query datasets, which are column-major.
extern arma::mat referenceData;
extern arma::mat queryData;

// This will initialize the FastMKS object with the triangular kernel with
// default options (bandwidth of 1).  The reference tree is built in the
// constructor.
FastMKS<TriangularKernel> f(referenceData);

// The results will be stored in these matrices.
arma::Mat<size_t> indices;
arma::mat products;

// Run FastMKS.  The query tree is built during the call to Search().
f.Search(queryData, 10, indices, products);
```

### FastMKS with an initialized kernel

Often, kernels have parameters which need to be specified.  `FastMKS<>` has
constructors which take initialized kernels.  Note that temporary kernels cannot
be passed as an argument.  The example below initializes a `PolynomialKernel`
object and then runs FastMKS with a query and reference dataset.

```c++
#include <mlpack.hpp>

using namespace mlpack::fastmks;
using namespace mlpack::kernel;

// The reference and query datasets, which are column-major.
extern arma::mat referenceData;
extern arma::mat queryData;

// Initialize the polynomial kernel with degree of 3 and offset of 2.5.
PolynomialKernel pk(3.0, 2.5);

// Create the FastMKS object with the initialized kernel.
FastMKS<PolynomialKernel> f(referenceData, pk);

// The results will be stored in these matrices.
arma::Mat<size_t> indices;
arma::mat products;

// Run FastMKS.
f.Search(queryData, 10, indices, products);
```

The syntax for running FastMKS with one dataset and an initialized kernel is
very similar:

```c++
f.Search(10, indices, products);
```

### FastMKS with an already-created tree

By default, `FastMKS<>` uses the cover tree datastructure (see the `CoverTree`
documentation).  Sometimes, it is useful to modify the parameters of the cover
tree.  In this scenario, a tree must be built outside of the constructor, and
then passed to the appropriate `FastMKS<>` constructor.  An example on just a
reference dataset is shown below, where the base of the cover tree is modified.

We also use an instantiated kernel, but because we are building our own tree, we
must use `IPMetric` so that our tree is built on the metric induced by our
kernel function.

```c++
#include <mlpack.hpp>

// The reference dataset, which is column-major.
extern arma::mat data;

// Initialize the polynomial kernel with a degree of 4 and offset of 2.0.
PolynomialKernel pk(4.0, 2.0);

// Create the metric induced by this kernel (because a kernel is not a metric
// and we can't build a tree on a kernel alone).
IPMetric<PolynomialKernel> metric(pk);

// Now build a tree on the reference dataset using the instantiated metric and
// the custom base of 1.5 (default is 1.3).  We have to be sure to use the right
// type here -- FastMKS needs the FastMKSStat object as the tree's
// StatisticType.
typedef CoverTree<IPMetric<PolynomialKernel>, FirstPointIsRoot, FastMKSStat>
    TreeType; // Convenience typedef.
TreeType* tree = new TreeType(data, metric, 1.5);

// Now initialize FastMKS with that statistic.  We don't need to specify the
// TreeType template parameter since we are still using the default.  We don't
// need to pass the kernel because that is contained in the tree.
FastMKS<PolynomialKernel> f(tree);

// The results will be stored in these matrices.
arma::Mat<size_t> indices;
arma::mat products;

// Run FastMKS.
f.Search(10, indices, products);
```

The syntax is similar for the case where different query and reference datasets
are given; but trees for both need to be built in the manner specified above.
Be sure to build both trees using the same metric (or at least a metric with the
exact same parameters).

```c++
f.Search(queryTree, 10, indices, products);
```

### Writing a custom kernel for FastMKS

While mlpack provides some number of kernels in the `mlpack::kernel` namespace,
it is easy to create a custom kernel.  To satisfy the [KernelType
policy](../developer/kernels.md), a class must implement the following methods:

```c++
// Empty constructor is required.
KernelType();

// Evaluate the kernel between two points.
template<typename VecType>
double Evaluate(const VecType& a, const VecType& b);
```

The template parameter `VecType` is helpful (but not necessary) so that the
kernel can be used with both sparse and dense matrices (`arma::sp_mat` and
`arma::mat`).

### Using other tree types for FastMKS

The use of the cover tree is not necessary for FastMKS, although it is the
default tree type.  A different type of tree can be specified with the TreeType
template parameter.  However, the tree type is required to have
`mlpack::fastmks::FastMKSStat` as the `StatisticType`, and for FastMKS to work,
the tree must be built only on kernel evaluations (or distance evaluations in
the kernel space via `IPMetric::Evaluate()`).

Below is an example where a custom tree class, `CustomTree`, is used as the
tree type for FastMKS.  In this example FastMKS is only run on one dataset.

```c++
#include <mlpack.hpp>
#include "custom_tree.hpp"

using namespace mlpack::fastmks;
using namespace mlpack::tree;

// The dataset that FastMKS will be run on.
extern arma::mat data;

// The custom tree type.  We'll assume that the first template parameter is the
// statistic type.
typedef CustomTree<FastMKSStat> TreeType;

// The FastMKS constructor will create the tree.
FastMKS<LinearKernel, arma::mat, TreeType> f(data);

// These will hold the results.
arma::Mat<size_t> indices;
arma::mat products;

// Run FastMKS.
f.Search(5, indices, products);
```

### Running FastMKS on objects

FastMKS has a lot of utility on objects which are not representable in some sort
of metric space.  These objects might be strings, graphs, models, or other
objects.  For these types of objects, questions based on distance don't really
make sense.  One good example is with strings.  The question "how far is 'dog'
from 'Taki Inoue'?" simply doesn't make sense.  We can't have a centroid of the
terms 'Fritz', 'E28', and 'popsicle'.

However, what we can do is define some sort of kernel on these objects.  These
kernels generally correspond to some similarity measure, with one example being
the p-spectrum string kernel (see `PSpectrumStringKernel`).  Using that, we can
say "how similar is 'dog' to 'Taki Inoue'?" and get an actual numerical result
by evaluating `K('dog', 'Taki Inoue')` (where `K` is our p-spectrum string
kernel).

The only requirement on these kernels is that they are positive definite kernels
(or Mercer kernels).  For more information on those details, refer to the
FastMKS paper.

Remember that FastMKS is a tree-based method.  But trees like the binary space
tree require centroids---and as we said earlier, centroids often don't make
sense with these types of objects.  Therefore, we need a type of tree which is
built *exclusively* on points in the dataset---those are points which we can
evaluate our kernel function on.  The cover tree is one example of a type of
tree satisfying this condition; its construction will only call the kernel
function on two points that are in the dataset.

But, we have one more problem.  The `CoverTree` class is built on `arma::mat`
objects (dense matrices).  Our objects, however, are not necessarily
representable in a column of a matrix.  To use the example we have been using,
strings cannot be represented easily in a matrix because they may all have
different lengths.

The way to work around this problem is to create a "fake" data matrix which
simply holds indices to objects.  A good example of how to do this is detailed
in the documentation for the `PSpectrumStringKernel` class.

In short, the trick is to make each data matrix one-dimensional and containing
linear indices:

```c++
arma::mat data = "0 1 2 3 4 5 6 7 8";
```

Then, when `Evaluate()` is called on the kernel function, the parameters will be
two one-dimensional vectors that simply contain indices to objects.  The example
below details the process a little better:

```c++
// This function evaluates the kernel on two Objects (in this example, its
// implementation is not important; the only important thing is that the
// function exists).
double ObjectKernel::Evaluate(const Object& a, const Object& b) const;

template<typename VecType>
double ObjectKernel::Evaluate(const VecType& a, const VecType& b) const
{
  // Extract the indices from the vectors.
  const size_t indexA = size_t(a[0]);
  const size_t indexB = size_t(b[0]);

  // Assume that 'objects' is an array (or std::vector or other container)
  // holding Objects.
  const Object& objectA = objects[indexA];
  const Object& objectB = objects[indexB];

  // Now call the function that does the actual evaluation on the objects and
  // return its result.
  return Evaluate(objectA, objectB);
}
```

As written earlier, the documentation for `PSpectrumStringKernel` is a good
place to consult for further reference on this.  That kernel uses two
dimensional indices; one dimension represents the index of the string, and the
other represents whether it is referring to the query set or the reference set.
If your kernel is meant to work on separate query and reference sets, that
strategy should be considered.

## Further documentation

For further documentation on the FastMKS class, consult the documentation in the
source code for FastMKS, in `mlpack/methods/fastmks/`.
