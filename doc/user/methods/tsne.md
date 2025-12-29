## `TSNE`: t-Distributed Stochastic Neighbor Embedding

The `TSNE` class implements t-distributed Stochastic Neighbor Embedding
(t-SNE), a nonlinear dimensionality reduction technique designed mainly for
visualization of high-dimensional datasets. It captures pairwise
similarities in the high-dimensional space and finds a low-dimensional
representation that preserves them.

The t-SNE algorithm works in two stages. First, it builds a probability
distribution over pairs of points in the high-dimensional space using a
Gaussian kernel, assigning higher probabilities to closer points and lower
probabilities to farther ones. Second, it models pairwise similarities in
the low-dimensional space with a heavy-tailed Student's t-distribution and
adjusts point positions by minimizing the Kullback-Leibler (KL) divergence 
between the two distributions.

This implementation supports multiple gradient computation methods.
(see [Advanced Functionality](#advanced-functionality-template-parameters) for more details)

#### Simple usage example

```c++
// Use t-SNE to reduce the number of dimensions to 2 on some dataset.
// Replace with a data::Load() call or similar for a real application.
arma::mat dataset(10, 1000, arma::fill::randu);

// Step 1: Create TSNE object.
mlpack::TSNE tsne;

// Step 2: Embed the dataset into two dimensions.
arma::mat output;
tsne.Embed(dataset, output);

// Print some information about the modified dataset.
std::cout << "The transformed data matrix has size ";
std::cout << output.n_rows << " x " << output.n_cols << "." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links

 * [Constructors](#constructors): create `TSNE` objects.
 * [`Embed()`](#embed): embed data into a lower-dimensional space.
 * [Examples](#simple-examples) of simple usage and links to detailed examples.
 * [Template parameters](#advanced-functionality-template-parameters) for using different gradient computation methods, matrix types, and distance metrics.
 * [Advanced examples](#advanced-examples) with custom template parameters.

#### See also

 * [`PCA`](pca.md): principal components analysis
 * [mlpack transformations](../transformations.md)
 * [mlpack preprocessing utilities](../preprocessing.md)
 * [t-distributed Stochastic Neighbor Embedding on Wikipedia](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)

### Constructors

* `tsne = TSNE(outputDims=2, perplexity=30.0, exaggeration=12.0, stepSize=0.0, maxIter=1000, tolerance=1e-12, init="pca", theta=0.5)`
  - Construct a `TSNE` object with the given parameters.

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `outputDims` | `size_t` | Dimensionality of the embedded space. | `2` |
| `perplexity` | `double` | Regulates the balance between local and global structure preservation. Typically set between 5 and 50. | `30.0` |
| `exaggeration` | `double` | Amplifies pairwise similarities during the initial optimization phase. This helps form tighter clusters and clearer separation between them. A higher value increases spacing between clusters, but if the cost grows during initial iterations consider reducing this value or lowering the step size. | `12.0` |
| `stepSize` | `double` | Step size (learning rate) for the optimizer. If the specified value is zero or negative, the step size is set to `max(50.0, N / exaggeration / 4.0)`, where `N` is number of points in the dataset. | `0.0` |
| `maxIter` | `size_t` | Maximum number of iterations for the optimizer. | `1000` |
| `tolerance` | `double` | Minimum improvement in the objective value required to perform another iteration. | `1e-12` |
| `init` | `std::string` | Initialization method for the embedding. Options: `"random"`, `"pca"`. PCA initialization is recommended for speed and quality. | `"pca"` |
| `theta` | `double` | Regulates the trade-off between speed and accuracy for the `BarnesHutTSNE` and `DualTreeTSNE` methods. Higher values of theta result in coarser approximations. The optimal value depends on the chosen methods. | `0.5` |

### Embed

* `double klDivergence = tsne.Embed(X, Y)`
  * Embed the [column-major matrix](../matrices.md#representing-data-in-mlpack) `X` into a lower-dimensional space, storing the result in `Y`.
  * `X` should be a floating-point matrix (e.g. `arma::mat`, `arma::fmat`,
     etc.) or an expression that evaluates to one.
  * `Y` will be overwritten with the output embedding, and will have
     `outputDims` rows and the same number of columns as `X`.
  * Returns the final objective value (Kullback-Leibler divergence).

---

### Simple examples

See also the [simple usage example](#simple-usage-example) for a trivial
usage of the `TSNE` class.

---

Embed a dataset into three dimensions using the dual-tree method.

```c++
// See https://datasets.mlpack.org/iris.csv.
arma::mat data;
mlpack::data::Load("iris.csv", data, true);

mlpack::TSNE<mlpack::DualTreeTSNE> tsne(3);

arma::mat output;
tsne.Embed(data, output);

mlpack::data::Save("iris_embedded.csv", output, true);
```

---

Embed a dataset with all parameters explicitly specified.

```c++
// See https://datasets.mlpack.org/satellite.train.csv.
arma::mat data;
mlpack::data::Load("satellite.train.csv", data, true);

mlpack::TSNE<mlpack::BarnesHutTSNE> tsne(
    2,      // outputDims
    30.0,   // perplexity
    12.0,   // exaggeration
    0.0,    // stepSize (0.0 means automatic)
    1000,   // maxIter
    1e-12,  // tolerance
    "pca",  // init
    0.5     // theta
);

arma::mat output;
tsne.Embed(data, output);

mlpack::data::Save("satellite.train.2d.csv", output, true);
```

---

Compare the performance of different gradient computation methods on the MNIST dataset.

```c++
arma::mat data;
// See https://datasets.mlpack.org/mnist.train.csv.
// Note: this is a large dataset and may take a while.
mlpack::data::Load("mnist.train.csv", data, true);

arma::mat output1, output2, output3;

mlpack::TSNE<mlpack::BarnesHutTSNE> tsne1;
mlpack::TSNE<mlpack::DualTreeTSNE> tsne2;
mlpack::TSNE<mlpack::ExactTSNE> tsne3;

arma::wall_clock c;

c.tic();
tsne1.Embed(data, output1);
const double tsne1Time = c.toc();

c.tic();
tsne2.Embed(data, output2);
const double tsne2Time = c.toc();

c.tic();
tsne3.Embed(data, output3);
const double tsne3Time = c.toc();

std::cout << "t-SNE computation times for ";
std::cout << data.n_rows << " x " << data.n_cols << " data:" << std::endl;
std::cout << " - BarnesHutTSNE: " << tsne1Time << "s." << std::endl;
std::cout << " - DualTreeTSNE:  " << tsne2Time << "s." << std::endl;
std::cout << " - ExactTSNE:     " << tsne3Time << "s." << std::endl;
```

### Advanced functionality: template parameters

The `TSNE` class has three template parameters that can be used for custom behavior.  The full signature of the class is:

```
TSNE<TSNEMethod, MatType, DistanceType>
```

 * `TSNEMethod`: specifies the method to be used to compute the t-SNE gradient.
 * `MatType`: specifies the type of matrix used for representation of data.
 * `DistanceType`: specifies the [distance metric](../core/distances.md) to be
   used.

---

#### `TSNEMethod`

The following methods are already implemented and ready for drop-in usage:

* `BarnesHutTSNE` *(default)*

  * Approximates repulsive forces using the Barnes-Hut algorithm applied in the embedding space. The attractive forces are approximated by only considering a fixed number of nearest neighbors in the input space.
  * Time and memory complexity are O(NlogN), where N is the number of data points.
  * Provides a strong empirical trade-off between runtime efficiency and embedding quality.
  * Recommended for medium to large datasets and represents the standard choice in most practical applications.

* `DualTreeTSNE`

  * Uses a dual tree traversal to approximate gradient computation by jointly pruning interactions between groups of points.
  * Complexity is O(NlogN) in both time and memory.
  * Can outperform Barnes Hut in some regimes, particularly for large datasets where effective pruning can be achieved.

* `ExactTSNE`

  * Computes the exact gradient of the t-SNE objective by evaluating all pairwise interactions between points.
  * Complexity is O(N^2) in both time and memory.
  * Produces the reference solution for t-SNE optimization, with no approximation error introduced by the method itself.
  * Becomes computationally impractical beyond small datasets, typically on the order of a few thousand points.

---

#### `MatType`

 * Specifies the type of matrix to use for representing data (the input and output).
 * This type must implement the same operations as an Armadillo matrix, and should be a floating-point matrix. (e.g. `arma::mat`, `arma::fmat`, etc.)

---

#### `DistanceType`

 * Specifies the distance metric that will be used when calculating similarities in the high-dimensional space.
 * The default distance type is [`SquaredEuclideanDistance`](../core/distances.md#lmetric).

### Advanced examples

Use `TSNE` with `float` precision (saving memory).

```c++
// See https://datasets.mlpack.org/iris.csv.
arma::fmat dataset;
mlpack::data::Load("iris.csv", dataset);

// Construct a TSNE object using the dual-tree method with float MatType.
mlpack::TSNE<mlpack::DualTreeTSNE, arma::fmat> tsne;

// Configure parameters if needed.
tsne.Perplexity() = 20.0;
tsne.MaximumIterations() = 500;

arma::fmat output;
tsne.Embed(dataset, output);

std::cout << "Embedding complete. Output size: " 
          << output.n_rows << " x " << output.n_cols << std::endl;
```