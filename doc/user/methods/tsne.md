## `TSNE`

The `TSNE` class implements t-distributed Stochastic Neighbor Embedding
(t-SNE), a nonlinear dimensionality reduction technique designed mainly for
visualization of high-dimensional datasets. It captures pairwise similarities in the high-dimensional space
and finds a low-dimensional representation that preserves them.

The t-SNE algorithm works in two stages. First, it builds a probability
distribution over pairs of points in the high-dimensional space using a
Gaussian kernel, assigning higher probabilities to closer points and lower
probabilities to farther ones. Second, it models pairwise similarities in the
low-dimensional space with a heavy-tailed Student’s t-distribution and adjusts
point positions by minimizing the Kullback-Leibler (KL) divergence between the
two distributions.

This implementation provides multiple methods for computing the KL
divergence and its gradient. (see [methods](#methods) for more detail)

#### Simple usage example

```c++
// Use t-SNE to reduce the number of dimensions to 2 on some dataset.
// Replace with a data::Load() call or similar for a real application.
arma::mat dataset(10, 1000, arma::fill::randu); // 1000 points in 10d.

// Step 1: create TSNE object (using defaults).
mlpack::TSNE<> tsne;

// Step 2: embed data into 2 dimensions.
arma::mat output;
tsne.Embed(dataset, output);

// Print some information about the modified dataset.
std::cout << "The transformed data matrix has size ";
std::cout << output.n_rows << " x " << output.n_cols << "." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#examples">More examples...</a></p>

#### Quick links

- [Constructors](#constructors): create `TSNE` objects.
- [`Embed()`](#embed): embed data into a lower-dimensional space.
- [Examples](#examples) of simple usage and links to detailed example
   projects.
- [Template parameters](#methods) for using different gradient computation
   methods.

#### See also

- [mlpack transformations](../transformations.md)
- [`PCA`](pca.md): principal components analysis
- [t-distributed Stochastic Neighbor Embedding on Wikipedia](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)

### Constructors

- `tsne = TSNE(outputDim=2, perplexity=30.0, exaggeration=12.0, stepSize=200.0, maxIter=1000, tolerance = 1e-12, init="pca", theta=0.5)`
  - `outputDim`: Dimensionality of the embedded space. *(Default: 2)*  
  - `perplexity`: Regulates the balance between local and global structure preservation. Typically set between 5 and 50. *(Default: 30.0)*  
  - `exaggeration`: Amplifies pairwise similarities during the initial optimization phase. This helps form tighter clusters and clearer separation between them. A higher value increases spacing between clusters, but if the cost grows during initial iterations consider reducing this value or lowering the step size. *(Default: 12.0)*  
  - `stepSize`: Step size (learning rate) for the optimizer. If the specified value is `0`, the step size is computed as number of points divided by exaggeration every time `Embed` is called. *(Default: 200.0)*  
  - `maxIter`: Maximum number of iterations. *(Default: 1000)*  
  - `tolerance`: Minimum improvement in the objective value required to perform another iteration. *(Default: 1e-12)*
  - `init`: Initialization method for the embedding. Options: `"random"`, `"pca"`. PCA initialization is recommended for speed and quality. *(Default: `"pca"`)*  
  - `theta`: Regulates the trade-off between speed and accuracy for the `barnes-hut` and `dual-tree` methods. Higher values of theta result in coarser approximations, and the optimal value depends on the chosen methods (Default: 0.5)*  

---

### Embed

- `tsne.Embed(X, Y)`
  - Embed the [column-major matrix](../matrices.md#representing-data-in-mlpack) `X`
     into a lower-dimensional space, storing the result in `Y`.
  - `X` should be a floating-point matrix (e.g. `arma::mat`, `arma::fmat`,
     etc.) or an expression that evaluates to one.
  - `Y` will be overwritten with the output embedding, and will have
     `outputDim` rows.

---

### Examples

See also the [simple usage example](#simple-usage-example) for a trivial usage
of the `TSNE` class.

---

Embed a dataset into 3 dimensions using the dual-tree method.

```c++
// See https://datasets.mlpack.org/iris.csv.
arma::mat data;
mlpack::data::Load("satellite.train.csv", data, true);

// Use the dual-tree approximation and embed to 3 dimensions.
mlpack::TSNE<mlpack::DualTreeTSNE> tsne(3);

arma::mat output;
tsne.Embed(data, output);

mlpack::data::Save("satellite.train.3d.csv", output, true);
```

---

Embed a dataset, with custom parameters.

```c++
// See https://datasets.mlpack.org/satellite.train.csv.
arma::mat data;
mlpack::data::Load("satellite.train.csv", data, true);

// Specify a perplexity of 50 and 1500 iterations.
mlpack::TSNE<> tsne(2, 50.0, 12.0, 200.0, 1500);

arma::mat output;
tsne.Embed(data, output);

mlpack::data::Save("satellite.train.3d.csv", output, true);
```

---

### Methods

By default, `TSNE` uses the Barnes-Hut method to approximate the gradient
calculation.  However, for smaller datasets, it may be possible to use the exact
method, and for some situations the dual-tree approximation may be preferable.
The `TSNE` class has a template parameter `TSNEMethod` that allows the different
gradient computation methods to be used.  The full signature of the class is:

```
TSNE<TSNEMethod, MatType, DistanceType>
```

 * `TSNEMethod`: specifies the TSNEMethod to be used to compute the t-SNE gradient.
 * `MatType`: specifies the type of matrix used for representation of data.
 * `DistanceType`: specifies the [distance metric](../core/distances.md) to be
   used for finding nearest neighbors.

These methods are already implemented and ready for drop-in usage:

- `ExactTSNE`
  - Computes the exact gradient of the t-SNE objective.  
  - Time Complexity: O(N^2).
  - Provides the most accurate results but quickly becomes impractical for large
    datasets (e.g., beyond a few thousand points).  
  - Use when dataset size is small and exact precision is required.  

- `BarnesHutTSNE` _(default)_  
  - Uses the Barnes-Hut approximation for computing the gradient.
  - Time Complexity: O(NlogN).  
  - Produces high-quality embeddings much faster than the exact method.  
  - Works well for medium to large datasets and is the most commonly used method in
  practice.

- `DualTreeTSNE`
  - Uses a dual-tree based approximation for computing the gradient.  
  - Time Complexity: O(NlogN).
  - Can be faster than Barnes-Hut in certain scenarios, especially for
    higher-dimensional embeddings.  
  - Useful when Barnes-Hut performance starts degrading with increasing dataset size
    or dimensionality.
  
The simple example program below uses all three strategies on the same
data, timing how long each one takes.

```c++
arma::mat data;
// See https://datasets.mlpack.org/mnist.train.csv.
// Note: this is a large dataset and may take a while.
mlpack::data::Load("mnist.train.csv", data, true);

arma::mat output1, output2, output3;

mlpack::TSNE<mlpack::BarnesHutTSNE> tsne1;
mlpack::TSNE<mlpack::DualTreeTSNE> tsne2;
mlpack::TSNE<mlpack::ExactTSNE> tsne3;

// Compute embeddings with all three, timing each one.
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

std::cout << "t-SNE computation times for " << data.n_rows << " x " << data.n_cols
    << " data:" << std::endl;
std::cout << " - BarnesHutTSNE: " << tsne1Time << "s." << std::endl;
std::cout << " - DualTreeTSNE:  " << tsne2Time << "s." << std::endl;
std::cout << " - ExactTSNE:     " << tsne3Time << "s." << std::endl;
```
