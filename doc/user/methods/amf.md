## `AMF`

The `AMF` class implements a general **a**lternating **m**atrix
**f**actorization framework, allowing numerous types of matrix decompositions.
The `AMF` class can decompose a large (potentially sparse) matrix `V` into two
smaller matrices `W` and `H`, such that `V ~= W * H`, using a variety of
strategies that involve iteratively updating first the `W` matrix, then the `H`
matrix, and so forth.  This technique may be used for dimensionality reduction,
or as part of a recommender system.

The behavior of the `AMF` class is controlled entirely by its template
parameters.  Different choices of these template parameters lead to different
algorithms for matrix decomposition.  For instance, mlpack's implementation of
non-negative matrix factorization ([`NMF`](nmf.md)) is built on the `AMF` class
with NMF-specific template parameters.

#### Simple usage example:

```c++
// Create a random sparse matrix (V) of size 10x100, with 15% nonzeros.
arma::sp_mat V;
V.sprandu(100, 100, 0.15);

// W and H will be low-rank matrices of size 100x10 and 10x100.
arma::mat W, H;

// Step 1: create object.  The choices of template parameters control the
// behavior of the decomposition.
mlpack::AMF<mlpack::SimpleResidueTermination   /* termination policy */,
            mlpack::RandomAcolInitialization<> /* policy to initialize W/H */,
            mlpack::SVDBatchLearning<>         /* alternating update rules */>
    amf;

// Step 2: apply alternating matrix factorization to decompose V.
double residue = amf.Apply(V, 10, W, H);

// Now print some information about the factorized matrices.
std::cout << "W has size: " << W.n_rows << " x " << W.n_cols << "."
    << std::endl;
std::cout << "H has size: " << H.n_rows << " x " << H.n_cols << "."
    << std::endl;
std::cout << "RMSE of reconstructed matrix: "
    << arma::norm(V - W * H, "fro") / std::sqrt(V.n_elem) << "." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Template parameter overview](#template-parameter-overview): description of
   the template parameters for the `AMF` class.
   * [`TerminationPolicyType`](#terminationpolicytype): behavior for terminating
     the `AMF` optimization.
   * [`InitializationRuleType`](#initializationruletype): behavior for
     initializing the `W` and `H` matrices.
   * [`UpdateRuleType`](#updateruletype): behavior for updating `W` and `H`.
 * [Constructors](#constructors): create `AMF` objects.
 * [`Apply()`](#applying-decompositions): apply `AMF` decomposition to data.
 * [Examples](#simple-examples) of usage and links to detailed example projects.
 * [Custom `TerminationPolicyType`s](#custom-terminationpolicytypes)
 * [Custom `InitializationRuleType`s](#custom-initializationruletypes)
 * [Custom `UpdateRuleType`s](#custom-updateruletypes)

#### See also:

<!-- TODO: add these links
 * [`CF`](cf.md): collaborative filtering (recommender system)
-->

 * [`NMF`](nmf.md): non-negative matrix factorization (a version of `AMF`)
 * [`SparseCoding`](sparse_coding.md)
 * [mlpack transformations](../transformations.md)
 * [Matrix factorization on Wikipedia](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems))

### Template parameter overview

The behavior of the `AMF` class is controlled by its three template parameters.
The full signature of the class is:

```
AMF<TerminationPolicyType, InitializationRuleType, UpdateRuleType>
```

 * `TerminationPolicyType`: determines the strategy used to terminate the
   alternating matrix factorization.  [Details...](#terminationpolicytype)

 * `InitializationRuleType`: determines the strategy used to initialize the `W`
   and `H` matrices at the start of the factorization.
   [Details...](#initializationruletype)

 * `UpdateRuleType`: determines the rules used to update `W` and `H` at each
   iteration in the factorization.  [Details...](#updateruletype)

The `AMF` class is most useful when each of these three parameters are
intentionally chosen.  The default template parameters simply configure the
algorithm as non-negative matrix factorization (NMF), and so in that situation
the [`NMF`](nmf.md) class can be used instead.

---

A number of convenient typedefs are possible to configure the `AMF` class as a
predefined algorithm.  It may be easier to use these than to manually specify
each template parameter.

 * `SVDBatchFactorizer<WHMatType = arma::mat>`
   - Use batch SVD factorizer (Algorithm 4 from Chih-Chao Ma's
     [A Guide to Singular Value Decomposition for Collaborative Filtering](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=9d14285a32d268b69d51e7036d5a391c007df886).
   - `WHMatType` (default `arma::mat`) represents the type used to represent the
     `W` and `H` matrices.
  -  Uses [`SimpleResidueTermination`](#simpleresiduetermination-default) and
     [`RandomAcolInitialization`](#randomacolinitializationn-default).
   - See the [`SVDCompleteIncrementalLearning`](#svdcompleteincrementallearning)
     update rule.

---

 * `SVDIncompleteIncrementalFactorizer<VMatType = arma::mat>`
   - Use incomplete incremental SVD factorizer (Algorithm 2 from Chih-Chao Ma's
     [A Guide to Singular Value Decomposition for Collaborative Filtering](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=9d14285a32d268b69d51e7036d5a391c007df886).
   - `VMatType` (default `arma::mat`) represents the type of the `V` matrix that
     will be decomposed.
   - Uses
     [`IncompleteIncrementalTermination`](#incompleteincrementalterminationterminationpolicy),
     [`SimpleResidueTermination`](#simpleresiduetermination-default), and
     [`RandomAcolInitialization`](#randomacolinitializationn-default).
   - See the
     [`SVDIncompleteIncrementalLearning`](#svdincompleteincrementallearning)
     update rule.

---

 * `SVDCompleteIncrementalFactorizer<VMatType = arma::mat>`
   - Use complete incremental SVD factorizer (Algorithm 3 from Chih-Chao Ma's
     [A Guide to Singular Value Decomposition for Collaborative Filtering](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=9d14285a32d268b69d51e7036d5a391c007df886).
   - `VMatType` (default `arma::mat`) represents the type of the `V` matrix that
     will be decomposed.
   - Uses
     [`CompleteIncrementalTermination`](#completeincrementalterminationterminationpolicy),
     [`SimpleResidueTermination`](#simpleresiduetermination-default), and
     [`RandomAcolInitialization`](#randomacolinitializationn-default).
   - See the [`SVDBatchLearning`](#svdbatchlearning) update rule.

---

 * `NMF<TerminationPolicyType, InitializationRuleType, UpdateRuleType>`
   - Perform non-negative matrix factorization using multiplicative distance
     update rules.
   - See the [`NMF`](nmf.md) class documentation for more details.

---

### `TerminationPolicyType`

 * Specifies the strategy to use to choose when to stop the AMF algorithm.
 * An instantiated `TerminationPolicyType` can be passed to the `AMF`
   [constructor](#constructors).
 * The following choices are available for drop-in usage:

---

#### ***`SimpleResidueTermination`*** (default):

 - Terminates when a maximum number of iterations is reached, or when the
   residue (change in norm of `W * H` between iterations) is sufficiently small.
 - Constructor: `SimpleResidueTermination(minResidue=1e-5, maxIterations=10000)`
   * `minResidue` (a `double`) specifies the sufficiently small residue for
     termination.
   * `maxIterations` (a `size_t`) specifies the maximum number of iterations.
 - [`amf.Apply()`](#applying-decompositions) will return the residue of the last
   iteration.

---

#### ***`MaxIterationTermination`***:

 - Terminates when the maximum number of iterations is reached.
 - No other condition is checked.
 - Constructor: `MaxIterationTermination(maxIterations=1000)`
 - [`amf.Apply()`](#applying-decompositions) will return the number of
   iterations performed.

---

#### ***`SimpleToleranceTermination<MatType, WHMatType>`***:

 - Terminates when the nonzero residual decreases a sufficiently small relative
   amount between iterations (e.g.
   `(lastNonzeroResidual - nonzeroResidual) / lastNonzeroResidual` is below a
   threshold), or when the maximum number of iterations is reached.
 - The residual must remain below the threshold for a specified number of
   iterations.
 - The nonzero residual is defined as the root of the sum of squared elements in
   the reconstruction error matrix `(V - WH)`, limited to locations where `V` is
   nonzero.
 - Constructor: `SimpleToleranceTermination<MatType, WHMatType>(tol=1e-5, maxIter=10000, extraSteps=3)`
   * `MatType` should be set to the type of `V` (see
     [`Apply()` Parameters](#apply-parameters)).
   * `WHMatType` (default `arma::mat`) should be set to the type of `W` and `H`
     (see [`Apply()` Parameters](#apply-parameters)).
   * `tol` (a `double`) specifies the relative nonzero residual tolerance for
     convergence.
   * `maxIter` (a `size_t`) specifies the maximum number of iterations
     before termination.
   * `extraSteps` (a `size_t`) specifies the number of iterations
     where the relative nonzero residual must be below the tolerance for
     convergence.
 - The best `W` and `H` matrices (according to the nonzero residual) from the
   final `extraSteps` iterations are returned by
   [`amf.Apply()`](#applying-decompositions).
 - [`amf.Apply()`](#applying-decompositions) will return the nonzero residue of
   the iteration corresponding to the best `W` and `H` matrices.

---

#### ***`CompleteIncrementalTermination<TerminationPolicy>`***

 - Meant to be used with the
   [`SVDCompleteIncrementalLearning`](#svdcompleteincrementallearning)
   update rules.
 - Checks for convergence only after the entire `V` matrix has been used to
   update `W` and `H`, instead of checking after the updates for each element of
   `V`.
 - `TerminationPolicy` specifies the actual convergence condition to check.
 - Constructors:
    * `CompleteIncrementalTermination<TerminationPolicy>()`
    * `CompleteIncrementalTermination<TerminationPolicy>(terminationPolicy)`

---

#### ***`IncompleteIncrementalTermination<TerminationPolicy>`***

 - Meant to be used with the
   [`SVDIncompleteIncrementalLearning`](#svdincompleteincrementallearning)
   update rules.
 - Checks for convergence only after the entire `V` matrix has been used to
   update `W` and `H`, instead of checking after the updates for each column of
   `V`.
 - `TerminationPolicy` specifies the actual convergence condition to check.
 - Constructors:
    * `IncompleteIncrementalTermination<TerminationPolicy>()`
    * `IncompleteIncrementalTermination<TerminationPolicy>(terminationPolicy)`

---

For custom termination policies, see
[Custom `TerminationPolicyType`s](#custom-terminationpolicytypes).

---

### `InitializationRuleType`

 * Specifies the strategy to use to initialize `W` and `H` at the beginning of
   the NMF algorithm.
 * An initialized `InitializationRuleType` can be passed to the
   [`AMF` constructor](#constructors).
 * The following choices are available for drop-in usage:

---

#### ***`RandomAcolInitialization<N>`*** (default):

 - Initialize `W` by averaging `N` randomly chosen columns of `V`.
 - Initialize `H` as uniform random in the range `[0, 1]`.
 - The default value for `N` is 5.
 - See also [the paper](https://arxiv.org/abs/1407.7299) describing the
   strategy.

---

#### ***`NoInitialization`***:

 - When [`amf.Apply(V, rank, W, H)`](#applying-decompositions), the existing
   values of `W` and `H` will be used.
 - If `W` is not of size `V.n_rows` x `rank`, or if `H` is not of size `rank` x
   `V.n_cols`, a `std::invalid_argument` exception will be thrown.

---

#### ***`GivenInitialization<MatType>`***:

 - Set `W` and/or `H` to the given matrices when `Apply()` is called.
 - `MatType` should be set to the type of `W` or `H` (default `arma::mat`); see
   [`Apply()` Parameters](#apply-parameters).
 - Constructors:
   * `GivenInitialization<MatType>(W, H)`
     - Specify both initial `W` and `H` matrices.
   * `GivenInitialization<MatType>(M, isW=true)`
     - If `isW` is `true`, then set initial `W` to `M`.
     - If `isW` is `false`, then set initial `H` to `M`.
     - This constructor is meant to only be used with `MergeInitialization`
       (below).

---

#### ***`RandomAMFInitialization`***:

 - Initialize `W` and `H` as uniform random in the range `[0, 1]`.

---

#### ***`AverageInitialization`***:

 - Initialize each element of `W` and `H` to the square root of the average
   value of `V`, adding uniform random noise in the range `[0, 1]`.

---

#### ***`MergeInitialization<WRule, HRule>`***:

 - Use two different initialization rules, one for `W` (`WRule`) and one for `H`
   (`HRule`).
 - Constructors:
   * `MergeInitialization<WRule, HRule>()`
     - Create the merge initialization with default-constructed rules for `W`
       and `H`.
   * `MergeInitialization<WRule, HRule>(wRule, hRule)`
     - Create the merge initialization with instantiated rules for `W` and `H`.
     - `wRule` and `hRule` will be copied.
 - Any `WRule` and `HRule` classes must implement the `InitializeOne()`
   function.

---

For custom initialization rules, see
[Custom `InitializationRuleType`s](#custom-initializationruletypes).

---

### `UpdateRuleType`

 * Specifies the rules to use for the `W` update step and the `H` update step.
 * These rules are applied iteratively until convergence (controlled by
   [`TerminationPolicyType`](#terminationpolicytype).
 * The following choices are available for drop-in usage:

---

#### `NMF` updates

Non-negative matrix factorization (NMF) can be expressed with the `AMF` class
using any of the following `UpdateRuleType`s.

 - `NMFMultiplicativeDistanceUpdate`: update rule that ensure the Frobenius
   norm of the reconstruction error is decreasing at each iteration.
 - `NMFMultiplicativeDivergenceUpdate`: update rules that ensure
   Kullback-Leibler divergence is decreasing at each iteration.
 - `NMFALSUpdate`: alternating least-squares projections for `W` and `H`.

***Note***: when using these update rules, it may be more convenient to use the
more specific [`NMF`](nmf.md) class.  `NMF` is just a typedef for
`AMF<SimpleResidueTermination, RandomAcolInitialization<5>, NMFMultiplicativeDistanceUpdate>`.

---

#### `SVDBatchLearning`

 - Use gradient descent with momentum on the full matrix `V` to iteratively
   update `W` and then `H`.
 - Takes one template parameter: `SVDBatchLearning<WHMatType>`.
   * `WHMatType` specifies the type of matrix that will be used to store `W` and
     `H` (default: `arma::mat`).
 - Implements Algorithm 4 from Chih-Chao Ma's [A Guide to Singular Value
   Decomposition for Collaborative Filtering](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=9d14285a32d268b69d51e7036d5a391c007df886).
 - Constructor: `SVDBatchLearning<WHMatType>(u=0.0002, kw=0.0, kh=0.0, momentum=0.9)`
   * `u` (a `double`) is the learning rate (step size).
   * `kw` (a `double`) is the regularization penalty for the `W` matrix.
   * `kh` (a `double`) is the regularization penalty for the `H` matrix.
   * `momentum` (a `double`) is the momentum rate for each gradient descent
     step.

---

#### `SVDCompleteIncrementalLearning`

 - Use gradient descent on individual values of the full matrix `V` to
   iteratively update `W` and then `H`.
 - Takes one template parameter: `MatType`
   * `MatType` specifies the type of the `V` matrix (e.g. `arma::mat` or
     `arma::sp_mat`).
 - Each update to `W` and `H` is done by computing the gradient using a single
   nonzero value from `V` (similar to stochastic gradient descent with a batch
   size of 1).
 - Implements Algorithm 3 from Chih-Chao Ma's [A Guide to Singular Value
   Decomposition for Collaborative Filtering](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=9d14285a32d268b69d51e7036d5a391c007df886).
 - Constructor: `SVDCompleteIncrementalLearning(u=0.001, kw=0.0, kh=0.0)`
   * `u` (a `double`) is the learning rate (step size).
   * `kw` (a `double`) is the regularization penalty for the `W` matrix.
   * `kh` (a `double`) is the regularization penalty for the `H` matrix.

---

#### `SVDIncompleteIncrementalLearning`

 - Use gradient descent on individual columns of the full matrix `V` to
   iteratively update `W` and then `H`.
 - Takes one template parameter: `MatType`
   * `MatType` specifies the type of the `V` matrix (e.g. `arma::mat` or
     `arma::sp_mat`).
 - Each update to `W` and `H` is done by computing the gradient using all
   nonzero values in a single column of `V`.
 - Implements Algorithm 2 from Chih-Chao Ma's [A Guide to Singular Value
   Decomposition for Collaborative Filtering](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=9d14285a32d268b69d51e7036d5a391c007df886).
 - Constructor: `SVDIncompleteIncrementalLearning(u=0.001, kw=0.0, kh=0.0)`
   * `u` (a `double`) is the learning rate (step size).
   * `kw` (a `double`) is the regularization penalty for the `W` matrix.
   * `kh` (a `double`) is the regularization penalty for the `H` matrix.

---

For custom update rules, see
[Custom `UpdateRuleType`s](#custom-updateruletypes).

---

### Constructors

 * `amf = AMF<TerminationPolicyType, InitializationRuleType, UpdateRuleType>()`
   - Create an `AMF` object.
   - The rank of the decomposition is specified in the call to
     [`Apply()`](#applying-decompositions).

---

 * `amf = AMF<TerminationPolicyType, InitializationRuleType, UpdateRuleType>(terminationPolicy, initializeRule, updateRule)`
   - Create an AMF object with custom termination parameters.
   - `minResidue` (a `double`) specifies the minimum difference of the norm of
     `W * H` between iterations for termination.
   - `maxIterations` specifies the maximum number of iterations before
     decomposition terminates.

---

### Applying Decompositions

 * `double residue = amf.Apply(V, rank, W, H)`
   - Decompose the matrix `V` into two non-negative matrices `W` and `H` with
     rank `rank`.
   - `W` will be set to size `V.n_rows` x `rank`.
   - `H` will be set to size `rank` x `V.n_cols`.
   - `W` and `H` are initialized using the specified
     [`InitializationRuleType`](#initializationruletype).
   - The return value is determined by the
     [`TerminationPolicyType`](#terminationpolicytype); termination policies
     typically return residue or a similar measure of goodness-of-fit.

---

***Notes***:

 - Low values of `rank` will give smaller matrices `W` and `H`, but the
   decomposition will be less accurate.  Larger values of `rank` will give more
   accurate decompositions, but will take longer to compute.  Every problem is
   different, so `rank` must be specified manually.

 - The expression `W * H` can be used to reconstruct the matrix `V`.

---

#### `Apply()` Parameters:

| **name** | **type** | **description** |
|----------|----------|-----------------|
| `V` | [`arma::sp_mat` or `arma::mat`](../matrices.md) | Input matrix to be factorized. |
| `rank` | `size_t` | Rank of decomposition; lower is smaller, higher is more accurate. |
| `W` | [`arma::mat`](../matrices.md) | Output matrix in which `W` will be stored. |
| `H` | [`arma::mat`](../matrices.md) | Output matrix in which `H` will be stored. |

***Note:*** Matrices with different element types can be used for `V`, `W`, and
`H`; e.g., `arma::fmat`.  While `V` can be sparse or dense, `W` and `H` must be
dense matrices.

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial use of
`AMF`.

---

Decompose a dense matrix with simple residue termination using custom
parameters.

```c++
// Create a low-rank V matrix by multiplying together two random matrices.
arma::mat V = arma::randu<arma::mat>(500, 25) *
              arma::randn<arma::mat>(25, 5000);

// Create the AMF object with a looser tolerance of 1e-3 and a maximum of 100
// iterations only.
// Since we have not specified the update rules, this will by default use the
// NMF multiplicative distance update.
mlpack::AMF<mlpack::SimpleResidueTermination> amf(
    mlpack::SimpleResidueTermination(1e-3, 500));

arma::mat W, H;

// Decompose with a rank of 25.
// W will have size 500 x 25, and H will have size 25 x 5000.
const double residue = amf.Apply(V, 25, W, H);

std::cout << "Residue of decomposition: " << residue << "." << std::endl;

// Compute RMSE of decomposition.
const double rmse = arma::norm(V - W * H, "fro") / std::sqrt(V.n_elem);
std::cout << "RMSE of decomposition: " << rmse << "." << std::endl;
```

---

Decompose the sparse MovieLens dataset using batch SVD learning, a rank-12
decomposition, and `float` element type.

```c++
// See https://datasets.mlpack.org/movielens-100k.csv.
arma::sp_fmat V;
mlpack::data::Load("movielens-100k.csv", V, true);

// Create the AMF object.  Use default parameters for the termination policy,
// initialization rule, and update rules.
mlpack::AMF<mlpack::SimpleResidueTermination,
            mlpack::RandomAcolInitialization<5>,
            mlpack::SVDBatchLearning<arma::fmat>> amf;

arma::fmat W, H;

// Decompose the Movielens dataset with rank 12.
const double residue = amf.Apply(V, 12, W, H);

std::cout << "Residue of MovieLens decomposition: " << residue << "."
    << std::endl;

// Compute RMSE of decomposition.
const double rmse = arma::norm(V - W * H, "fro") / std::sqrt(V.n_elem);
std::cout << "RMSE of decomposition: " << rmse << "." << std::endl;
```

---

Compare quality of decompositions of MovieLens with different update rules.

```c++
// See https://datasets.mlpack.org/movielens-100k.csv.
arma::sp_mat V;
mlpack::data::Load("movielens-100k.csv", V, true);

// Create four AMF objects using different update rules:
//  - SVDBatchLearning
//  - SVDCompleteIncrementalLearning
//  - SVDIncompleteIncrementalLearning
//  - NMFALSUpdate
// We use MaxIterationTermination for each, wrapped in incremental terminators
// if appropriate.

mlpack::AMF<mlpack::MaxIterationTermination,
            mlpack::RandomAcolInitialization<5>,
            mlpack::SVDBatchLearning<>>
    svdBatchAMF(mlpack::MaxIterationTermination(500),
                mlpack::RandomAcolInitialization<5>(),
                mlpack::SVDBatchLearning<>(0.0005 /* step size */));

mlpack::AMF<mlpack::CompleteIncrementalTermination<
                mlpack::MaxIterationTermination
            >,
            mlpack::RandomAcolInitialization<5>,
            mlpack::SVDCompleteIncrementalLearning<arma::sp_mat>>
    svdCompleteAMF(mlpack::CompleteIncrementalTermination<
                       mlpack::MaxIterationTermination>(500),
                   mlpack::RandomAcolInitialization<5>(),
                   mlpack::SVDCompleteIncrementalLearning<
                       arma::sp_mat>(0.0002));

mlpack::AMF<mlpack::IncompleteIncrementalTermination<
                mlpack::MaxIterationTermination>,
            mlpack::RandomAcolInitialization<5>,
            mlpack::SVDIncompleteIncrementalLearning<arma::sp_mat>>
    svdIncompleteAMF(mlpack::IncompleteIncrementalTermination<
                         mlpack::MaxIterationTermination>(500),
                     mlpack::RandomAcolInitialization<5>(),
                     mlpack::SVDIncompleteIncrementalLearning<
                         arma::sp_mat>(0.0002));

// NMFALSUpdate does not have any template parameters, so we don't need to pass
// it to the constructor.
mlpack::AMF<mlpack::MaxIterationTermination,
            mlpack::RandomAcolInitialization<5>,
            mlpack::NMFALSUpdate>
    nmf(mlpack::MaxIterationTermination(500));

// Decompose with the given rank.
arma::mat W, H;
const size_t rank = 15;

const double svdBatchResidue = svdBatchAMF.Apply(V, rank, W, H);
const double svdBatchRMSE = arma::norm(V - W * H, "fro") / std::sqrt(V.n_elem);
std::cout << "RMSE for SVD batch learning: " << svdBatchRMSE << "."
    << std::endl;

const double svdCompleteResidue = svdCompleteAMF.Apply(V, rank, W, H);
const double svdCompleteRMSE = arma::norm(V - W * H, "fro") /
    std::sqrt(V.n_elem);
std::cout << "RMSE for SVD complete incremental learning: " << svdCompleteRMSE
    << "." << std::endl;

const double svdIncompleteResidue = svdIncompleteAMF.Apply(V, rank, W, H);
const double svdIncompleteRMSE = arma::norm(V - W * H, "fro") /
    std::sqrt(V.n_elem);
std::cout << "RMSE for SVD incomplete incremental learning: "
    << svdIncompleteRMSE << "." << std::endl;

const double nmfResidue = nmf.Apply(V, rank, W, H);
const double nmfRMSE = arma::norm(V - W * H, "fro") / std::sqrt(V.n_elem);
std::cout << "RMSE for NMF with ALS update rules: " << nmfRMSE << "."
    << std::endl;
```

---

Use a pre-specified initialization for `W` and `H`.

```c++
// See https://datasets.mlpack.org/movielens-100k.csv.
arma::sp_mat V;
mlpack::data::Load("movielens-100k.csv", V, true);

arma::mat W, H;

// Pre-initialize W and H.
// W will be filled with random values from a normal distribution.
// H will be filled with 1s.
W.randn(V.n_rows, 15);
H.set_size(15, V.n_cols);
H.fill(0.2);

mlpack::AMF<mlpack::MaxIterationTermination,
            mlpack::NoInitialization,
            mlpack::SVDBatchLearning<>>
    amf(mlpack::MaxIterationTermination(1000));
const double residue = amf.Apply(V, 15, W, H);
const double rmse = arma::norm(V - W * H, "fro") / std::sqrt(V.n_elem);

std::cout << "RMSE of SVDBatchLearning decomposition with pre-specified W and "
    << "H: " << rmse << "." << std::endl;
```

---

Use `MergeInitialization` to specify different strategies to initialize the `W`
and `H` matrices.

```c++
// See https://datasets.mlpack.org/movielens-100k.csv.
arma::sp_mat V;
mlpack::data::Load("movielens-100k.csv", V, true);

arma::mat W, H;

// This will initialize the W matrix.
mlpack::RandomAcolInitialization<5> initW;

// This will initialize the H matrix.
mlpack::RandomAMFInitialization initH;

// Combine the two initializations so we can pass it to the AMF class.
using InitType =
    mlpack::MergeInitialization<mlpack::RandomAcolInitialization<5>,
                                mlpack::RandomAMFInitialization>;
InitType init(initW, initH);

// Create an AMF object with the custom initialization.
mlpack::AMF<mlpack::CompleteIncrementalTermination<
                mlpack::SimpleResidueTermination
            >,
            InitType,
            mlpack::SVDCompleteIncrementalLearning<arma::sp_mat>>
    amf(mlpack::CompleteIncrementalTermination<
            mlpack::SimpleResidueTermination>(), init);

// Perform AMF with a rank of 10.
const double residue = amf.Apply(V, 10, W, H);

std::cout << "Residue after training: " << residue << "." << std::endl;
```

---

Use `ValidationRMSETermination` to decompose the MovieLens dataset until the
RMSE of the held-out validation set is sufficiently low.

```c++
// See https://datasets.mlpack.org/movielens-100k.csv.
arma::sp_mat V;
mlpack::data::Load("movielens-100k.csv", V, true);

arma::mat W, H;

// Create a ValidationRMSETermination class that will hold out 3k points from V.
// This will remove 3000 nonzero entries from V.
mlpack::ValidationRMSETermination<arma::sp_mat> t(V, 3000);

// Create the AMF object with the instantiated termination policy.
mlpack::AMF<mlpack::ValidationRMSETermination<arma::sp_mat>,
            mlpack::RandomAcolInitialization<5>,
            mlpack::SVDBatchLearning<>> amf(t);

// Perform AMF with a rank of 20.
// Note the RMSE returned here is the RMSE on the validation set.
const double rmse = amf.Apply(V, 20, W, H);
const double rmseTrain = arma::norm(V - W * H, "fro") / std::sqrt(V.n_elem);

std::cout << "Training RMSE:   " << rmseTrain << "." << std::endl;
std::cout << "Validation RMSE: " << rmse << "." << std::endl;
```

---

### Custom `TerminationPolicyType`s

See also the [list of available `TerminationPolicyType`s](#terminationpolicytype).

If custom functionality is desired for controlling the termination of the `AMF`
algorithm, a custom class may be implemented that must implement the following
functions:

```c++
// You can use this as a starting point for implementation.
class CustomTerminationPolicy
{
 public:
  // Initialize the termination policy for the given matrix V.  (It is okay to
  // do nothing.)  This function is called at the beginning of Apply().
  //
  // If the termination policy requires V to compute convergence, store a
  // reference or pointer to it in this function.
  template<typename MatType>
  void Initialize(const MatType& V);

  // Check if convergence has occurred for the given W and H matrices.  Return
  // `true` if so.
  //
  // Note that W and H may have different types than V (i.e. V may be sparse,
  // and W and H must be dense.)
  template<typename WHMatType>
  bool IsConverged(const MatType& H, const MatType& W);

  // Return the value that should be returned for the `amf.Apply()` function
  // when convergence has been reached.  This is called at the end of
  // `amf.Apply()`.
  const double Index();

  // Return the number of iterations that have been completed.  This is called
  // at the end of `amf.Apply()`.
  const size_t Iteration();
};
```

---

### Custom `InitializationRuleType`s

See also the [list of available `InitializationRuleType`s](#initializationruletype).

If custom functionality is desired for initializing `W` and `H`, a custom class
may be implemented that must implement the following functions:

```c++
// You can use this as a starting point for implementation.
class CustomInitialization
{
 public:
  // Initialize the W and H matrices, given V and the rank of the decomposition.
  // This is called at the start of `Apply()`.
  //
  // Note that `MatType` may be different from `WHMatType`; e.g., `V` could be
  // sparse, but `W` and `H` must be dense.
  template<typename MatType, typename WHMatType>
  void Initialize(const MatType& V,
                  const size_t rank,
                  WHMatType& W,
                  WHMatType& H);

  // Initialize one of the W or H matrices, given V and the rank of the
  // decomposition.
  //
  // If `isW` is `true`, then `M` should be treated as though it is `W`;
  // if `isW` is `false`, then `M` should be treated as thought it is `H`.
  //
  // This function only needs to be implemented if it is intended to use the
  // custom initialization strategy with `MergeInitialization`.
  template<typename MatType, typename WHMatType>
  void InitializeOne(const MatType& V,
                     const size_t rank,
                     WHMatType& M,
                     const bool isW);
};
```

For example, the code below implements a custom termination policy that sets a
limit on how long AMF is allowed to take:

```c++
class CustomTimeTermination
{
 public:
  CustomTimeTermination(const double totalAllowedTime) :
      totalAllowedTime(totalAllowedTime) { }

  template<typename MatType>
  void Initialize(const MatType& /* V */)
  {
    totalTime = 0.0;
    iteration = 0;
    c.tic();
  }

  template<typename WHMatType>
  bool IsConverged(const WHMatType& /* W */, const WHMatType& /* H */)
  {
    totalTime += c.toc();
    c.tic();
    ++iteration;
    return (totalTime > totalAllowedTime);
  }

  const double Index() const { return totalTime; }
  const size_t Iteration() const { return iteration; }

 private:
  double totalAllowedTime;
  double totalTime;
  size_t iteration;
  arma::wall_clock c; // used for convenient timing
};
```

Then we can use it in a program:

```c++
// See https://datasets.mlpack.org/movielens-100k.csv.
arma::sp_fmat V;
mlpack::data::Load("movielens-100k.csv", V, true);

CustomTimeTermination t(5 /* seconds */);
mlpack::AMF<CustomTimeTermination,
            mlpack::RandomAcolInitialization<5>,
            mlpack::SVDBatchLearning<arma::fmat>> amf(t);

arma::fmat W, H;
const double actualTime = amf.Apply(V, 10, W, H);
const double rmse = arma::norm(V - W * H, "fro") / std::sqrt(V.n_elem);

std::cout << "Actual time used for decomposition: " << actualTime << "."
    << std::endl;
std::cout << "RMSE after ~5 seconds: " << rmse << "." << std::endl;
```

---

### Custom `UpdateRuleType`s

See also the [list of available `UpdateRuleType`s](#updateruletype).

If custom functionality is desired for the update rules to be applied to `W` and
`H`, a custom class may be implemented that must implement the following
functions:

```c++
// You can use this as a starting point for implementation.
class CustomUpdateRule
{
 public:
  // Set initial values for the factorization.  This is called at the beginning
  // of Apply(), before WUpdate() or HUpdate() are called.
  //
  // `MatType` will be the type of `V` (an Armadillo dense or sparse matrix).
  //
  template<typename MatType>
  void Initialize(const MatType& V, const size_t rank);

  // Update the `W` matrix given `V` and the current `H` matrix.
  //
  // `MatType` will be the type of `V`, and `WHMatType` will be the type of `W`
  // and `H`.  Both will be matrix types matching the Armadillo API.
  template<typename MatType, typename WHMatType>
  void WUpdate(const MatType& V, WHMatType& W, const WHMatType& H);

  // Update the `H` matrix given `V` and the current `W` matrix.
  //
  // `MatType` will be the type of `V`, and `WHMatType` will be the type of `W`
  // and `H`.  Both will be matrix types matching the Armadillo API.
  template<typename MatType, typename WHMatType>
  void HUpdate(const MatType& V, const WHMatType& W, WHMatType& H);

  // Serialize the update rule using the cereal library.
  // This is only necessary if the update rule will be used with an AMF object
  // that is saved or loaded with data::Save() or data::Load().
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version);
};
```
