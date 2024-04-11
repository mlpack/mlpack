## `NMF`

The `NMF` class implements non-negative matrix factorization, a technique to
decompose a large (potentially sparse) matrix `V` into two smaller matrices `W`
and `H`, such that `V ~= W * H`, and `W` and `H` only contain nonnegative
elements.  This technique may be used for dimensionality reduction, or as part
of a recommender system.

The `NMF` class allows fully configurable behavior via [template
parameters](#advanced-functionality-template-parameters).  For more general
matrix factorization strategies, see the [`AMF`](user/methods/amf.md)
(alternating matrix factorization) class documentation.

#### Simple usage example:

```c++
// Create a random sparse matrix (V) of size 10x100, with 5% nonzeros.
arma::sp_mat V;
V.sprandu(100, 100, 0.05);

// W and H will be low-rank matrices of size 100x10 and 10x100.
arma::mat W, H;

NMF nmf;                              // Step 1: create object.
double rmse = nmf.Apply(V, 10, W, H); // Step 2: apply NMF to decompose V.

// Now print some information about the factorized matrices.
std::cout << "W has size: " << W.n_rows << " x " << W.n_cols << "."
    << std::endl;
std::cout << "H has size: " << H.n_rows << " x " << H.n_cols << "."
    << std::endl;
std::cout << "Reconstruction error (RMSE): " << rmse << "." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `NMF` objects.
 * [`Apply()`](#applying-decompositions): apply `NMF` decomposition to data.
 * [Examples](#simple-examples) of simple usage and links to detailed example
   projects.
 * [Template parameters](#advanced-functionality-template-parameters) for
   using different update rules, initialization strategies, and termination
   criteria.
 * [Advanced template examples](#advanced-functionality-examples) of use with
   custom template parameters.

#### See also:

 * [`AMF`](amf.md): alternating matrix factorization
 * [`CF`](cf.md): collaborative filtering (recommender system)
 * [`SparseCoding`](sparse_coding.md)
 * [mlpack transformations](../../index.md#transformations)
 * [Non-negative matrix factorization on Wikipedia](https://en.wikipedia.org/wiki/Non-negative_Matrix_Factorization)
 * [original paper](...) <!-- TODO -->

### Constructors

 * `nmf = NMF()`
   - Create an `NMF` object.
   - The rank of the decomposition is specified in the call to
     [`Apply()`](#applying-decompositions).

---

 * `nmf = NMF(SimpleResidueTermination(minResidue=1e-5, maxIterations=10000))`
   - Create an NMF object with custom termination parameters.
   - `minResidue` (a `double`) specifies the minimum difference of the norm of
     `W * H` between iterations for termination.
   - `maxIterations` specifies the maximum number of iterations before
     decomposition terminates.

---

### Applying Decompositions

 * `double rmse = nmf.Apply(V, rank, W, H)`
   - Decompose the matrix `V` into two non-negative matrices `W` and `H` with
     rank `rank`.
   - `W` will be set to size `V.n_rows` x `rank`.
   - `H` will be set to size `rank` x `V.n_cols`.
   - `W` and `H` are initialized randomly using the
     [Acol](#initialization-rules) initialization strategy; i.e., each column of
     `W` is an average of 5 random columns of `V`, and `H` is initialized
     uniformly randomly.
   - The RMSE (root mean squared error) of the decomposition is returned.

---

***Notes***:

 - Low values of `rank` will give smaller matrices `W` and `H`, but the
   decomposition will be less accurate.  Every problem is different, so `rank`
   must be specified manually.

 - The expression `W * H` can be used to reconstruct the matrix `V`.

 - Custom behavior, such as custom initialization of `W` and `H`, different or
   custom termination rules, and different update rules are discussed in the
   [advanced functionality](#advanced-functionality-template-parameters)
   section.

#### `Apply()` Parameters:

| **name** | **type** | **description** |
|----------|----------|-----------------|
| `V` | [`arma::sp_mat` or `arma::mat`](../matrices.md) | Input matrix to be factorized. |
| `rank` | `size_t` | Rank of decomposition; lower is smaller, higher is more
accurate. |
| `W` | [`arma::mat`](../matrices.md) | Output matrix in which `W` will be stored. |
| `H` | [`arma::mat`](../matrices.md) | Output matrix in which `H` will be stored. |

***Note:*** Matrices with different element types can be used for `V`, `W`, and
`H`; e.g., `arma::fmat`.  While `V` can be sparse or dense, `W` and `H` must be
dense matrices.

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial use of
`NMF`.

---

Decompose a dense matrix with custom termination parameters.

```c++
// The matrix will have size 500x5000 with normally distributed elements.
arma::mat V(500, 5000, arma::fill::randn);

// Create the NMF object with a looser tolerance of 1e-3 and a maximum of 100
// iterations only.
mlpack::NMF nmf(mlpack::SimpleResidueTermination(1e-3, 100));

arma::mat W, H;

// Decompose with a rank of 25.
// W will have size 500 x 25, and H will have size 25 x 5000.
const double rmse = nmf.Apply(V, 25, W, H);

std::cout << "RMSE of decomposition: " << rmse << "." << std::endl;

// Compute norm of error matrix.
const double errorNorm = arma::norm(V - W * H);
std::cout << "Norm of error matrix: " << errorNorm << "." << std::endl;
```

---

Decompose the sparse MovieLens dataset using a rank-12 decomposition and `float`
element type.

```c++
// See https://datasets.mlpack.org/movielens-100k.csv.
arma::sp_fmat V;
mlpack::data::Load("movielens-100k.csv", V, true);

// Create the NMF object.
mlpack::NMF nmf;

arma::fmat W, H;

// Decompose the Movielens dataset with rank 12.
const double rmse = nmf.Apply(V, 12, W, H);

std::cout << "RMSE of MovieLens decomposition: " << rmse << "." << std::endl;
```

---

Compare quality of decompositions of MovieLens with different ranks.

```c++
// See https://datasets.mlpack.org/movielens-100k.csv
arma::sp_mat V;
mlpack::data::Load("movielens-100k.csv", V, true);

// Create the NMF object.
mlpack::NMF nmf;
arma::mat W, H;

for (size_t rank = 10; rank < 100; rank += 5)
{
  // Decompose with the given rank.
  const double rmse = nmf.Apply(V, rank, W, H);

  std::cout << "RMSE for rank-" << rank << " decomposition: " << rmse << "."
      << std::endl;
}
```

---

### Advanced Functionality: Template Parameters

The `NMF` class has three template parameters that can be used for custom
behavior.  The full signature of the class is:

```
NMF<TerminationPolicyType, InitializationRuleType, UpdateRuleType>
```

 * `TerminationPolicyType`: the strategy used to choose when to terminate NMF.
 * `InitializationRuleType`: the strategy used to choose the initial `W` and `H`
   matrices.
 * `UpdateRuleType`: the update rules used for NMF:
   - `NMFMultiplicativeDistanceUpdate`: update rule that ensure the Frobenius
     norm of the reconstruction error is decreasing at each iteration.
   - `NMFMultiplicativeDivergenceUpdate`: update rules that ensure
     Kullback-Leibler divergence is decreasing at each iteration.
   - `NMFALSUpdate`: alternating least-squares projections for `W` and `H`.
   - For custom update rules, use the more general
     [`AMF`](/src/mlpack/methods/amf/amf.hpp) class.

<!-- TODO: update link above to Markdown documentation -->

---

#### `TerminationPolicyType`

 * Specifies the strategy to use to choose when to stop the NMF algorithm.
 * An instantiated `TerminationPolicyType` can be passed to the NMF constructor.
 * The following choices are available for drop-in usage:

***`SimpleResidueTermination`*** (default):

 - Terminates when a maximum number of iterations is reached, or when the
   residue (change in norm of `W * H` between iterations) is sufficiently small.
 - Constructor: `SimpleResidueTermination(minResidue=1e-5, maxIterations=10000)`
   * `minResidue` (a `double`) specifies the sufficiently small residue for
     termination.
   * `maxIterations` (a `size_t`) specifies the maximum number of iterations.

***`MaxIterationTermination`***:

 - Terminates when the maximum number of iterations is reached.
 - No other condition is checked.
 - Constructor: `MaxIterationTermination(maxIterations=1000)`

***`SimpleToleranceTermination<MatType, WHMatType>`***:

 - Terminates when the nonzero residual decreases a sufficiently small relative
   amount between iterations (e.g. `(lastNonzeroResidual - nonzeroResidual) /
   lastNonzeroResidual` is below a threshold), or when the maximum number of
   iterations is reached.
 - The residual must remain below the threshold for a specified number of
   iterations.
 - The nonzero residual is defined as the root of the sum of squared elements in
   the reconstruction error matrix `(V - WH)`, limited to locations where `V` is
   nonzero.
 - Constructor: `SimpleToleranceTermination<MatType, WHMatType>(tolerance=1e-5, maxIterations=10000,
   reverseStepTolerance=3)`
   * `MatType` should be set to the type of `V` (see
     [`Apply()` Parameters](#apply-parameters).
   * `WHMatType` (default `arma::mat`) should be set to the type of `W` and `H`
     (see [`Apply()` Parameters](#apply-parameters)).
   * `tolerance` (a `double`) specifies the relative nonzero residual tolerance
     for convergence.
   * `maxIterations` (a `size_t`) specifies the maximum number of iterations
     before termination.
   * `reverseStepTolerance` (a `size_t`) specifies the number of iterations
     where the relative nonzero residual must be below the tolerance for
     convergence.
 - The best `W` and `H` matrices (according to the nonzero residual) from the
   final `reverseStepTolerance` iterations are returned by `Apply()`.

***`ValidationRMSETermination<MatType>`***:

 - Holds out a validation set of nonzero elements from `V`, and terminates when
   the RMSE (root mean squared error) on this validation set is sufficiently
   small between iterations.
 - The validation RMSE must remain below the threshold for a specified number of
   iterations.
 - `MatType` should be set to the type of `V` (see
   [`Apply()` Parameters](#apply-parameters)).
 - Constructor: `ValidationRMSETermination<MatType>(V, numValPoints, tolerance=1e-5, maxIterations=10000, reverseStepTolerance=3)`
   * `V` is the matrix to be decomposed by `Apply()`.  This will be modified
     (validation elements will be removed).
   * `numValPoints` (a `size_t`) specifies number of test points from `V` to be
     held out.
   * `tolerance` (a `double`) specifies the relative tolerance for the
     validation RMSE for termination.
   * `maxIterations` (a `size_t`) specifies the maximum number of iterations
     before termination.
   * `reverseStepTolerance` (a `size_t`) specifies the number of iterations
     where the validation RMSE must be below the tolerance for convergence.
 - The best `W` and `H` matrices (according to the validation RMSE) from the
   final `reverseStepTolerance` iterations are returned by `Apply()`.

***Custom policies***:

 - A custom class for termination behavior must implement the following
   functions.

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
};
```

---

#### `InitializationRuleType`

 * Specifies the strategy to use to initialize `W` and `H` at the beginning of
   the NMF algorithm.
 * An initialization `InitializationRuleType` can be passed to the following
   constructor:
   - `nmf = NMF(terminationPolicy, initializationRule)`
 * The following choices are available for drop-in usage:

***`RandomAcolInitialization<N>`*** (default):

 - Initialize `W` by averaging `N` randomly chosen columns of `V`.
 - Initialize `H` as uniform random in the range `[0, 1]`.
 - The default value for `N` is 5.
 - See also [the paper](https://arxiv.org/abs/1407.7299) describing the
   strategy.

***`NoInitialization`***:

 - When `nmf.Apply(V, rank, W, H)`, the existing values of `W` and `H` will be
   used.
 - If `W` is not of size `V.n_rows` x `rank`, or if `H` is not of size `rank` x
   `V.n_cols`, a `std::invalid_argument` exception will be thrown.

***`GivenInitialization<MatType>`***:

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

***`RandomInit`***:

 - Initialize `W` and `H` as uniform random in the range `[0, 1]`.

***`AverageInitialization`***:

 - Initialize each element of `W` and `H` to the square root of the average
   value of `V`, adding uniform random noise in the range `[0, 1]`.

***`MergeInitialization<WRule, HRule>`***:

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

***Custom rules***:

 - A custom class for initializing `W` and `H` must implement the following
   functions.

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

---

### Advanced Functionality Examples

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
H.fill(1.0);

mlpack::NMF<NoInitialization> nmf;
const double rmse = nmf.Apply(V, 15, W, H);

std::cout << "RMSE of NMF decomposition with pre-specified W and H: " << rmse
    << "." << std::endl;
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

// Create the NMF object with the instantiated termination policy.
mlpack::NMF<mlpack::ValidationRMSETermination<arma::sp_mat>> nmf(t);

// Perform NMF with a rank of 20.
// Note the RMSE returned here is the RMSE on V itself, but *not* the validation
// set.
const double rmse = nmf.Apply(V, 20, W, H);

std::cout << "Validation RMSE: " << t.RMSE() << "." << std::endl;
```

---

Use a custom termination policy that sets a limit on how long NMF is allowed to
take.  First, we define the termination policy:

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
    c.tic();
  }

  template<typename WHMatType>
  bool IsConverged(const WHMatType& /* W */, const WHMatType& /* H */)
  {
    totalTime += c.toc();
    c.tic();
    return (totalTime > totalAllowedTime);
  }

 private:
  double totalAllowedTime;
  double totalTime;
  arma::wall_clock c; // used for convenient timing
};
```

Then we can use it in the test program:

```c++
// See https://datasets.mlpack.org/movielens-100k.csv.
arma::sp_fmat V;
mlpack::data::Load("movielens-100k.csv", V, true);

CustomTimeTermination t(5 /* seconds */);
mlpack::NMF<CustomTimeTermination> nmf(t);

arma::fmat W, H;
const double rmse = nmf.Apply(V, 10, W, H);

std::cout << "RMSE after 5 seconds: " << rmse << "." << std::endl;
```
