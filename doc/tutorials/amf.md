# Alternating Matrix Factorization Tutorial

Alternating Matrix Factorization (AMF) decomposes a matrix `V` into the product `V ≈ WH`, where:

- `W` is the **basis matrix** of size `n × r`.
- `H` is the **encoding matrix** of size `r × m`.

Here, `V` is an `n × m` matrix, and `r` is the **rank** of the factorization. The factorization process alternates between calculating `W` and `H`, keeping the other matrix constant during each step.

mlpack provides a simple C++ interface to perform Alternating Matrix
Factorization.

## The `AMF` Class

The `AMF` class is templated with three parameters:

1. **Termination Policy**: Determines when the algorithm has converged.
2. **Initialization Rule**: Specifies how the `W` and `H` matrices are initialized.
3. **Update Rule**: Defines the update mechanism used during each iteration.

This templated design allows users to experiment with various update rules, initialization strategies, and termination criteria, including custom policies not provided by mlpack.

### Method Overview

The class provides the following method that performs factorization

```cpp
template<typename MatType>
double Apply(const MatType& V,
            const size_t r,
            arma::mat& W,
            arma::mat& H);
```

- **Parameters**:
  - `V`: The input matrix to factorize.
  - `r`: The rank of the factorization.
  - `W`: Output basis matrix.
  - `H`: Output encoding matrix.

- **Returns**: The residue obtained by comparing `W * H` with the original matrix `V`.

## Using Different Termination Policies

The `AMF` implementation comes with different termination policies to support
many implemented algorithms. Every termination policy implements the following
method which returns the status of convergence.

```cpp
bool IsConverged(const arma::mat& W, const arma::mat& H);
```

### Available Termination Policies

- **`SimpleResidueTermination`**
- **`SimpleToleranceTermination`**
- **`ValidationRMSETermination`**

#### `SimpleResidueTermination`
Termination is based on two factors:
- The residue value drops below a predefined threshold.
- The number of iterations exceeds a specified limit.

If either condition is met, the algorithm terminates.

#### `SimpleToleranceTermination`
The algorithm terminates when the increase in residue falls below a given tolerance. To handle occasional spikes, a certain number of successive residue decreases are tolerated. Additionally, the algorithm will terminate if the iteration count surpasses the threshold.

#### `ValidationRMSETermination`
`ValidationRMSETermination` divides the data into two sets: training and validation. Entries of the validation set are nullified in the input matrix. The termination criterion is met when the increase in the validation set RMSE value drops below a given tolerance. To accommodate spikes, a certain number of successive validation RMSE drops are accepted. This upper limit on successive drops can be adjusted with the `reverseStepCount` parameter. A secondary termination criterion terminates the algorithm when the iteration count exceeds the threshold. Although this termination policy provides a better measure of convergence than the other two policies, it may decrease performance due to its computational expense.

On the other hand, `CompleteIncrementalTermination` and
`IncompleteIncrementalTermination` are just wrapper classes for other
termination policies. These policies are used when AMF is applied with
`SVDCompleteIncrementalLearning` and `SVDIncompleteIncrementalLearning`,
respectively.

## Using Different Initialization Policies

mlpack currently supports two initialization policies for AMF:

- **`RandomInitialization`**: Initializes matrices `W` and `H` with a random uniform distribution.
- **`RandomAcolInitialization`**: Initializes the `W` matrix by averaging `p` randomly chosen columns of `V`. Here, `p` is a template parameter.

### Custom Initialization Policies

To implement a custom initialization policy, users must define the following static method within their class:

```cpp
template<typename MatType>
inline static void Initialize(const MatType& V,
                              const size_t r,
                              arma::mat& W,
                              arma::mat& H)
```

This method should handle the initialization of `W` and `H` based on the provided matrix `V` and rank `r`.

## Using Different Update Rules

mlpack implements the following update rules for the AMF class:

- **Non-Negative Matrix Factorization (NMF) Updates**:
  - `NMFALSUpdate`
  - `NMFMultiplicativeDistanceUpdate`
  - `NMFMultiplicativeDivergenceUpdate`

- **Singular Value Decomposition (SVD) Updates**:
  - `SVDBatchLearning`
  - `SVDIncompleteIncrementalLearning`
  - `SVDCompleteIncrementalLearning`

### Non-Negative Matrix Factorization (NMF)

NMF can be achieved using the following update rules:

- **`NMFALSUpdate`**: Implements a simple Alternating Least Squares optimization.
- **`NMFMultiplicativeDistanceUpdate`**: Utilizes a multiplicative update rule based on distance measures.
- **`NMFMultiplicativeDivergenceUpdate`**: Employs a multiplicative update rule based on divergence measures.

These update rules are based on algorithms described in the paper [Algorithms for Non-negative Matrix Factorization](https://proceedings.neurips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf).

### Singular Value Decomposition (SVD)

The remaining update rules perform Singular Value Decomposition (SVD) of the matrix `V`. These SVD factorizations are optimized for mlpack's collaborative filtering functionalities, as detailed in the [Collaborative Filtering Tutorial](https://github.com/mlpack/mlpack/wiki/CollaborativeFiltering).

For more detailed algorithmic explanations, refer to the respective class documentation.

## Using Non-Negative Matrix Factorization with `AMF`

The use of `AMF` for Non-Negative Matrix factorization is simple. The AMF module
defines `NMFALSFactorizer` which can be used directly without knowing the
internal structure of `AMF`. For example:

```cpp
#include <mlpack.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;

int main()
{
  // Create a random 100x100 matrix with uniform distribution.
  mat V = randu<mat>(100, 100);

  // Desired rank for factorization.
  size_t r = 10;

  // Matrices to store the factorization results.
  mat W, H;

  // Instantiate the NMF ALS factorizer.
  NMFALSFactorizer nmf;

  // Perform factorization and retrieve the residue.
  double residue = nmf.Apply(V, r, W, H);

  // Output the residue.
  cout << "Residue: " << residue << endl;

  return 0;
}
```

**Notes**:
- `NMFALSFactorizer` uses `SimpleResidueTermination` by default, which is well-suited for NMF tasks.
- The initialization of `W` and `H` is random.
- The `Apply()` function returns the residue, which measures the difference between `W * H` and the original matrix `V`.

## Using Singular Value Decomposition with `AMF`

mlpack has the following SVD factorizers implemented for AMF:

- **`SVDBatchFactorizer`**
- **`SVDIncompleteIncrementalFactorizer`**
- **`SVDCompleteIncrementalFactorizer`**

Each factorizer accepts a template parameter `MatType`, specifying the type of the input matrix `V` (either dense `arma::mat` or sparse `arma::sp_mat`). For sparse matrices, using `arma::sp_mat` can significantly enhance runtime performance.

### Example

```cpp
#include <mlpack.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;

int main()
{
  // Create a random 100x100 sparse matrix with 10% non-zero entries.
  sp_mat V = sprandu<sp_mat>(100, 100, 0.1);

  // Desired rank for factorization.
  size_t r = 10;

  // Matrices to store the factorization results.
  mat W, H;

  // Instantiate the SVD Batch Factorizer for sparse matrices.
  SVDBatchFactorizer<sp_mat> svd;

  // Perform factorization and retrieve the residue.
  double residue = svd.Apply(V, r, W, H);

  // Output the residue.
  cout << "Residue: " << residue << endl;

  return 0;
}
```

**Notes**:
- Choose the appropriate factorizer based on your specific use case and data characteristics.
- For collaborative filtering applications, refer to the [Collaborative Filtering Tutorial](https://github.com/mlpack/mlpack/wiki/CollaborativeFiltering) for optimized usage.

## Further Documentation

For more in-depth information about the `AMF` class and its components, please refer to the [AMF Source Code Documentation](https://mlpack.org/doc/mlpack-<version>/classmlpack_1_1amf_1_1AMF.html).

# Additional Resources

- [mlpack Official Documentation](https://www.mlpack.org/doc/index.html)
- [Collaborative Filtering Tutorial](https://github.com/mlpack/mlpack/wiki/CollaborativeFiltering)

# Conclusion

This tutorial introduced mlpack's `AMF` class for Alternating Matrix Factorization. By leveraging various termination policies, initialization methods, and update rules, you can customize the factorization process for applications like Non-Negative Matrix Factorization and Singular Value Decomposition.

For more details, visit the [mlpack documentation](https://www.mlpack.org/doc/index.html) or join the community forums.