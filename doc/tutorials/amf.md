# Alternating Matrix Factorization tutorial

Alternating matrix factorization decomposes a matrix `V` in the form `V ~ WH`
where `W` is called the basis matrix and `H` is called the encoding matrix.. `V`
is taken to be of size `n x m` and the obtained `W` is `n x r` and `H` is `r x
m`. The size `r` is called the *rank* of the factorization. Factorization is
done by alternately calculating `W` and `H` respectively while holding the other
matrix constant.

mlpack provides a simple C++ interface to perform Alternating Matrix
Factorization.

## The `AMF` class

The `AMF` class is templatized with 3 parameters; the first contains the policy
used to determine when the algorithm has converged; the second contains the
initialization rule for the `W` and `H` matrix; the last contains the update
rule to be used during each iteration. This templatization allows the user to
try various update rules, initialization rules, and termination policies
(including ones not supplied with mlpack) for factorization.

The class provides the following method that performs factorization

```c++
template<typename MatType> double Apply(const MatType& V,
                                        const size_t r,
                                        arma::mat& W,
                                        arma::mat& H);
```

## Using different termination policies

The `AMF` implementation comes with different termination policies to support
many implemented algorithms. Every termination policy implements the following
method which returns the status of convergence.

```c++
bool IsConverged(arma::mat& W, arma::mat& H)
```

Below is a list of all the termination policies that mlpack contains.

 - `SimpleResidueTermination`
 - `SimpleToleranceTermination`
 - `ValidationRMSETermination`

In `SimpleResidueTermination`, the termination decision depends on two factors,
value of residue and number of iteration. If the current value of residue drops
below the threshold or the number of iterations goes beyond the threshold,
positive termination signal is passed to AMF.

In `SimpleToleranceTermination`, termination criterion is met when the increase
in residue value drops below the given tolerance. To accommodate spikes, certain
number of successive residue drops are accepted. Secondary termination criterion
terminates algorithm when iteration count goes beyond the threshold.

`ValidationRMSETermination` divides the data into 2 sets, training set and
validation set. Entries of the validation set are nullifed in the input matrix.
Termination criterion is met when increase in validation set RMSe value drops
below the given tolerance. To accommodate spikes certain number of successive
validation RMSE drops are accepted. This upper imit on successive drops can be
adjusted with `reverseStepCount`. A secondary termination criterion terminates
the algorithm when the iteration count goes above the threshold. Though this
termination policy is better measure of convergence than the above 2 termination
policies, it may cause a decrease in performance since it is computationally
expensive.

On the other hand, `CompleteIncrementalTermination` and
`IncompleteIncrementalTermination` are just wrapper classes for other
termination policies. These policies are used when AMF is applied with
`SVDCompleteIncrementalLearning` and `SVDIncompleteIncrementalLearning`,
respectively.

## Using different initialization policies

mlpack currently has 2 initialization policies implemented for AMF:

 - `RandomInitialization`
 - `RandomAcolInitialization`

`RandomInitialization` initializes matrices `W` and `H` with random uniform
distribution while `RandomAcolInitialization` initializes the `W` matrix by
averaging p randomly chosen columns of `V`.  In the case of
`RandomAcolInitialization`, `p` is a template parameter.

To implement their own initialization policy, users need to define the following
function in their class.

```c++
template<typename MatType>
inline static void Initialize(const MatType& V,
                              const size_t r,
                              arma::mat& W,
                              arma::mat& H)
```

## Using different update rules

mlpack implements the following update rules for the AMF class:

 - `NMFALSUpdate`
 - `NMFMultiplicativeDistanceUpdate`
 - `NMFMultiplicativeDivergenceUpdate`
 - `SVDBatchLearning`
 - `SVDIncompleteIncrementalLearning`
 - `SVDCompleteIncrementalLearning`

Non-Negative Matrix factorization can be achieved with `NMFALSUpdate`,
`NMFMultiplicativeDivergenceUpdate` or `NMFMultiplicativeDivergenceUpdate`.
`NMFALSUpdate` implements a simple Alternating Least Squares optimization while
the other rules implement algorithms given in the paper 'Algorithms for
Non-negative Matrix Factorization'.

The remaining update rules perform the singular value decomposition of the
matrix `V`.  This SVD factorization is optimized for use by mlpack's
collaborative filtering code (see the [collaborative filtering
tutorial](cf.md)). This use of SVD factorizers for collaborative filtering is
described in the paper 'A Guide to Singular Value Decomposition for
Collaborative Filtering' by Chih-Chao Ma. For further details about the
algorithms refer to the respective class documentation.

## Using Non-Negative Matrix Factorization with `AMF`

The use of `AMF` for Non-Negative Matrix factorization is simple. The AMF module
defines `NMFALSFactorizer` which can be used directly without knowing the
internal structure of `AMF`. For example:

```c++
#include <mlpack.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;

int main()
{
  NMFALSFactorizer nmf;
  mat W, H;
  mat V = randu<mat>(100, 100);
  double residue = nmf.Apply(V, W, H);
}
```

`NMFALSFactorizer` uses `SimpleResidueTermination`, which is most preferred with
Non-Negative Matrix factorizers.  The initialization of `W` and `H` in
`NMFALSFactorizer` is random. The `Apply()` function returns the residue
obtained by comparing the constructed matrix `W * H` with the original matrix
`V`.

## Using Singular Value Decomposition with `AMF`

mlpack has the following SVD factorizers implemented for AMF:

 - `SVDBatchFactorizer`
 - `SVDIncompleteIncrementalFactorizer`
 - `SVDCompleteIncrementalFactorizer`

Each of these factorizers takes a template parameter `MatType`, which specifies
the type of the matrix `V` (dense or sparse---these have types `arma::mat` and
`arma::sp_mat`, respectively).  When the matrix to be factorized is relatively
sparse, specifying `MatType = arma::sp_mat` can provide a runtime boost.

```c++
#include <mlpack.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;

int main()
{
  sp_mat V = randu<sp_mat>(100,100);
  mat W, H;

  SVDBatchFactorizer<sp_mat> svd;
  double residue = svd.Apply(V, W, H);
}
```

## Further documentation

For further documentation on the `AMF` class, consult the `AMF`
source code comments.
