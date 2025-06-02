# The KernelType policy in mlpack

Kernel methods make up a large class of machine learning techniques.  Each of
these methods is characterized by its dependence on a *kernel function*.  In
rough terms, a kernel function is a general notion of similarity between two
points, with its value large when objects are similar and its value small when
objects are dissimilar (note that this is not the only interpretation of what a
kernel is).

A kernel (or 'Mercer kernel') `K(a, b)` takes two objects as input and returns
some sort of similarity value.  The specific details and properties of kernels
are outside the scope of this documentation; for a better introduction to
kernels and kernel methods, there are numerous better resources available,
including
[Eric Kim's tutorial](http://www.eric-kim.net/eric-kim-net/posts/1/kernel_trick.html).

mlpack implements a number of kernel methods and, accordingly, each of these
methods allows arbitrary kernels to be used via the `KernelType` template
parameter.  Like the [DistanceType policy](distances.md), the requirements are
quite simple: a class implementing the `KernelType` policy must have

 - an `Evaluate()` function
 - a default constructor

The signature of the `Evaluate()` function is straightforward:

```c++
template<typename VecTypeA, typename VecTypeB>
double Evaluate(const VecTypeA& a, const VecTypeB& b);
```

The function takes two vector arguments, `a` and `b`, and returns a `double`
that is the evaluation of the kernel between the two arguments.  So, for a
particular kernel `K`, the `Evaluate()` function should return `K(a, b)`.

The arguments `a` and `b`, of types `VecTypeA` and `VecTypeB`, respectively,
will be an Armadillo-like vector type (usually `arma::vec`, `arma::sp_vec`, or
similar).  In general it should be valid to assume that `VecTypeA` is a class
with the same API as `arma::vec`.

Note that for kernels that do not hold any state, the `Evaluate()` method can be
marked as `static`.

Overall, the `KernelType` template policy is quite simple (much like the
[DistanceType policy](distances.md)).  Below is an example kernel class, which
outputs `1` if the vectors are close and `0` otherwise.

```c++
class ExampleKernel
{
  // Default constructor is required.
  ExampleKernel() { }

  // The example kernel holds no state, so we can mark Evaluate() as static.
  template<typename VecTypeA, typename VecTypeB>
  static double Evaluate(const VecTypeA& a, const VecTypeB& b)
  {
    // Get how far apart the vectors are (using the Euclidean distance).
    const double distance = arma::norm(a - b);

    if (distance < 0.05) // Less than 0.05 distance is "close".
      return 1;
    else
      return 0;
  }
};
```

Then, this kernel may be easily used inside of mlpack algorithms.  For instance,
the code below runs kernel PCA (`KernelPCA`) on a random dataset using the
`ExampleKernel`.  The results are saved to a file called `results.csv`.  (Note
that this is simply an example to demonstrate usage, and this example kernel
isn't actually likely to be useful in practice.)

```c++
#include <mlpack.hpp>
#include "example_kernel.hpp" // Contains the ExampleKernel class.

using namespace mlpack;
using namespace arma;

int main()
{
  // Generate the random dataset; 10 dimensions, 5000 points.
  mat dataset = randu<mat>(10, 5000);

  // Instantiate the KernelPCA object with the ExampleKernel kernel type.
  KernelPCA<ExampleKernel> kpca;

  // The dataset will be transformed using kernel PCA with the example kernel to
  // contain only 2 dimensions.
  kpca.Apply(dataset, 2);

  // Save the results to 'results.csv'.
  data::Save(dataset, "results.csv");
}
```

## The `KernelTraits` trait class

Some algorithms that use kernels can specialize if the kernel fulfills some
certain conditions.  An example of a condition might be that the kernel is
shift-invariant or that the kernel is normalized.  In the case of fast
max-kernel search (`mlpack::fastmks::FastMKS`), the computation can be
accelerated if the kernel is normalized.  For this reason, the `KernelTraits`
trait class exists.  This allows a kernel to specify via a `const static bool`
when these types of conditions are satisfied.  *Note that a KernelTraits class
is not required,* but may be helpful.

The `KernelTraits` trait class is a template class that takes a `KernelType` as
a parameter, and exposes `const static bool` values that depend on the kernel.
Setting these values is achieved by specialization.  The code below provides an
example, specializing `KernelTraits` for the `ExampleKernel` from earlier:

```c++
template<>
class KernelTraits<ExampleKernel>
{
 public:
  //! The example kernel is normalized (K(x, x) = 1 for all x).
  const static bool IsNormalized = true;
};
```

At this time, there is only one kernel trait that is used in mlpack code:

 - `IsNormalized` (defaults to `false`): if `K(x, x) = 1` for all `x`,
   then the kernel is normalized and this should be set to `true`.

## List of kernels and classes that use a `KernelType`

mlpack comes with a number of pre-implemented and ready-to-use kernels:

 * [`GaussianKernel`](../user/core/kernels.md#gaussiankernel): standard
   Gaussian/radial basis function/RBF kernel
 * [`CauchyKernel`](../user/core/kernels.md#cauchykernel): Cauchy kernel, with
   longer tails than the standard Gaussian kernel
 * [`CosineSimilarity`](../user/core/kernels.md#cosinesimilarity): dot-product
   vector similarity
 * [`EpanechnikovKernel`](../user/core/kernels.md#epanechnikovkernel):
   Epanechnikov kernel (parabolic), with zero tails
 * [`HyperbolicTangentKernel`](../user/core/kernels.md#hyperbolictangentkernel):
   hyperbolic tangent kernel (not positive definite)
 * [`LaplacianKernel`](../user/core/kernels.md#laplaciankernel): Laplacian
   kernel/exponential kernel
 * [`LinearKernel`](../user/core/kernels.md#linearkernel): linear (dot-product)
   kernel
 * [`PolynomialKernel`](../user/core/kernels.md#polynomialkernel):
   arbitrary-power polynomial kernel with offset
 * [`PSpectrumStringKernel`](../user/core/kernels.md#pspectrumstringkernel):
   kernel to compute length-p subsequence match counts
 * [`SphericalKernel`](../user/core/kernels.md#sphericalkernel):
   spherical/uniform/rectangular window kernel
 * [`TriangularKernel`](../user/core/kernels.md#triangularkernel): triangular
   kernel, with zero tails
 * [Implement a custom kernel](../user/core/kernels.md#implement-a-custom-kernel)

These kernels (or a custom kernel) may be used in a variety of mlpack methods:

 * [`KDE`](/src/mlpack/methods/kde/kde.hpp)
 * [`MeanShift`](/src/mlpack/methods/mean_shift/mean_shift.hpp)
 * [`KernelPCA`](/src/mlpack/methods/kernel_pca/kernel_pca.hpp)
 * [`FastMKS`](/src/mlpack/methods/fastmks/fastmks.hpp)
 * [`NystroemMethod`](/src/mlpack/methods/nystroem_method/nystroem_method.hpp)
