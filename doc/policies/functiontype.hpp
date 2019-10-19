/*! @page function The FunctionType policy in mlpack

@section Overview

To represent the various types of loss functions encountered in machine
learning problems, mlpack provides the \c FunctionType template parameter in
the optimizer interface. The various optimizers available in the core library
rely on this policy to gain the necessary information required by the optimizing
algorithm.

The \c FunctionType template parameter required by the Optimizer class can have
additional requirements imposed on it, depending on the type of optimizer used.

@section requirements Interface requirements

The most basic requirements for the \c FunctionType parameter are the
implementations of two public member functions, with the following interface
and semantics

@code
// Evaluate the loss function at the given coordinates.
double Evaluate(const arma::mat& coordinates);
@endcode


@code
// Evaluate the gradient at the given coordinates, where 'gradient' is an
// output parameter for the required gradient.
void Gradient(const arma::mat& coordinates, arma::mat& gradient);
@endcode


Optimizers like SGD and RMSProp require a \c DecomposableFunctionType having the
following requirements

@code
// Return the number of functions. In a data-dependent function, this would
// return the number of points in the dataset.
size_t NumFunctions();
@endcode


@code
// Evaluate the 'i' th loss function. For example, for a data-dependent
// function, Evaluate(coordinates, 0) should evaluate the loss function at the
// first point in the dataset.
double Evaluate(const arma::mat& coordinates, const size_t i);
@endcode

@code
// Evaluate the gradient of the 'i' th loss function at the given coordinates,
// where 'gradient' is an output parameter for the required gradient.
void Gradient(const arma::mat& coordinates, const size_t i, arma::mat& gradient);
@endcode



\c ParallelSGD optimizer requires a \c SparseFunctionType interface.
\c SparseFunctionType requires the gradient to be in a sparse matrix (\c
arma::sp_mat), as ParallelSGD, implemented with the HOGWILD!  scheme of
unsynchronised updates, is expected to be relevant only in situations where the
individual gradients are sparse. So, the interface requires function with the
following signatures

@code
// Return the number of functions. In a data-dependent function, this would
// return the number of points in the dataset.
size_t NumFunctions();
@endcode


@code
// Evaluate the loss function at the given coordinates.
double Evaluate(const arma::mat& coordinates);
@endcode


@code
// Evaluate the (sparse) gradient of the 'i' th loss function at the given
// coordinates, where 'gradient' is an output parameter for the required
// gradient.
void Gradient(const arma::mat& coordinates, const size_t i, arma::sp_mat& gradient);
@endcode


The \c SCD optimizer requires a \c ResolvableFunctionType interface, to
calculate partial gradients with respect to individual features. The optimizer
requires the decision variable to be arranged in a particular fashion to allow
for disjoint updates. The features should be arranged columnwise in the decision
variable. For example, in \c SoftmaxRegressionFunction the decision variable has
size \c numClasses x \c featureSize (+ 1 if an intercept also needs to be fit).
Similarly, for \c LogisticRegression, the decision variable is a row vector,
with the number of columns determined by the dimensionality of the dataset.

The interface expects the following member functions from the function class

@code
// Return the number of features in the decision variable.
size_t NumFeatures();
@endcode

@code
// Evaluate the loss function at the given coordinates.
double Evaluate(const arma::mat& coordinates);
@endcode

@code
// Evaluate the partial gradient of the loss function with respect to the 'j' th
// coordinate at the given coordinates, where 'gradient' is an output parameter
// for the required gradient. The 'gradient' matrix is supposed to be non-zero
// in the jth column, which contains the relevant partial gradient.
void PartialGradient(const arma::mat& coordinates, const size_t j, arma::sp_mat& gradient);
@endcode
*/
