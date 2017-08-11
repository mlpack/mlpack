/*! @page function The FunctionType policy in mlpack

@section Overview

\b To represent the various types of loss functions encountered in machine
learning problems, mlpack provides the \c FunctionType template parameter in
the optimizer interface. The various optimizers available in the core library
rely on this policy to gain the necessary information required by the optimizing
algorithm.

The \c FunctionType template parameter required by the Optimizer class can have
additional requirements imposed on it, depending on the type of optimizer used.

@section Interface requirements

The most basic requirements for the \c FunctionType parameter are the
implementations of two public member functions, with the following interface
and semantics

@code
double Evaluate(const arma::mat& coordinates);
@endcode

To evaluate the loss function at the given coordinates.

@code
void Gradient(const arma::mat& coordinates, arma::mat& gradient);
@endcode

To evaluate the gradient at the given coordinates, where \c gradient is an
out-param for the required gradient.

Optimizers like SGD and RMSProp require a \c DecomposableFunctionType having the
following requirements

@code
size_t NumFunctions();
@endcode

Return the number of functions. In a data-dependent function, this would return
the number of points in the dataset.

@code
double Evaluate(const arma::mat& coordinates, const size_t i);
@endcode

Evaluate the \c i th loss function. For example, for a data-dependent function,
Evaluate(coordinates, 0) should evaluate the loss function at the first point
in the dataset.

@code 
void Gradient(const arma::mat& coordinates, const size_t i, arma::mat& gradient);
@endcode

Evaluate the gradient of the \c i th loss function.


\c ParallelSGD optimizer requires a \c SparseFunctionType interface.
The only difference between the above \c DecomposableFunctionType and
\c SparseFunctionType interface is the type of the out-param used in the 
\c Gradient function. \c SparseFunctionType requires the gradient to be in a
sparse matrix (\c arma::sp_mat), as ParallelSGD, implemented with the HOGWILD!
scheme of unsynchronised updates, is expected to be relavant only in situations
where the individual gradients are sparse.


The \c SCD optimizer requires a \c ResolvableFunctionType interface, to calculate
partial gradients with respect to individual features. The interface expects the
following member functions from the function class

@code
size_t NumFeatures();
@endcode

Return the number of features in the decision variable.

@code
double Evaluate(const arma::mat& coordinates);
@endcode

To evaluate the loss function at the given coordinates, same as the
\c FunctionType interface.

@code
void FeatureGradient(const arma::mat& coordinates, const size_t j, arma::sp_mat& gradient);
@endcode

To evaluate the gradient at the given coordinates, where \c gradient is an
out-param for the required gradient. The out-param is a sparse matrix.
*/
