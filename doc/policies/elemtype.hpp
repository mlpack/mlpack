/*! @page elem The ElemType policy in mlpack

@section Overview

\b mlpack algorithms should be as generic as possible.  Often this means
allowing arbitrary metrics or kernels to be used, but this also means allowing
any type of data point to be used.  This means that \b mlpack classes should
support \c float, \c double, and other observation types.  Some algorithms
support this through the use of a \c MatType template parameter; others will
have their own template parameter, \c ElemType.

The \c ElemType template parameter can take any value that can be used by
Armadillo (or, specifically, classes like \c arma::Mat<> and others); this
encompasses the types

 - \c double
 - \c float
 - \c int
 - \c unsigned int
 - \c std::complex<double>
 - \c std::complex<float>

and other primitive numeric types.  Note that Armadillo does not support some
integer types for functionality such as matrix decompositions or other more
advanced linear algebra.  This means that when these integer types are used,
some algorithms may fail with Armadillo error messages indicating that those
types cannot be used.

@section A note for developers

If the class has a \c MatType template parameter, \c ElemType can be easily
defined as below:

@code
typedef typename MatType::elem_type ElemType;
@endcode

and otherwise a template parameter with the name \c ElemType can be used.  It is
generally a good idea to expose the element type somehow for use by other
classes.

*/
