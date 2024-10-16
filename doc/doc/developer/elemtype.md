# The ElemType policy in mlpack

mlpack algorithms should be as generic as possible.  Often this means
allowing arbitrary metrics or kernels to be used, but this also means allowing
any type of data point to be used.  This means that mlpack classes should
support `float`, `double`, and other observation types.  Some algorithms
support this through the use of a `MatType` template parameter; others will
have their own template parameter, `ElemType`.

The `ElemType` template parameter can take any value that can be used by
Armadillo (or, specifically, classes like `arma::Mat<>` and others); this
encompasses the types

 - `double`
 - `float`
 - `int`
 - `unsigned int`
 - `std::complex<double>`
 - `std::complex<float>`

and other primitive numeric types.  Note that Armadillo does not support some
integer types for functionality such as matrix decompositions or other more
advanced linear algebra.  This means that when these integer types are used,
some algorithms may fail with Armadillo error messages indicating that those
types cannot be used.

*Note*: if the class has a `MatType` template parameter, `ElemType` can be
easily defined as below:

```c++
typedef typename MatType::elem_type ElemType;
```

and otherwise a template parameter with the name `ElemType` can be used.  It is
generally a good idea to expose the element type somehow for use by other
classes.
