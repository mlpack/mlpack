/*! @page matrices Matrices in mlpack

@section matintro Introduction

mlpack uses Armadillo matrices for matrix support.  Armadillo is a fast C++
matrix library which makes use of advanced template techniques to provide the
fastest possible matrix operations.

Documentation on Armadillo can be found on their website:

http://arma.sourceforge.net/docs.html

Nonetheless, there are a few further caveats for mlpack Armadillo usage.

@section format Column-major Matrices

Armadillo matrices are stored in a column-major format; this means that on disk,
each column is located in contiguous memory.

This means that, for the vast majority of machine learning methods, it is faster
to store observations as columns and dimensions as rows.  This is counter to
most standard machine learning texts!

Major implications of this are for linear algebra.  For instance, the covariance
of a matrix is typically

@f$ C = X^T X @f$

but for a column-wise matrix, it is

@f$ C = X X^T @f$

and this is very important to keep in mind!  If your mlpack code is not working,
this may be a factor in why.

@section loading Loading Matrices

mlpack provides a data::Load() and data::Save() function, which should be used
instead of Armadillo's loading and saving functions.

Most machine learning data is stored in row-major format; a CSV, for example,
will generally have one observation per line and each column will correspond to
a dimension.

The data::Load() and data::Save() functions transpose the matrix upon loading,
meaning that the following CSV:

@code
$ cat data.csv
3,3,3,3,0
3,4,4,3,0
3,4,4,3,0
3,3,4,3,0
3,6,4,3,0
2,4,4,3,0
2,4,4,1,0
3,3,3,2,0
3,4,4,2,0
3,4,4,2,0
3,3,4,2,0
3,6,4,2,0
2,4,4,2,0
@endcode

is actually loaded with 5 rows and 13 columns, not 13 rows and 5 columns like
the CSV is written.

This is important to remember!

*/
