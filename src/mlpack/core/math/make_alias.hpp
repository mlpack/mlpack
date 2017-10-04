/**
 * @file make_alias.hpp
 * @author Ryan Curtin
 *
 * Make an alias of a matrix.  For sparse matrices, unfortunately no alias can
 * be made and a copy must be incurred.
 */
#ifndef MLPACK_CORE_MATH_MAKE_ALIAS_HPP
#define MLPACK_CORE_MATH_MAKE_ALIAS_HPP

namespace mlpack {
namespace math {

/**
 * Make an alias of a dense matrix.
 */
template<typename ElemType>
arma::Mat<ElemType> MakeAlias(arma::Mat<ElemType>& input)
{
  // Use the advanced constructor.
  return arma::Mat<ElemType>(input.memptr(), input.n_rows, input.n_cols, false,
      true);
}

/**
 * Make an alias of a dense row.
 */
template<typename ElemType>
arma::Row<ElemType> MakeAlias(arma::Row<ElemType>& input)
{
  // Use the advanced constructor.
  return arma::Row<ElemType>(input.memptr(), input.n_elem, false, true);
}

/**
 * Make an alias of a dense column.
 */
template<typename ElemType>
arma::Col<ElemType> MakeAlias(arma::Col<ElemType>& input)
{
  // Use the advanced constructor.
  return arma::Col<ElemType>(input.memptr(), input.n_elem, false, true);
}

/**
 * Make a copy of a sparse matrix (an alias is not possible).
 */
template<typename ElemType>
arma::SpMat<ElemType> MakeAlias(const arma::SpMat<ElemType>& input)
{
  // Make a copy...
  return arma::SpMat<ElemType>(input);
}

/**
 * Make a copy of a sparse row (an alias is not possible).
 */
template<typename ElemType>
arma::SpRow<ElemType> MakeAlias(const arma::SpRow<ElemType>& input)
{
  // Make a copy...
  return arma::SpRow<ElemType>(input);
}

/**
 * Make a copy of a sparse column (an alias is not possible).
 */
template<typename ElemType>
arma::SpCol<ElemType> MakeAlias(const arma::SpCol<ElemType>& input)
{
  // Make a copy...
  return arma::SpCol<ElemType>(input);
}

} // namespace math
} // namespace mlpack

#endif
