/**
 * @file core/math/make_alias.hpp
 * @author Ryan Curtin
 *
 * Make an alias of a matrix.  For sparse matrices, unfortunately no alias can
 * be made and a copy must be incurred.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MATH_MAKE_ALIAS_HPP
#define MLPACK_CORE_MATH_MAKE_ALIAS_HPP

namespace mlpack {

/**
 * Make an alias of a dense cube.  If strict is true, then the alias cannot be
 * resized or pointed at new memory.
 */
template<typename ElemType>
arma::Cube<ElemType> MakeAlias(arma::Cube<ElemType>& input,
                               const bool strict = true)
{
  // Use the advanced constructor.
  return arma::Cube<ElemType>(input.memptr(), input.n_rows, input.n_cols,
      input.n_slices, false, strict);
}

/**
 * Make an alias of a dense matrix.  If strict is true, then the alias cannot be
 * resized or pointed at new memory.
 */
template<typename ElemType>
arma::Mat<ElemType> MakeAlias(arma::Mat<ElemType>& input,
                              const bool strict = true)
{
  // Use the advanced constructor.
  return arma::Mat<ElemType>(input.memptr(), input.n_rows, input.n_cols, false,
      strict);
}

/**
 * Make an alias of a dense row.  If strict is true, then the alias cannot be
 * resized or pointed at new memory.
 */
template<typename ElemType>
arma::Row<ElemType> MakeAlias(arma::Row<ElemType>& input,
                              const bool strict = true)
{
  // Use the advanced constructor.
  return arma::Row<ElemType>(input.memptr(), input.n_elem, false, strict);
}

/**
 * Make an alias of a dense column.  If strict is true, then the alias cannot be
 * resized or pointed at new memory.
 */
template<typename ElemType>
arma::Col<ElemType> MakeAlias(arma::Col<ElemType>& input,
                              const bool strict = true)
{
  // Use the advanced constructor.
  return arma::Col<ElemType>(input.memptr(), input.n_elem, false, strict);
}

/**
 * Make a copy of a sparse matrix (an alias is not possible).  The strict
 * parameter is ignored.
 */
template<typename ElemType>
arma::SpMat<ElemType> MakeAlias(const arma::SpMat<ElemType>& input,
                                const bool /* strict */ = true)
{
  // Make a copy...
  return arma::SpMat<ElemType>(input);
}

/**
 * Make a copy of a sparse row (an alias is not possible).  The strict
 * parameter is ignored.
 */
template<typename ElemType>
arma::SpRow<ElemType> MakeAlias(const arma::SpRow<ElemType>& input,
                                const bool /* strict */ = true)
{
  // Make a copy...
  return arma::SpRow<ElemType>(input);
}

/**
 * Make a copy of a sparse column (an alias is not possible).  The strict
 * parameter is ignored.
 */
template<typename ElemType>
arma::SpCol<ElemType> MakeAlias(const arma::SpCol<ElemType>& input,
                                const bool /* strict */ = true)
{
  // Make a copy...
  return arma::SpCol<ElemType>(input);
}

/**
 * Clear an alias so that no data is overwritten.  This resets the matrix if it
 * is an alias (and does nothing otherwise).
 */
template<typename ElemType>
void ClearAlias(arma::Mat<ElemType>& mat)
{
  if (mat.mem_state >= 1)
    mat.reset();
}

/**
 * Clear an alias for a sparse matrix.  This does nothing because no sparse
 * matrices can have aliases.
 */
template<typename ElemType>
void ClearAlias(arma::SpMat<ElemType>& /* mat */)
{
  // Nothing to do.
}


} // namespace mlpack

#endif
