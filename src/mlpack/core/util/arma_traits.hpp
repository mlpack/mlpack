/**
 * @file core/util/arma_traits.hpp
 * @author Ryan Curtin
 *
 * Some traits used for template metaprogramming (SFINAE) with Armadillo types.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_ARMA_TRAITS_HPP
#define MLPACK_CORE_UTIL_ARMA_TRAITS_HPP

// Structs have public members by default (that's why they are chosen over
// classes).

/**
 * If value == true, then VecType is some sort of Armadillo vector or subview.
 * You might use this struct like this:
 *
 * @code
 * // Only accepts VecTypes that are actually Armadillo vector types.
 * template<typename VecType>
 * void Function(const VecType& argumentA,
 *               typename std::enable_if_t<IsVector<VecType>::value>* = 0);
 * @endcode
 *
 * The use of the enable_if_t object allows the compiler to instantiate
 * Function() only if VecType is one of the Armadillo vector types.  It has a
 * default argument because it isn't meant to be used in either the function
 * call or the function body.
 */
template<typename VecType>
struct IsVector
{
  const static bool value = false;
};

template<typename MatType>
struct IsMatrix
{
  const static bool value = false;
};

template<typename CubeType>
struct IsCube
{
  const static bool value = false;
};

// Commenting out the first template per case, because
// Visual Studio doesn't like this instantiaion pattern (error C2910).
// template<>
template<typename eT>
struct IsVector<arma::Col<eT> >
{
  const static bool value = true;
};

// template<>
template<typename eT>
struct IsVector<arma::SpCol<eT> >
{
  const static bool value = true;
};

// template<>
template<typename eT>
struct IsVector<arma::Row<eT> >
{
  const static bool value = true;
};

// template<>
template<typename eT>
struct IsVector<arma::SpRow<eT> >
{
  const static bool value = true;
};

// template<>
template<typename eT>
struct IsVector<arma::subview_col<eT> >
{
  const static bool value = true;
};

// template<>
template<typename eT>
struct IsVector<arma::subview_row<eT> >
{
  const static bool value = true;
};

template<typename eT>
struct IsMatrix<arma::Mat<eT> >
{
  const static bool value = true;
};

template<typename eT>
struct IsCube<arma::Cube<eT> >
{
  const static bool value = true;
};


#if ((ARMA_VERSION_MAJOR >= 10) || \
    ((ARMA_VERSION_MAJOR == 9) && (ARMA_VERSION_MINOR >= 869)))

  // Armadillo 9.869+ has SpSubview_col and SpSubview_row

  template<typename eT>
  struct IsVector<arma::SpSubview_col<eT> >
  {
    const static bool value = true;
  };

  template<typename eT>
  struct IsVector<arma::SpSubview_row<eT> >
  {
    const static bool value = true;
  };

#else

  // fallback for older Armadillo versions

  template<typename eT>
  struct IsVector<arma::SpSubview<eT> >
  {
    const static bool value = true;
  };

#endif

// Get the row vector type corresponding to a given MatType.

template<typename MatType>
struct GetRowType
{
  typedef arma::Row<typename MatType::elem_type> type;
};

template<typename eT>
struct GetRowType<arma::Mat<eT>>
{
  typedef arma::Row<eT> type;
};

template<typename eT>
struct GetRowType<arma::SpMat<eT>>
{
  typedef arma::SpRow<eT> type;
};

// Get the column vector type corresponding to a given MatType.

template<typename MatType>
struct GetColType
{
  typedef arma::Col<typename MatType::elem_type> type;
};

template<typename MatType>
struct GetUColType
{
  typedef arma::Col<arma::uword> type;
};

template<typename eT>
struct GetColType<arma::Mat<eT>>
{
  typedef arma::Col<eT> type;
};

template<typename eT>
struct GetColType<arma::SpMat<eT>>
{
  typedef arma::SpCol<eT> type;
};

// Get the dense row vector type corresponding to a given MatType.

template<typename MatType>
struct GetDenseRowType
{
  typedef typename GetRowType<MatType>::type type;
};

template<typename eT>
struct GetDenseRowType<arma::SpMat<eT>>
{
  typedef arma::Row<eT> type;
};

// Get the dense column vector type corresponding to a given MatType.

template<typename MatType>
struct GetDenseColType
{
  typedef typename GetColType<MatType>::type type;
};

template<typename eT>
struct GetDenseColType<arma::SpMat<eT>>
{
  typedef arma::Col<eT> type;
};

// Get the dense matrix type corresponding to a given MatType.

template<typename MatType>
struct GetDenseMatType
{
  typedef arma::Mat<typename MatType::elem_type> type;
};

template<typename MatType>
struct GetUDenseMatType
{
  typedef arma::Mat<arma::uword> type;
};

template<typename eT>
struct GetDenseMatType<arma::SpMat<eT>>
{
  typedef arma::Mat<eT> type;
};

// Get the cube type corresponding to a given MatType.

template<typename MatType>
struct GetCubeType;

template<typename eT>
struct GetCubeType<arma::Mat<eT>>
{
  typedef arma::Cube<eT> type;
};

// Get the sparse matrix type corresponding to a given MatType.

template<typename MatType>
struct GetSparseMatType
{
  typedef arma::SpMat<typename MatType::elem_type> type;
};

template<typename eT>
struct GetSparseMatType<arma::SpMat<eT>>
{
  typedef arma::SpMat<eT> type;
};

// Get whether or not the given type is a base matrix type (e.g. not an
// expression).

template<typename MatType>
struct IsBaseMatType
{
  constexpr static bool value = false;
};

template<typename eT>
struct IsBaseMatType<arma::Mat<eT>>
{
  constexpr static bool value = true;
};

template<typename eT>
struct IsBaseMatType<arma::Col<eT>>
{
  constexpr static bool value = true;
};

template<typename eT>
struct IsBaseMatType<arma::Row<eT>>
{
  constexpr static bool value = true;
};

template<typename eT>
struct IsBaseMatType<arma::SpMat<eT>>
{
  constexpr static bool value = true;
};

template<typename eT>
struct IsBaseMatType<arma::SpCol<eT>>
{
  constexpr static bool value = true;
};

template<typename eT>
struct IsBaseMatType<arma::SpRow<eT>>
{
  constexpr static bool value = true;
};

#endif
