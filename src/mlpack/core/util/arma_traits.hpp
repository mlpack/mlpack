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
  static const bool value = false;
};

template<typename MatType>
struct IsMatrix
{
  static const bool value = false;
};

template<typename CubeType>
struct IsCube
{
  static const bool value = false;
};

template<typename FieldType>
struct IsField
{
  static const bool value = false;
};

template<typename T>
struct IsAnyArmaBaseType
{
  static const bool value = IsVector<T>::value || IsMatrix<T>::value ||
      IsCube<T>::value || IsField<T>::value;
};

template<typename eT>
struct IsVector<arma::Col<eT> >
{
  static const bool value = true;
};

template<typename eT>
struct IsVector<arma::SpCol<eT> >
{
  static const bool value = true;
};

template<typename eT>
struct IsVector<arma::Row<eT> >
{
  static const bool value = true;
};

template<typename eT>
struct IsVector<arma::SpRow<eT> >
{
  static const bool value = true;
};

template<typename eT>
struct IsVector<arma::subview_col<eT> >
{
  static const bool value = true;
};

template<typename eT>
struct IsVector<arma::subview_row<eT> >
{
  static const bool value = true;
};

template<typename eT>
struct IsVector<arma::SpSubview_col<eT> >
{
  static const bool value = true;
};

template<typename eT>
struct IsVector<arma::SpSubview_row<eT> >
{
  static const bool value = true;
};

template<typename eT>
struct IsMatrix<arma::Mat<eT> >
{
  static const bool value = true;
};

template<typename eT>
struct IsMatrix<arma::SpMat<eT> >
{
  static const bool value = true;
};

template<typename eT>
struct IsCube<arma::Cube<eT> >
{
  static const bool value = true;
};

template<typename eT>
struct IsField<arma::field<eT> >
{
  static const bool value = true;
};

// Get the row vector type corresponding to a given MatType.

template<typename MatType>
struct GetRowType
{
  using type = arma::Row<typename MatType::elem_type>;
};

template<typename eT>
struct GetRowType<arma::Mat<eT>>
{
  using type = arma::Row<eT>;
};

template<typename eT>
struct GetRowType<arma::SpMat<eT>>
{
  using type = arma::SpRow<eT>;
};

// Get the column vector type corresponding to a given MatType.

template<typename MatType>
struct GetColType
{
  using type = arma::Col<typename MatType::elem_type>;
};

template<typename MatType>
struct GetUColType
{
  using type = arma::Col<arma::uword>;
};

template<typename eT>
struct GetColType<arma::Mat<eT>>
{
  using type = arma::Col<eT>;
};

template<typename eT>
struct GetColType<arma::SpMat<eT>>
{
  using type = arma::SpCol<eT>;
};

// Get the dense row vector type corresponding to a given MatType.

template<typename MatType>
struct GetDenseRowType
{
  using type = typename GetRowType<MatType>::type;
};

template<typename eT>
struct GetDenseRowType<arma::SpMat<eT>>
{
  using type = arma::Row<eT>;
};

// Get the dense column vector type corresponding to a given MatType.

template<typename MatType>
struct GetDenseColType
{
  using type = typename GetColType<MatType>::type;
};

template<typename eT>
struct GetDenseColType<arma::SpMat<eT>>
{
  using type = arma::Col<eT>;
};

// Get the dense matrix type corresponding to a given MatType.

template<typename MatType>
struct GetDenseMatType
{
  using type = arma::Mat<typename MatType::elem_type>;
};

template<typename MatType>
struct GetUDenseMatType
{
  using type = arma::Mat<arma::uword>;
};

template<typename eT>
struct GetDenseMatType<arma::SpMat<eT>>
{
  using type = arma::Mat<eT>;
};

// Get the cube type corresponding to a given MatType.

template<typename MatType>
struct GetCubeType;

template<typename eT>
struct GetCubeType<arma::Mat<eT>>
{
  using type = arma::Cube<eT>;
};

// Get the sparse matrix type corresponding to a given MatType.

template<typename MatType>
struct GetSparseMatType
{
  using type = arma::SpMat<typename MatType::elem_type>;
};

template<typename eT>
struct GetSparseMatType<arma::SpMat<eT>>
{
  using type = arma::SpMat<eT>;
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

template<typename T>
struct IsArma
{
  constexpr static bool value = arma::is_arma_type<T>::value;
};

#if defined(MLPACK_HAS_COOT)

template<typename T>
struct IsCoot
{
  constexpr static bool value = coot::is_coot_type<T>::value;
};

#else

template<typename T>
struct IsCoot
{
  constexpr static bool value = false;
};

#endif

#endif
