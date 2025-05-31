/**
 * @file combine_options.hpp
 * @author Ryan Curtin
 * @author Omar Shrit
 *
 * Definition of operator+() for DataOptions types.
 */
#ifndef MLPACK_CORE_DATA_COMBINE_OPTIONS_HPP
#define MLPACK_CORE_DATA_COMBINE_OPTIONS_HPP

#include <mlpack/prereqs.hpp>

#include "data_options.hpp"
#include "matrix_options.hpp"
#include "text_options.hpp"

namespace mlpack {
namespace data {

// This template metaprogram encodes what the result of combining two different
// DataOptions types are.  For now it exhaustively considers every possibility,
// but, in the future, perhaps it will be possible to come up with something
// that scales a little better.

template<typename DataOptionsType1, typename DataOptionsType2>
struct GetCombinedDataOptionsType
{
  typedef void result; // unknown result
};

template<typename T>
struct MostDerivedType
{
  typedef void result;
};

template<typename Derived>
struct MostDerivedType<DataOptionsBase<Derived>>
{
  typedef Derived result;
};

template<>
struct MostDerivedType<DataOptionsBase<PlainDataOptions>>
{
  typedef DataOptions result;
};

template<>
struct MostDerivedType<DataOptionsBase<MatrixOptionsBase<PlainMatrixOptions>>>
{
  typedef MatrixOptions result;
};

template<>
struct MostDerivedType<DataOptionsBase<MatrixOptionsBase<TextOptions>>>
{
  typedef TextOptions result;
};

// When both types are the same, we return that type.
template<typename Derived>
struct GetCombinedDataOptionsType<DataOptionsBase<Derived>,
                                  DataOptionsBase<Derived>>
{
  typedef typename MostDerivedType<DataOptionsBase<Derived>>::result result;
};

// When both types are different, the result is hardcoded.
template<>
struct GetCombinedDataOptionsType<DataOptionsBase<PlainDataOptions>,
    DataOptionsBase<MatrixOptionsBase<PlainMatrixOptions>>>
{
  typedef MatrixOptions result;
};

template<>
struct GetCombinedDataOptionsType<DataOptionsBase<
    MatrixOptionsBase<PlainMatrixOptions>>,
    DataOptionsBase<PlainDataOptions>>
{
  typedef MatrixOptions result;
};

template<>
struct GetCombinedDataOptionsType<DataOptionsBase<
    MatrixOptionsBase<TextOptions>>, DataOptionsBase<PlainDataOptions>>
{
  typedef TextOptions result;
};

template<>
struct GetCombinedDataOptionsType<DataOptionsBase<PlainDataOptions>,
    DataOptionsBase<MatrixOptionsBase<TextOptions>>>
{
  typedef TextOptions result;
};

template<>
struct GetCombinedDataOptionsType<
    DataOptionsBase<MatrixOptionsBase<PlainMatrixOptions>>,
    DataOptionsBase<MatrixOptionsBase<TextOptions>>>
{
  typedef TextOptions result;
};

template<>
struct GetCombinedDataOptionsType<
    DataOptionsBase<MatrixOptionsBase<TextOptions>>,
    DataOptionsBase<MatrixOptionsBase<PlainMatrixOptions>>>
{
  typedef TextOptions result;
};

// Using the template metaprogram above, return a combined DataOptions.

template<typename Derived1, typename Derived2>
typename GetCombinedDataOptionsType<DataOptionsBase<Derived1>,
                                    DataOptionsBase<Derived2>>::result
operator+(const DataOptionsBase<Derived1>& a,
          const DataOptionsBase<Derived2>& b)
{
  using ReturnType = typename GetCombinedDataOptionsType<
      DataOptionsBase<Derived1>, DataOptionsBase<Derived2>>::result;

  if (std::is_same_v<Derived1, Derived2>)
  {
    // This is the easy case!  They are the same.
    ReturnType output(a);
    output += b;
    return output;
  }
  else
  {
    // Here's the hard part, where the types are different.
    // In this case, we can keep only the options from both sides that will be
    // used in the result.  If members of a or b can't be represented in
    // ReturnType, then a warning will be printed.
    ReturnType convertedA(a);
    ReturnType convertedB(b);
    return convertedA + convertedB;
  }
}

} // namespace data
} // namespace mlpack

#endif
