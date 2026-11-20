/**
 * @file find_nan.hpp
 * @author Ryan Curtin
 *
 * When find_nan() is not available (Armadillo < 11.4), provide an internal
 * mlpack implementation that operates the same way.  It is slower.
 */
#ifndef MLPACK_CORE_ARMA_EXTEND_FIND_NAN_HPP
#define MLPACK_CORE_ARMA_EXTEND_FIND_NAN_HPP

namespace mlpack {

#if ARMA_VERSION_MAJOR < 11 || \
    (ARMA_VERSION_MAJOR == 11 && ARMA_VERSION_MINOR < 4)

template<typename T>
arma::uvec find_nan(const T& m,
                    const std::enable_if_t<arma::is_arma_type<T>::value>* = 0)
{
  typedef typename T::elem_type ElemType;

  if (!std::numeric_limits<ElemType>::has_quiet_NaN)
    return arma::uvec(); // There can't be any NaNs.

  // find_nonfinite() exists on older Armadillo, and we can also search for +Inf
  // and -Inf.
  arma::uvec nonfiniteIndices = arma::find_nonfinite(m);
  if (nonfiniteIndices.n_elem == 0)
    return arma::uvec();

  arma::uvec infIndices = arma::find(
      m == std::numeric_limits<ElemType>::infinity());
  arma::uvec neginfIndices = arma::find(
      m == -std::numeric_limits<ElemType>::infinity());

  arma::uvec result(nonfiniteIndices.n_elem -
      (infIndices.n_elem + neginfIndices.n_elem));
  if (result.n_elem == 0)
    return result;

  size_t infIndex = 0;
  size_t neginfIndex = 0;
  size_t outputIndex = 0;
  for (size_t i = 0; i < nonfiniteIndices.n_elem; ++i)
  {
    if (infIndex < infIndices.n_elem &&
        nonfiniteIndices[i] == infIndices[infIndex])
      ++infIndex;
    else if (neginfIndex < neginfIndices.n_elem &&
             nonfiniteIndices[i] == neginfIndices[neginfIndex])
      ++neginfIndex;
    else
      result[outputIndex++] = nonfiniteIndices[i];
  }

  return result;
}

#endif

} // namespace mlpack

#endif
