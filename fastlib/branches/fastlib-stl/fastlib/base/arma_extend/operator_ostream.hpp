/***
 * @file operator_ostream.hpp
 * @author Ryan Curtin
 *
 * Overloads of operator<< functions so that Armadillo ostream output will work
 * with IO output.
 */
#include <fastlib/fx/prefixedoutstream.h>
#include <fastlib/fx/nulloutstream.h>

template<typename eT, typename T1>
inline
mlpack::io::PrefixedOutStream&
operator<< (mlpack::io::PrefixedOutStream& o, const Base<eT, T1>& X)
  {
  arma_extra_debug_sigprint();

  const unwrap<T1> tmp(X.get_ref());

  arma_ostream_prefixed::print(o, tmp.M, true);

  return o;
  }

// Does not need to do anything (null output stream).
template<typename eT, typename T1>
inline
mlpack::io::NullOutStream&
operator<< (mlpack::io::NullOutStream& o, const Base<eT, T1>& X)
  {
  arma_extra_debug_sigprint();

  return o;
  }

template<typename T1>
inline
mlpack::io::PrefixedOutStream&
operator<< (mlpack::io::PrefixedOutStream& o,
            const BaseCube<typename T1::elem_type, T1>& X)
  {
  arma_extra_debug_sigprint();

  const unwrap_cube<T1> tmp(X.get_ref());

  arma_ostream_prefixed::print(o, tmp.M, true);

  return o;
  }

// Does not need to do anything (null output stream).
template<typename T1>
inline
mlpack::io::NullOutStream&
operator<< (mlpack::io::NullOutStream& o,
            const BaseCube<typename T1::elem_type, T1>& X)
  {
  arma_extra_debug_sigprint();

  return o;
  }

template<typename T1>
inline
mlpack::io::PrefixedOutStream&
operator<< (mlpack::io::PrefixedOutStream& o, const field<T1>& X)
  {
  arma_extra_debug_sigprint();

  arma_ostream_prefixed::print(o, X);

  return o;
  }

// Does not need to do anything (null output stream).
template<typename T1>
inline
mlpack::io::NullOutStream&
operator<< (mlpack::io::NullOutStream& o, const field<T1>& X)
  {
  arma_extra_debug_sigprint();

  return o;
  }

template<typename T1>
inline
mlpack::io::PrefixedOutStream&
operator<< (mlpack::io::PrefixedOutStream& o, const subview_field<T1>& X)
  {
  arma_extra_debug_sigprint();

  arma_ostream_prefixed::print(o, X);

  return o;
  }

// Does not need to do anything (null output stream).
template<typename T1>
inline
mlpack::io::NullOutStream&
operator<< (mlpack::io::NullOutStream& o, const subview_field<T1>& X)
  {
  arma_extra_debug_sigprint();

  return o;
  }
