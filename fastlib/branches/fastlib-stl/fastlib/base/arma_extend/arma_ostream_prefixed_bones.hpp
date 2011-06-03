/***
 * @file arma_ostream_prefixed_bones.hpp
 * @author Ryan Curtin
 *
 * Slight change of arma_ostream_bones.hpp so that the arma output functions can
 * work with IO seamlessly.  We do not need to reimplement arma_ostream_state
 * (we can reuse it).  We implement one class, which uses PrefixedOutStream
 * where the Armadillo base library uses std::ostream.  It does not need to
 * implement every function from arma_ostream.
 */
#include <fastlib/fx/io.h>

class arma_ostream_prefixed
  {
  public:

  template<typename eT>
  inline static void print_elem_zero(mlpack::io::PrefixedOutStream& o);

  template<typename eT>
  arma_inline static void print_elem(mlpack::io::PrefixedOutStream& o,
                                     const eT& x);

  template<typename  T>
  inline static void print_elem(mlpack::io::PrefixedOutStream& o,
                                const std::complex<T>& x);

  template<typename eT>
  inline static void print(mlpack::io::PrefixedOutStream& o, const Mat<eT>& m,
                           const bool modify);

  template<typename eT>
  inline static void print(mlpack::io::PrefixedOutStream& o, const Cube<eT>& m,
                           const bool modify);

  template<typename oT>
  inline static void print(mlpack::io::PrefixedOutStream& o,
                           const field<oT>& m);

  template<typename oT>
  inline static void print(mlpack::io::PrefixedOutStream& o,
                           const subview_field<oT>& m);
  };
