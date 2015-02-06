// Backport unary minus operator for sparse matrices to Armadillo 4.000 and
// older.

#if (ARMA_VERSION_MAJOR < 4) || \
    (ARMA_VERSION_MAJOR == 4 && ARMA_VERSION_MINOR <= 0)

template<typename T1>
inline
typename
enable_if2
  <
  is_arma_sparse_type<T1>::value && is_signed<typename T1::elem_type>::value,
  SpOp<T1,spop_scalar_times>
  >::result
operator-
(const T1& X)
  {
  arma_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  return SpOp<T1,spop_scalar_times>(X, eT(-1));
  }

#endif
