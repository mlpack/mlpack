/**
 * @file Col_extra_meat.hpp
 * @author Ryan Curtin
 *
 * Add an extra explicit constructor for sparse vectors, but only if it doesn't
 * already exist (Armadillo 3.4 specific).
 */
#if ARMA_VERSION_MAJOR == 3 && ARMA_VERSION_MINOR == 4
  template<typename eT>
  inline
  Col<eT>::Col(const SpCol<eT>& X)
    : Mat<eT>(arma_vec_indicator(), X.n_elem, 1, 1)
    {
    arma_extra_debug_sigprint_this(this);

    arrayops::inplace_set(Mat<eT>::memptr(), eT(0), X.n_elem);

    for(typename SpCol<eT>::const_iterator it = X.begin(); it != X.end(); ++it)
      at(it.row()) = (*it);
    }
#endif
