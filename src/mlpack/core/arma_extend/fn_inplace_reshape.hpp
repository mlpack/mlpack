//! \addtogroup fn_inplace_reshape
//! @{



/**
 * This does not handle column vectors or row vectors entirely correctly.  You
 * should be able to do multiplication or other basic operations with the
 * resulting matrix, but it may have other problems.  So if you are using this
 * on vectors (arma::Col<> or arma::Row<>), be careful, and be warned that
 * bizarre behavior may occur.
 */
template<typename eT>
inline
Mat<eT>&
inplace_reshape(Mat<eT>& X,
                const uword new_n_rows,
                const uword new_n_cols)
  {
  arma_extra_debug_sigprint();

  arma_debug_check((new_n_rows * new_n_cols) != X.n_elem,
      "inplace_reshape(): cannot add or remove elements");

  access::rw(X.n_rows) = new_n_rows;
  access::rw(X.n_cols) = new_n_cols;

  return X;
  }



//! @}
