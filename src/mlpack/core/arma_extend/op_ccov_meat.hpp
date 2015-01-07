//! \addtogroup op_cov
//! @{



template<typename eT>
inline
void
op_ccov::direct_ccov(Mat<eT>& out, const Mat<eT>& A, const uword norm_type)
  {
  arma_extra_debug_sigprint();

  if(A.is_vec())
    {
    if(A.n_rows == 1)
      {
      out = var(trans(A), norm_type);
      }
    else
      {
      out = var(A, norm_type);
      }
    }
  else
    {
    const uword N = A.n_cols;
    const eT norm_val = (norm_type == 0) ? ( (N > 1) ? eT(N-1) : eT(1) ) : eT(N);

    const Col<eT> acc = sum(A, 1);

    out = A * trans(A);
    out -= (acc * trans(acc)) / eT(N);
    out /= norm_val;
    }
  }



template<typename T>
inline
void
op_ccov::direct_ccov(Mat< std::complex<T> >& out, const Mat< std::complex<T> >& A, const uword norm_type)
  {
  arma_extra_debug_sigprint();

  typedef typename std::complex<T> eT;

  if(A.is_vec())
    {
    if(A.n_rows == 1)
      {
      const Mat<T> tmp_mat = var(trans(A), norm_type);
      out.set_size(1,1);
      out[0] = tmp_mat[0];
      }
    else
      {
      const Mat<T> tmp_mat = var(A, norm_type);
      out.set_size(1,1);
      out[0] = tmp_mat[0];
      }
    }
  else
    {
    const uword N = A.n_cols;
    const eT norm_val = (norm_type == 0) ? ( (N > 1) ? eT(N-1) : eT(1) ) : eT(N);

    const Col<eT> acc = sum(A, 1);

    out = A * trans(conj(A));
    out -= (acc * trans(conj(acc))) / eT(N);
    out /= norm_val;
    }
  }



template<typename T1>
inline
void
op_ccov::apply(Mat<typename T1::elem_type>& out, const Op<T1,op_ccov>& in)
  {
  arma_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  const unwrap_check<T1> tmp(in.m, out);
  const Mat<eT>& A     = tmp.M;

  const uword norm_type = in.aux_uword_a;

  op_ccov::direct_ccov(out, A, norm_type);
  }



//! @}
