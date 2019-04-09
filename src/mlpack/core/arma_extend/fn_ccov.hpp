//! \addtogroup fn_ccov
//! @{



template<typename eT, typename T1>
inline
Mat<eT>
ccov(const Base<eT,T1>& A_expr, const uword norm_type = 0)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (norm_type > 1), "ccov(): norm_type must be 0 or 1");
  
  const Mat<eT>& A = A_expr.get_ref();
  
  Mat<eT> out;
  
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
  
  return out;
  }



template<typename T, typename T1>
inline
Mat< std::complex<T> >
ccov(const Base<std::complex<T>,T1>& A_expr, const uword norm_type = 0)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (norm_type > 1), "ccov(): norm_type must be 0 or 1");
  
  typedef typename std::complex<T> eT;
  
  const Mat<eT>& A = A_expr.get_ref();
  
  Mat<eT> out;
  
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
  
  return out;
  }



template<typename eT, typename T1, typename T2>
inline
Mat<eT>
ccov(const Base<eT,T1>& A_expr, const Base<eT,T2>& B_expr, const uword norm_type = 0)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (norm_type > 1), "ccov(): norm_type must be 0 or 1");
  
  const Mat<eT>& A = A_expr.get_ref();
  const Mat<eT>& B = B_expr.get_ref();
  
  Mat<eT> out;

  if(A.is_vec() && B.is_vec())
    {
    arma_debug_check( (A.n_elem != B.n_elem), "ccov(): the number of elements in A and B must match" );

    const eT* A_ptr = A.memptr();
    const eT* B_ptr = B.memptr();

    eT A_acc   = eT(0);
    eT B_acc   = eT(0);
    eT out_acc = eT(0);

    const uword N = A.n_elem;

    for(uword i=0; i<N; ++i)
      {
      const eT A_tmp = A_ptr[i];
      const eT B_tmp = B_ptr[i];

      A_acc += A_tmp;
      B_acc += B_tmp;

      out_acc += A_tmp * B_tmp;
      }

    out_acc -= (A_acc * B_acc)/eT(N);

    const eT norm_val = (norm_type == 0) ? ( (N > 1) ? eT(N-1) : eT(1) ) : eT(N);

    out.set_size(1,1);
    out[0] = out_acc/norm_val;
    }
  else
    {
    arma_debug_assert_same_size(A, B, "ccov()");

    const uword N = A.n_cols;
    const eT norm_val = (norm_type == 0) ? ( (N > 1) ? eT(N-1) : eT(1) ) : eT(N);

    out = A * trans(B);
    out -= (sum(A) * trans(sum(B))) / eT(N);
    out /= norm_val;
    }
  
  return out;
  }



template<typename T, typename T1, typename T2>
inline
Mat< std::complex<T> >
ccov(const Base<std::complex<T>,T1>& A_expr, const Base<std::complex<T>,T2>& B_expr, const uword norm_type = 0)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (norm_type > 1), "ccov(): norm_type must be 0 or 1");
  
  typedef typename std::complex<T> eT;
  
  const Mat<eT>& A = A_expr.get_ref();
  const Mat<eT>& B = B_expr.get_ref();
  
  Mat<eT> out;
  
  if(A.is_vec() && B.is_vec())
    {
    arma_debug_check( (A.n_elem != B.n_elem), "ccov(): the number of elements in A and B must match" );
    
    const eT* A_ptr = A.memptr();
    const eT* B_ptr = B.memptr();
    
    eT A_acc   = eT(0);
    eT B_acc   = eT(0);
    eT out_acc = eT(0);
    
    const uword N = A.n_elem;
    
    for(uword i=0; i<N; ++i)
      {
      const eT A_tmp = A_ptr[i];
      const eT B_tmp = B_ptr[i];
      
      A_acc += A_tmp;
      B_acc += B_tmp;
      
      out_acc += std::conj(A_tmp) * B_tmp;
      }
    
    out_acc -= (std::conj(A_acc) * B_acc)/eT(N);
    
    const eT norm_val = (norm_type == 0) ? ( (N > 1) ? eT(N-1) : eT(1) ) : eT(N);
    
    out.set_size(1,1);
    out[0] = out_acc/norm_val;
    }
  else
    {
    arma_debug_assert_same_size(A, B, "ccov()");
    
    const uword N = A.n_cols;
    const eT norm_val = (norm_type == 0) ? ( (N > 1) ? eT(N-1) : eT(1) ) : eT(N);
    
    out = A * trans(conj(B));
    out -= (sum(A) * trans(conj(sum(B)))) / eT(N);
    out /= norm_val;
    }
  
  return out;
  }



//! @}
