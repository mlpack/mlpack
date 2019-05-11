//! \addtogroup fn_ccov
//! @{



template<typename T1>
inline
const Op<T1, op_ccov>
ccov(const Base<typename T1::elem_type,T1>& X, const uword norm_type = 0)
  {
  arma_extra_debug_sigprint();

  arma_debug_check( (norm_type > 1), "ccov(): norm_type must be 0 or 1");

  return Op<T1, op_ccov>(X.get_ref(), norm_type, 0);
  }



template<typename T1, typename T2>
inline
const Glue<T1,T2,glue_ccov>
cov(const Base<typename T1::elem_type, T1>& A, const Base<typename T1::elem_type,T2>& B, const uword norm_type = 0)
  {
  arma_extra_debug_sigprint();

  arma_debug_check( (norm_type > 1), "ccov(): norm_type must be 0 or 1");

  return Glue<T1, T2, glue_ccov>(A.get_ref(), B.get_ref(), norm_type);
  }



//! @}
