//! \addtogroup op_cov
//! @{



class op_ccov
  {
  public:

  template<typename eT> inline static void direct_ccov(Mat<eT>&                out, const Mat<eT>& X,                const uword norm_type);
  template<typename  T> inline static void direct_ccov(Mat< std::complex<T> >& out, const Mat< std::complex<T> >& X, const uword norm_type);

  template<typename T1> inline static void apply(Mat<typename T1::elem_type>& out, const Op<T1,op_ccov>& in);
  };



//! @}
