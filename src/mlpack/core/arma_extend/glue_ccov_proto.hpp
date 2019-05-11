//! \addtogroup glue_ccov
//! @{

class glue_ccov
  {
  public:

  template<typename eT> inline static void direct_ccov(Mat<eT>&                out, const Mat<eT>&                A, const Mat<eT>&                B, const uword norm_type);
  template<typename T>  inline static void direct_ccov(Mat< std::complex<T> >& out, const Mat< std::complex<T> >& A, const Mat< std::complex<T> >& B, const uword norm_type);

  template<typename T1, typename T2> inline static void apply(Mat<typename T1::elem_type>& out, const Glue<T1, T2, glue_ccov>& X);
  };

//! @}

