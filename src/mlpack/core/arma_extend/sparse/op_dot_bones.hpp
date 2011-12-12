// Copyright (C) 2008-2011 NICTA (www.nicta.com.au)
// Copyright (C) 2008-2011 Conrad Sanderson
// 
// This file is part of the Armadillo C++ library.
// It is provided without any warranty of fitness
// for any purpose. You can redistribute this file
// and/or modify it under the terms of the GNU
// Lesser General Public License (LGPL) as published
// by the Free Software Foundation, either version 3
// of the License or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)


//! \addtogroup op_dot
//! @{

//! \brief
//! dot product operation 

class op_dot
  {
  public:
  
  template<typename eT>
  arma_hot arma_pure inline static eT direct_dot_arma(const uword n_elem, const eT* const A, const eT* const B);
  
  template<typename eT>
  arma_hot arma_pure inline static typename arma_float_only<eT>::result
  direct_dot(const uword n_elem, const eT* const A, const eT* const B);
  
  template<typename eT>
  arma_hot arma_pure inline static typename arma_cx_only<eT>::result
  direct_dot(const uword n_elem, const eT* const A, const eT* const B);
  
  template<typename eT>
  arma_hot arma_pure inline static typename arma_integral_only<eT>::result
  direct_dot(const uword n_elem, const eT* const A, const eT* const B);
  
  
  template<typename eT>
  arma_hot arma_pure inline static eT direct_dot(const uword n_elem, const eT* const A, const eT* const B, const eT* C);
  
  template<typename T1, typename T2>
  arma_hot arma_inline static typename T1::elem_type apply(const Base<typename T1::elem_type,T1>& X, const Base<typename T1::elem_type,T2>& Y);
  
  template<typename T1, typename T2>
  arma_hot inline static typename T1::elem_type apply_unwrap(const Base<typename T1::elem_type,T1>& X, const Base<typename T1::elem_type,T2>& Y);
  
  template<typename T1, typename T2>
  arma_hot inline static typename T1::elem_type apply_proxy (const Base<typename T1::elem_type,T1>& X, const Base<typename T1::elem_type,T2>& Y);
  };



//! \brief
//! normalised dot product operation 

class op_norm_dot
  {
  public:
  
  template<typename T1, typename T2>
  arma_hot inline static typename T1::elem_type apply       (const Base<typename T1::elem_type,T1>& X, const Base<typename T1::elem_type,T2>& Y);
  
  template<typename T1, typename T2>
  arma_hot inline static typename T1::elem_type apply_unwrap(const Base<typename T1::elem_type,T1>& X, const Base<typename T1::elem_type,T2>& Y);
  };



class op_cdot
  {
  public:
  
  template<typename T1, typename T2>
  arma_hot arma_inline static typename T1::elem_type apply(const Base<typename T1::elem_type,T1>& X, const Base<typename T1::elem_type,T2>& Y);
  };



//! @}
