// Copyright (C) 2010 NICTA and the authors listed below
// http://nicta.com.au
// 
// Authors:
// - Conrad Sanderson (conradsand at ieee dot org)
// - Dimitrios Bouzas (dimitris dot mpouzas at gmail dot com)
// - Ryan Curtin (ryan at igglybob dot com)
// 
// This file is part of the Armadillo C++ library.
// It is provided without any warranty of fitness
// for any purpose. You can redistribute this file
// and/or modify it under the terms of the GNU
// Lesser General Public License (LGPL) as published
// by the Free Software Foundation, either version 3
// of the License or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)


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
