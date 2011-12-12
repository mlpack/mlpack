// Copyright (C) 2011 Ryan Curtin <ryan@igglybob.com>
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

//! Specializations for SpMat cases.
template<typename T1, typename eT>
arma_hot
arma_inline
typename T1::elem_type
op_dot::apply(const Base<typename T1::elem_type, T1>& X, const SpMat<eT>& Y)
  {
  arma_extra_debug_sigprint();

  // identical to symmetric case... call that instead (this call should be optimized out)
  op_dot::apply(Y, X);
  }


template<typename eT, typename T2>
arma_hot
arma_inline
typename eT
op_dot::apply(const SpMat<eT>& X, const Base<typename T2::elem_type, T2>& Y)
  {
  arma_extra_debug_sigprint();

  //
  }
