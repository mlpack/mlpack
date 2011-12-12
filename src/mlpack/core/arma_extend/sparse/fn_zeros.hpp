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


//! \addtogroup fn_zeros
//! @{

template<typename spmat_type>
arma_inline
const Gen<typename spmat_type::elem_type, gen_zeros>
zeros(const uword n_rows, const uword n_cols, const typename arma_SpMat_SpCol_SpRow_only<spmat_type>::result* junk = 0)
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);

  return Gen<typename spmat_type::elem_type, gen_zeros>(n_rows, n_cols);
  }

//! @}
