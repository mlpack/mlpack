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


//! \addtogroup fn_accu
//! @{

//! smarter implementation of accu() for sparse things
template<typename eT>
arma_hot
inline
eT
accu(const SpMat<eT>& X)
  {
  arma_extra_debug_sigprint();

  // iterate through nonzero values
  eT sum = 0;

  for (typename std::vector<eT>::iterator it = X.values.begin(); it != X.values.end(); ++it)
    {
    sum += *it;
    }

  return sum;
  }

//! @}
