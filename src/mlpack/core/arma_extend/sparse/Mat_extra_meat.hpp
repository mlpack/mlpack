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


//! \addtogroup Mat
//! @{

template<typename eT>
inline
Mat<eT>::Mat(const SpMat<eT>& m)
  : n_rows(m.n_rows)
  , n_cols(m.n_cols)
  , n_elem(m.n_elem)
  , vec_state(0)
  , mem_state(0)
  , mem()
  {
  arma_extra_debug_sigprint_this(this);

  // Initializes memory.
  init_cold();

  // Set memory to zero.
  fill(eT(0));

  // Iterate over each nonzero element and set it.
  for(typename SpMat<eT>::const_iterator it = m.begin(); it != m.end(); it++)
    {
    at(it.row, it.col) = (*it);
    }
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator=(const SpMat<eT>& m)
  {
  arma_extra_debug_sigprint();

  init_warm(m.n_rows, m.n_cols);

  // Set memory to 0.
  fill(eT(0));

  for(typename SpMat<eT>::const_iterator it = m.begin(); it != m.end(); it++)
    {
    at(it.row, it.col) = (*it);
    }
  }

//! @}
