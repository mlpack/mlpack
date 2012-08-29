// Copyright (C) 2010 NICTA and the authors listed below
// http://nicta.com.au
//
// Authors:
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


//! \addtogroup fn_inplace_reshape
//! @{



template<typename eT>
inline
Mat<eT>&
inplace_reshape(Mat<eT>& X,
                const uword new_n_rows,
                const uword new_n_cols)
  {
  arma_extra_debug_sigprint();

  arma_debug_check((new_n_rows * new_n_cols) != X.n_elem,
      "inplace_reshape(): cannot add or remove elements");

  access::rw(X.n_rows) = new_n_rows;
  access::rw(X.n_cols) = new_n_cols;

  return X;
  }



//! @}
