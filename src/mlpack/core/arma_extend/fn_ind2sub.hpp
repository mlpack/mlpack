// Copyright (C) 2008-2016 National ICT Australia (NICTA)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
// -------------------------------------------------------------------
//
// Written by Conrad Sanderson - http://conradsanderson.id.au
  
  #if (ARMA_VERSION_MAJOR < 6 || \
    (ARMA_VERSION_MAJOR == 6 && ARMA_VERSION_MINOR < 399))
  inline
  uvec
  ind2sub(const SizeMat& s, const uword i)
    {
    arma_extra_debug_sigprint();
    
    arma_debug_check( (i >= (s.n_rows * s.n_cols) ), "ind2sub(): index out of range" );
    
    uvec out(2);
    
    out[0] = i % s.n_rows;
    out[1] = i / s.n_rows;
    
    return out;
    }


  inline
  uvec
  ind2sub(const SizeCube& s, const uword i)
    {
    arma_extra_debug_sigprint();
    
    arma_debug_check( (i >= (s.n_rows * s.n_cols * s.n_slices) ), "ind2sub(): index out of range" );
    
    const uword n_elem_slice = s.n_rows * s.n_cols;
    
    const uword slice  = i / n_elem_slice;
    const uword j      = i - (slice * n_elem_slice);
    const uword row    = j % s.n_rows;
    const uword col    = j / s.n_rows;
    
    uvec out(3);
    
    out[0] = row;
    out[1] = col;
    out[2] = slice;
    
    return out;
    }


  arma_inline
  uword
  sub2ind(const SizeMat& s, const uword row, const uword col)
    {
    arma_extra_debug_sigprint();
    
    arma_debug_check( ((row >= s.n_rows) || (col >= s.n_cols)), "sub2ind(): subscript out of range" );
    
    return uword(row + col*s.n_rows);
    }


  arma_inline
  uword
  sub2ind(const SizeCube& s, const uword row, const uword col, const uword slice)
    {
    arma_extra_debug_sigprint();
    
    arma_debug_check( ((row >= s.n_rows) || (col >= s.n_cols) || (slice >= s.n_slices)), "sub2ind(): subscript out of range" );
    
    return uword( (slice * s.n_rows * s.n_cols) + (col * s.n_rows) + row );
    }
#endif
 

