// Copyright (C) 2008-2011 NICTA (www.nicta.com.au)
// Copyright (C) 2008-2011 Conrad Sanderson
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


//! \addtogroup SpRow
//! @{



template<typename eT>
inline
SpRow<eT>::SpRow()
  : SpMat<eT>(1, 0)
  {
  arma_extra_debug_sigprint();
  }



template<typename eT>
inline
SpRow<eT>::SpRow(const uword in_n_elem)
  : SpMat<eT>(1, in_n_elem)
  {
  arma_extra_debug_sigprint();
  }



template<typename eT>
inline
SpRow<eT>::SpRow(const uword in_n_rows, const uword in_n_cols)
  {
  arma_extra_debug_sigprint();
  
  SpMat<eT>::init(in_n_rows, in_n_cols);
  }



template<typename eT>
inline
SpRow<eT>::SpRow(const char* text)
  {
  arma_extra_debug_sigprint();
  
  SpMat<eT>::operator=(text);
  }
  


template<typename eT>
inline
const SpRow<eT>&
SpRow<eT>::operator=(const char* text)
  {
  arma_extra_debug_sigprint();
  
  SpMat<eT>::operator=(text);
  
  return *this;
  }



template<typename eT>
inline
SpRow<eT>::SpRow(const std::string& text)
  {
  arma_extra_debug_sigprint();
  
  SpMat<eT>::operator=(text);
  }



template<typename eT>
inline
const SpRow<eT>&
SpRow<eT>::operator=(const std::string& text)
  {
  arma_extra_debug_sigprint();
  
  SpMat<eT>::operator=(text);
  
  return *this;
  }



template<typename eT>
inline
const SpRow<eT>&
SpRow<eT>::operator=(const eT val)
  {
  arma_extra_debug_sigprint();
  
  SpMat<eT>::operator=(val);
  
  return *this;
  }



template<typename eT>
template<typename T1>
inline
SpRow<eT>::SpRow(const Base<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  SpMat<eT>::operator=(X.get_ref());
  }



template<typename eT>
template<typename T1>
inline
const SpRow<eT>&
SpRow<eT>::operator=(const Base<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  SpMat<eT>::operator=(X.get_ref());
  
  return *this;
  }



template<typename eT>
template<typename T1, typename T2>
inline
SpRow<eT>::SpRow
  (
  const Base<typename SpRow<eT>::pod_type, T1>& A,
  const Base<typename SpRow<eT>::pod_type, T2>& B
  )
  {
  arma_extra_debug_sigprint();
  
  SpMat<eT>::init(A,B);
  }



template<typename eT>
template<typename T1>
inline
SpRow<eT>::SpRow(const BaseCube<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  SpMat<eT>::operator=(X);
  }



template<typename eT>
template<typename T1>
inline
const SpRow<eT>&
SpRow<eT>::operator=(const BaseCube<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  SpMat<eT>::operator=(X);
  
  return *this;
  }



template<typename eT>
inline
SpRow<eT>::SpRow(const subview_cube<eT>& X)
  {
  arma_extra_debug_sigprint();
  
  SpMat<eT>::operator=(X);
  }



template<typename eT>
inline
const SpRow<eT>&
SpRow<eT>::operator=(const subview_cube<eT>& X)
  {
  arma_extra_debug_sigprint();
  
  SpMat<eT>::operator=(X);
  
  return *this;
  }



template<typename eT>
arma_inline
eT&
SpRow<eT>::col(const uword col_num)
  {
  arma_debug_check( (col_num >= SpMat<eT>::n_cols), "SpRow::col(): out of bounds" );
  
  return SpMat<eT>::at(0, col_num);
  }



template<typename eT>
arma_inline
eT
SpRow<eT>::col(const uword col_num) const
  {
  arma_debug_check( (col_num >= SpMat<eT>::n_cols), "SpRow::col(): out of bounds" );
  
  return SpMat<eT>::at(0, col_num);
  }


/*
template<typename eT>
arma_inline
subview_row<eT>
SpRow<eT>::cols(const uword in_col1, const uword in_col2)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( ( (in_col1 > in_col2) || (in_col2 >= Mat<eT>::n_cols) ), "SpRow::cols(): indices out of bounds or incorrectly used");
  
  const uword subview_n_cols = in_col2 - in_col1 + 1;
  
  return subview_row<eT>(*this, 0, in_col1, subview_n_cols);
  }



template<typename eT>
arma_inline
const subview_row<eT>
SpRow<eT>::cols(const uword in_col1, const uword in_col2) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( ( (in_col1 > in_col2) || (in_col2 >= Mat<eT>::n_cols) ), "SpRow::cols(): indices out of bounds or incorrectly used");
  
  const uword subview_n_cols = in_col2 - in_col1 + 1;
  
  return subview_row<eT>(*this, 0, in_col1, subview_n_cols);
  }



template<typename eT>
arma_inline
subview_row<eT>
SpRow<eT>::subvec(const uword in_col1, const uword in_col2)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( ( (in_col1 > in_col2) || (in_col2 >= Mat<eT>::n_cols) ), "SpRow::subvec(): indices out of bounds or incorrectly used");
  
  const uword subview_n_cols = in_col2 - in_col1 + 1;
  
  return subview_row<eT>(*this, 0, in_col1, subview_n_cols);
  }



template<typename eT>
arma_inline
const subview_row<eT>
SpRow<eT>::subvec(const uword in_col1, const uword in_col2) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( ( (in_col1 > in_col2) || (in_col2 >= Mat<eT>::n_cols) ), "SpRow::subvec(): indices out of bounds or incorrectly used");
  
  const uword subview_n_cols = in_col2 - in_col1 + 1;
  
  return subview_row<eT>(*this, 0, in_col1, subview_n_cols);
  }



template<typename eT>
arma_inline
subview_row<eT>
SpRow<eT>::subvec(const span& col_span)
  {
  arma_extra_debug_sigprint();
  
  const bool col_all = col_span.whole;

  const uword local_n_cols = Mat<eT>::n_cols;
  
  const uword in_col1       = col_all ? 0            : col_span.a;
  const uword in_col2       =                          col_span.b;
  const uword subvec_n_cols = col_all ? local_n_cols : in_col2 - in_col1 + 1;

  arma_debug_check( ( col_all ? false : ((in_col1 > in_col2) || (in_col2 >= local_n_cols)) ), "SpRow::subvec(): indices out of bounds or incorrectly used");
  
  return subview_row<eT>(*this, 0, in_col1, subvec_n_cols);
  }



template<typename eT>
arma_inline
const subview_row<eT>
SpRow<eT>::subvec(const span& col_span) const
  {
  arma_extra_debug_sigprint();
  
  const bool col_all = col_span.whole;

  const uword local_n_cols = Mat<eT>::n_cols;
  
  const uword in_col1       = col_all ? 0            : col_span.a;
  const uword in_col2       =                          col_span.b;
  const uword subvec_n_cols = col_all ? local_n_cols : in_col2 - in_col1 + 1;

  arma_debug_check( ( col_all ? false : ((in_col1 > in_col2) || (in_col2 >= local_n_cols)) ), "SpRow::subvec(): indices out of bounds or incorrectly used");
  
  return subview_row<eT>(*this, 0, in_col1, subvec_n_cols);
  }
*/


// template<typename eT>
// arma_inline
// subview_row<eT>
// SpRow<eT>::operator()(const span& col_span)
//   {
//   arma_extra_debug_sigprint();
//   
//   return subvec(col_span);
//   }
// 
// 
// 
// template<typename eT>
// arma_inline
// const subview_row<eT>
// SpRow<eT>::operator()(const span& col_span) const
//   {
//   arma_extra_debug_sigprint();
//   
//   return subvec(col_span);
//   }



//! remove specified columns
template<typename eT>
inline
void
SpRow<eT>::shed_col(const uword col_num)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( col_num >= SpMat<eT>::n_cols, "SpRow::shed_col(): out of bounds");
  
  shed_cols(col_num, col_num);
  }



//! remove specified columns
template<typename eT>
inline
void
SpRow<eT>::shed_cols(const uword in_col1, const uword in_col2)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check
    (
    (in_col1 > in_col2) || (in_col2 >= SpMat<eT>::n_cols),
    "SpRow::shed_cols(): indices out of bounds or incorrectly used"
    );
  
  const uword diff = (in_col2 - in_col1 + 1);

  // We only need to be concerned with columns that could be affected.  Row indices won't be affected.
  for(uword col = in_col1; col < SpMat<eT>::n_cols; col++)
    {
    // If there are more than zero elements in this column
    if(SpMat<eT>::col_ptrs[col + 1] - SpMat<eT>::col_ptrs[col] > 0)
      {
      // then we have to remove that value.
      SpMat<eT>::values.erase(SpMat<eT>::values.begin() + SpMat<eT>::col_ptrs[col]);
      SpMat<eT>::row_indices.erase(SpMat<eT>::row_indices.begin() + SpMat<eT>::col_ptrs[col]);

      // Now, update the rest of the column pointers...
      for(uword col_b = col + 1; col <= SpMat<eT>::n_cols; col++)
        {
        SpMat<eT>::col_ptrs[col_b]--;
        }
      }
    }

  access::rw(SpMat<eT>::n_cols) -= diff;
  access::rw(SpMat<eT>::n_elem) -= diff;
  access::rw(SpMat<eT>::n_nonzero) = SpMat<eT>::col_ptrs[SpMat<eT>::n_cols];
  }



//! insert N cols at the specified col position,
//! optionally setting the elements of the inserted cols to zero
template<typename eT>
inline
void
SpRow<eT>::insert_cols(const uword col_num, const uword N, const bool set_to_zero)
  {
  arma_extra_debug_sigprint();

  // insertion at col_num == n_cols is in effect an append operation
  arma_debug_check( (col_num > SpMat<eT>::n_cols), "SpRow::insert_cols(): out of bounds");

  arma_debug_check( (set_to_zero == false), "SpRow::insert_cols(): cannot set elements to nonzero values");

  uword newVal = (col_num == 0) ? 0 : SpMat<eT>::col_ptrs[col_num];
  SpMat<eT>::col_ptrs.insert(col_num, N, newVal);

  access::rw(SpMat<eT>::n_cols) += N;
  access::rw(SpMat<eT>::n_elem) += N;
  }



//! insert the given object at the specified col position; 
//! the given object must have one row
template<typename eT>
template<typename T1>
inline
void
SpRow<eT>::insert_cols(const uword col_num, const Base<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  SpMat<eT>::insert_cols(col_num, X);
  }



template<typename eT>
inline
typename SpRow<eT>::row_iterator
SpRow<eT>::begin_row(const uword row_num)
  {
  arma_extra_debug_sigprint();
  
  return SpMat<eT>::begin();
  }



template<typename eT>
inline
typename SpRow<eT>::const_row_iterator
SpRow<eT>::begin_row(const uword row_num) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (row_num >= SpMat<eT>::n_rows), "begin_row(): index out of bounds");
  
  return SpMat<eT>::begin();
  }



template<typename eT>
inline
typename SpRow<eT>::row_iterator
SpRow<eT>::end_row(const uword row_num)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (row_num >= SpMat<eT>::n_rows), "end_row(): index out of bounds");
  
  return SpMat<eT>::end();
  }



template<typename eT>
inline
typename SpRow<eT>::const_row_iterator
SpRow<eT>::end_row(const uword row_num) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (row_num >= SpMat<eT>::n_rows), "end_row(): index out of bounds");
  
  return SpMat<eT>::end();
  }



  
#ifdef ARMA_EXTRA_SPROW_MEAT
  #include ARMA_INCFILE_WRAP(ARMA_EXTRA_SPROW_MEAT)
#endif



//! @}
