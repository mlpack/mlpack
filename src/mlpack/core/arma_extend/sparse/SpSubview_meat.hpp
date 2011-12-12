#include <algorithm>

template<typename eT>
inline
SpSubview<eT>::~SpSubview()
  {
  arma_extra_debug_sigprint();
  }

template<typename eT>
arma_inline 
SpSubview<eT>::SpSubview(const SpMat<eT>& in_m, const uword in_row1, const uword in_col1, const uword in_n_rows, const uword in_n_cols)
  : m(in_m)
  , aux_row1(in_row1)
  , aux_col1(in_col1)
  , n_rows(in_n_rows)
  , n_cols(in_n_cols)
  , n_elem(in_n_rows * in_n_cols)
  , n_nonzero(0)
  {
  arma_extra_debug_sigprint();

  // There must be a O(1) way to do this
  uword end     = m.col_ptrs[in_col1 + in_n_cols];
  uword end_row = in_row1 + in_n_rows;
  uword count   = 0;

  for(uword i = m.col_ptrs[in_col1]; i < end; ++i)
    if(m.row_indices[i] >= in_row1 && m.row_indices[i] < end_row)
      ++count;


  access::rw(n_nonzero) = count;
  }

template<typename eT>
arma_inline 
SpSubview<eT>::SpSubview(SpMat<eT>& in_m, const uword in_row1, const uword in_col1, const uword in_n_rows, const uword in_n_cols)
  : m(in_m)
  , aux_row1(in_row1)
  , aux_col1(in_col1)
  , n_rows(in_n_rows)
  , n_cols(in_n_cols)
  , n_elem(in_n_rows * in_n_cols)
  , n_nonzero(0)
  {
  arma_extra_debug_sigprint();

  // There must be a O(1) way to do this
  uword end     = m.col_ptrs[in_col1 + in_n_cols];
  uword end_row = in_row1 + in_n_rows;
  uword count   = 0;

  for(uword i = m.col_ptrs[in_col1]; i < end; ++i)
    if(m.row_indices[i] >= in_row1 && m.row_indices[i] < end_row)
      ++count;

  access::rw(n_nonzero) = count;
  }

template<typename eT>
inline void 
SpSubview<eT>::operator+= (const eT val)
  {
  arma_extra_debug_sigprint();

  uword size = n_elem - n_nonzero;

  std::vector<eT>& values = m.values;
  std::vector<uword>& row_indices = m.row_indices;
  std::vector<uword>& col_ptrs = m.col_ptrs;

  uword mnz = m.n_nonzero;

  m.print();
  std::cout << values.size() << '\t' << m.values.size() << std::endl;
  values.resize(mnz + size, 0);
  row_indices.resize(mnz + size, 0);
  std::cout << n_elem << '\t' << n_nonzero << '\t' << size << '\t' << values.size() << '\t' << m.values.size() << std::endl;

  uword start_row = aux_row1;
  uword end_row   = aux_row1 + n_rows;
  uword start_col = aux_col1;
  // The this->n_nonzero + 1'th element in m
  uword end_col   = aux_col1 + n_cols; 

  if(size == 0)
    {
    for(uword c = start_col; c < end_col;)
      {
      for(uword r = m.col_ptrs[c]; r < m.col_ptrs[++c]; ++r)
        {
        if(m.row_indices[r] >= start_row && m.row_indices[r] < end_row)
          m.values[r] += val;
        }
      }

    return;
    }

  typename std::vector<eT>::iterator v_beg_iter = values.begin() + col_ptrs[end_col];
  typename std::vector<eT>::iterator v_end_iter = values.end()   - size;

  std::vector<uword>::iterator r_beg_iter = row_indices.begin() + col_ptrs[end_col];
  std::vector<uword>::iterator r_end_iter = row_indices.end()   - size;
  
  // Move the elements in the full matrix after our submatrix to the right by size positions
  std::copy_backward(v_beg_iter, v_end_iter, values.end());
  std::copy_backward(r_beg_iter, r_end_iter, row_indices.end());

  // Adjust the column pointers in the full matrix after our submatrix to account for the shifting
  std::vector<uword>::iterator c_beg_iter = col_ptrs.begin() + end_col;
  std::vector<uword>::iterator c_end_iter = col_ptrs.end();
  for(std::vector<uword>::iterator i = c_beg_iter; i != c_end_iter; ++i)
    {
    *i += size;
    }

  uword col;
  uword dest;
  uword row;

  uword front_iter = col_ptrs[end_col] - 1;
  uword back_iter  = front_iter + size - 1;

  uword offset     = col_ptrs[start_col];

  // Shift the elements of our submatrix to the right positions
  // Adjust the values to the correct new values
  // Add in the new elements
  for(col = 0; col < n_cols; ++col)
    {
    for(row = 0; row < n_rows; ++row)
      {
      dest = offset + row + row * col;
      if(row_indices[front_iter] == row)
        {
        values[back_iter] = values[front_iter] + val;

        --front_iter;
        }
      else
        {
        values[back_iter] = val;
        ++col_ptrs[col+1];
        }

      row_indices[back_iter] = row;
      --back_iter;
      }
    }

  for(uword i = 0; i < values.size(); ++i)
    std::cout << values[i] << '\t';
  std::cout << std::endl;
  access::rw(n_nonzero)   += size - 1;
  access::rw(m.n_nonzero) += size - 1;
  m.print();
  }

template<typename eT>
inline void 
SpSubview<eT>::operator-= (const eT val)
  {
  arma_extra_debug_sigprint();
  }

template<typename eT>
inline void 
SpSubview<eT>::operator*= (const eT val)
  {
  arma_extra_debug_sigprint();

  uword start_row = aux_row1;
  uword end_row   = aux_row1 + n_rows;
  uword start_col = aux_col1;
  uword end_col   = aux_col1 + n_rows;

  for(uword c = start_col; c < end_col;)
    {
    for(uword r = m.col_ptrs[c]; r < m.col_ptrs[++c]; ++r)
      {
      if(m.row_indices[r] >= start_row && m.row_indices[r] < end_row)
        m.values[r] *= val;
      }
    }
  }

template<typename eT>
inline void 
SpSubview<eT>::operator/= (const eT val)
  {
  arma_extra_debug_sigprint();

  uword start_row = aux_row1;
  uword end_row   = aux_row1 + n_rows;
  uword start_col = aux_col1;
  uword end_col   = aux_col1 + n_rows;

  for(uword c = start_col; c < end_col;)
    {
    for(uword r = m.col_ptrs[c]; r < m.col_ptrs[++c]; ++r)
      {
      if(m.row_indices[r] >= start_row && m.row_indices[r] < end_row)
        m.values[r] /= val;
      }
    }
  }

/***
 * Sparse subview col
 */
template<typename eT>
inline
SpSubview_col<eT>::SpSubview_col(const Mat<eT>& in_m, const uword in_col)
  {
  arma_extra_debug_sigprint();
  }

template<typename eT>
inline
SpSubview_col<eT>::SpSubview_col(Mat<eT>& in_m, const uword in_col)
  {
  arma_extra_debug_sigprint();
  }

template<typename eT>
inline
SpSubview_col<eT>::SpSubview_col(const Mat<eT>& in_m, const uword in_col, const uword in_row1, const uword in_n_rows)
  {
  arma_extra_debug_sigprint();
  }

template<typename eT>
inline
SpSubview_col<eT>::SpSubview_col(Mat<eT>& in_m, const uword in_col, const uword in_row1, const uword in_n_rows)
  {
  arma_extra_debug_sigprint();
  }

/***
 * Sparse subview row
 */
template<typename eT>
inline
SpSubview_row<eT>::SpSubview_row(const Mat<eT>& in_m, const uword in_row)
  {
  arma_extra_debug_sigprint();
  }

template<typename eT>
inline
SpSubview_row<eT>::SpSubview_row(Mat<eT>& in_m, const uword in_row)
  {
  arma_extra_debug_sigprint();
  }

template<typename eT>
inline
SpSubview_row<eT>::SpSubview_row(const Mat<eT>& in_m, const uword in_row, const uword in_col1, const uword in_n_cols)
  {
  arma_extra_debug_sigprint();
  }

template<typename eT>
inline
SpSubview_row<eT>::SpSubview_row(Mat<eT>& in_m, const uword in_row, const uword in_col1, const uword in_n_cols)
  {
  arma_extra_debug_sigprint();
  }
