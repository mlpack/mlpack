// Copyright (C) 2011 Ryan Curtin <ryan@igglybob.com>
// Copyright (C) 2011 Matthew Amidon
//
// This file is part of the Armadillo C++ library.
// It is provided without any warranty of fitness
// for any purpose. You can redistribute this file
// and/or modify it under the terms of the GNU
// Lesser General Public License (LGPL) as published
// by the Free Software Foundation, either version 3
// of the License or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

//! \addtogroup SpMat
//! @{

///////////////////////////////////////////////////////////////////////////////
// SpMat::iterator implementation                                            //
///////////////////////////////////////////////////////////////////////////////

template<typename eT>
inline
SpMat<eT>::iterator::iterator(SpMat<eT>& in_M, uword initial_pos)
  : M(in_M)
  , row(0)
  , col(0)
  , pos(initial_pos)
  {

  if (pos >= M.n_nonzero)
    {
    row = 0;
    col = M.n_cols;
    }
  else
    {
    // Figure out the row and column of the position.
    while (M.col_ptrs[col + 1] <= pos)
      {
      col++;
      }

    row = M.row_indices[pos];
    }
  }

template<typename eT>
inline
SpMat<eT>::iterator::iterator(SpMat<eT>& in_M, uword in_row, uword in_col)
  : M(in_M)
  , row(in_row)
  , col(in_col)
  , pos(0)
  {
  // So we have a destination we want to be just after, but don't know what position that is.  Make another iterator to find out...
  const_iterator it(in_M);
  while((it.col < in_col) || ((it.col == in_col) && (it.row < in_row)))
    {
    it++;
    }

  // Now that it is at the right place, take its position.
  row = it.row;
  col = it.col;
  pos = it.pos;
  }

template<typename eT>
inline
SpMat<eT>::iterator::iterator(const typename SpMat<eT>::iterator& other)
  : M(other.M)
  , row(other.row)
  , col(other.col)
  , pos(other.pos)
  {
  // Nothing to do.
  }

template<typename eT>
inline
SpValProxy<eT>
SpMat<eT>::iterator::operator*()
  {
  return M(row, col);
  }

template<typename eT>
inline
typename SpMat<eT>::iterator&
SpMat<eT>::iterator::operator++()
  {
  ++pos;

  if (pos >= M.n_nonzero) // We are at the end.
    {
    row = 0;
    col = M.n_cols;

    return *this;
    }

  // Now we have to ascertain the position of the new
  // element.  First, see if we moved a column.
  while (M.col_ptrs[col + 1] <= pos)
    {
    ++col;
    }

  row = M.row_indices[pos];

  return *this;

  }

template<typename eT>
inline
void
SpMat<eT>::iterator::operator++(int)
  {
  // Same as previous implementation.
  ++pos;

  if (pos >= M.n_nonzero) // We are at the end.
    {
    row = 0;
    col = M.n_cols;

    return;
    }

  while (M.col_ptrs[col + 1] <= pos)
    {
    ++col;
    }

  row = M.row_indices[pos];

  }

template<typename eT>
inline
typename SpMat<eT>::iterator&
SpMat<eT>::iterator::operator--()
  {
  if (pos > 0) // Don't break everything.
    {
    --pos;
    }

  // First, see if we moved back a column.
  while (pos < M.col_ptrs[col])
    {
    --col;
    }

  row = M.row_indices[pos];

  return *this;

  }

template<typename eT>
inline
void
SpMat<eT>::iterator::operator--(int)
  {
  // Same as previous implementation.
  if (pos > 0)
    {
    --pos;
    }

  while (pos < M.col_ptrs[col])
    {
    --col;
    }

  row = M.row_indices[pos];

  }  

template<typename eT>
inline
bool
SpMat<eT>::iterator::operator!=(const typename SpMat<eT>::iterator& rhs) const
  {
  return ((row != rhs.row) || (col != rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::iterator::operator!=(const typename SpMat<eT>::const_iterator& rhs) const
  {
  return ((row != rhs.row) || (col != rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::iterator::operator!=(const typename SpMat<eT>::row_iterator& rhs) const
  {
  return ((row != rhs.row) || (col != rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::iterator::operator!=(const typename SpMat<eT>::const_row_iterator& rhs) const
  {
  return ((row != rhs.row) || (col != rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::iterator::operator==(const typename SpMat<eT>::iterator& rhs) const
  {
  arma_extra_debug_sigprint();

  return ((row == rhs.row) && (col == rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::iterator::operator==(const typename SpMat<eT>::const_iterator& rhs) const
  {
  arma_extra_debug_sigprint();

  return ((row == rhs.row) && (col == rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::iterator::operator==(const typename SpMat<eT>::row_iterator& rhs) const
  {
  return ((row == rhs.row) && (col == rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::iterator::operator==(const typename SpMat<eT>::const_row_iterator& rhs) const
  {
  return ((row == rhs.row) && (col == rhs.col));
  }

///////////////////////////////////////////////////////////////////////////////
// SpMat::const_iterator implementation                                      //
///////////////////////////////////////////////////////////////////////////////

template<typename eT>
inline
SpMat<eT>::const_iterator::const_iterator(const SpMat<eT>& in_M, uword initial_pos)
  : M(in_M)
  , row(0)
  , col(0)
  , pos(initial_pos)
  {

  if (pos >= M.n_nonzero)
    {
    row = 0;
    col = M.n_cols;
    }
  else
    {
    // Figure out the row and column of the position.
    while (M.col_ptrs[col + 1] <= pos)
      {
      col++;
      }

    row = M.row_indices[pos];

    }

  }

template<typename eT>
inline
SpMat<eT>::const_iterator::const_iterator(const SpMat<eT>& in_M, uword in_row, uword in_col)
  : M(in_M)
  , row(in_row)
  , col(in_col)
  , pos(0)
  {
  // So we have a destination we want to be just after, but don't know what position that is.  Make another iterator to find out...
  const_iterator it(in_M);
  while((it.col < in_col) || ((it.col == in_col) && (it.row < in_row)))
    {
    it++;
    }

  // Now that it is at the right place, take its position.
  row = it.row;
  col = it.col;
  pos = it.pos;
  }

template<typename eT>
inline
SpMat<eT>::const_iterator::const_iterator(const typename SpMat<eT>::const_iterator& other)
  : M(other.M)
  , row(other.row)
  , col(other.col)
  , pos(other.pos)
  {
  // Nothing to do.
  }

template<typename eT>
inline
SpMat<eT>::const_iterator::const_iterator(const typename SpMat<eT>::iterator& other)
  : M(other.M)
  , row(other.row)
  , col(other.col)
  , pos(other.pos)
  {
  // Nothing to do.
  }

template<typename eT>
inline
const eT
SpMat<eT>::const_iterator::operator*() const
  {
  return M(row, col);
  }

template<typename eT>
inline
typename SpMat<eT>::const_iterator&
SpMat<eT>::const_iterator::operator++()
  {
  ++pos;

  if (pos >= M.n_nonzero) // We are at the end.
    {
    row = 0;
    col = M.n_cols;

    return *this;
    }

  // Now we have to ascertain the position of the new
  // element.  First, see if we moved a column.
  while (M.col_ptrs[col + 1] <= pos)
    {
    ++col;
    }

  row = M.row_indices[pos];

  return *this;

  }

template<typename eT>
inline
void
SpMat<eT>::const_iterator::operator++(int)
  {
  // Same as previous implementation.
  ++pos;

  if (pos >= M.n_nonzero) // We are at the end.
    {
    row = 0;
    col = M.n_cols;

    return;
    }

  while (M.col_ptrs[col + 1] <= pos)
    {
    ++col;
    }

  row = M.row_indices[pos];

  }

template<typename eT>
inline
typename SpMat<eT>::const_iterator&
SpMat<eT>::const_iterator::operator--()
  {
  if (pos > 0) // Don't break everything.
    {
    --pos;
    }

  // First, see if we moved back a column.
  while (pos < M.col_ptrs[col])
    {
    --col;
    }

  row = M.row_indices[pos];

  return *this;

  }

template<typename eT>
inline
void
SpMat<eT>::const_iterator::operator--(int)
  {
  // Same as previous implementation.
  if (pos > 0)
    {
    --pos;
    }

  while (pos < M.col_ptrs[col])
    {
    --col;
    }

  row = M.row_indices[pos];

  }

template<typename eT>
inline
bool
SpMat<eT>::const_iterator::operator!=(const typename SpMat<eT>::iterator& rhs) const
  {
  return ((row != rhs.row) || (col != rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::const_iterator::operator!=(const typename SpMat<eT>::const_iterator& rhs) const
  {
  return ((row != rhs.row) || (col != rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::const_iterator::operator!=(const typename SpMat<eT>::row_iterator& rhs) const
  {
  return ((row != rhs.row) || (col != rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::const_iterator::operator!=(const typename SpMat<eT>::const_row_iterator& rhs) const
  {
  return ((row != rhs.row) || (col != rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::const_iterator::operator==(const typename SpMat<eT>::iterator& rhs) const
  {
  return ((row == rhs.row) && (col == rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::const_iterator::operator==(const typename SpMat<eT>::const_iterator& rhs) const
  {
  return ((row == rhs.row) && (col == rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::const_iterator::operator==(const typename SpMat<eT>::row_iterator& rhs) const
  {
  return ((row == rhs.row) && (col == rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::const_iterator::operator==(const typename SpMat<eT>::const_row_iterator& rhs) const
  {
  return ((row == rhs.row) && (col == rhs.col));
  }

///////////////////////////////////////////////////////////////////////////////
// SpMat::row_iterator implementation                                        //
///////////////////////////////////////////////////////////////////////////////

/**
 * Initialize the row_iterator.
 */
template<typename eT>
inline
SpMat<eT>::row_iterator::row_iterator(SpMat<eT>& in_M, uword initial_pos)
  : M(in_M)
  , row(0)
  , col(0)
  , pos(initial_pos)
  {
  // We don't count zeroes in our position count, so we have to find the nonzero
  // value corresponding to the given initial position.

  // First check: if the given position is greater than the number of nonzero
  // elements, we are at the end.
  if (pos >= M.n_nonzero)
    {
    row = 0;
    col = M.n_cols;
    }
  else
    {
    // This is irritating because we don't know where the elements are in each
    // row.  What we will do is loop across all columns looking for elements in
    // row 0 (and add to our sum), then in row 1, and so forth, until we get to
    // the desired position.
    uword cur_pos = -1;
    uword cur_row = 0;
    uword cur_col = 0;

    while(true) // This loop is terminated from the inside.
      {
      // Is there anything in the column we are looking at?
      for (uword ind = 0; ((M.col_ptrs[cur_col] + ind < M.col_ptrs[cur_col + 1]) && (M.row_indices[M.col_ptrs[cur_col] + ind] <= cur_row)); ind++)
        {
        // There is something in this column.  Is it in the row we are looking
        // at?
        if (M.row_indices[M.col_ptrs[cur_col] + ind] == cur_row)
          {
          // Yes, it is what we are looking for.  Increment our current
          // position.
          if (++cur_pos == pos)
            {
            row = cur_row;
            col = cur_col;

            return;
            }

          // We are done with this column.  Break to the column incrementing code (directly below).
          break;
          }
        }

      cur_col++; // Done with the column.  Move on.
      if (cur_col == M.n_cols)
        {
        // We are out of columns.  Loop back to the beginning and look on the
        // next row.
        cur_col = 0;
        cur_row++;

        }

      }

    row = cur_row;
    col = cur_col;

    }
  }

template<typename eT>
inline
SpMat<eT>::row_iterator::row_iterator(SpMat<eT>& in_M, uword in_row, uword in_col)
  : M(in_M)
  , row(in_row)
  , col(in_col)
  , pos(0)
  {
  // So we have a destination we want to be just after, but don't know what position that is.  Make another iterator to find out...
  const_row_iterator it(in_M);
  while((it.row < in_row) || ((it.row == in_row) && (it.col < in_col)))
    {
    it++;
    }

  // Now that it is at the right place, take its position.
  row = it.row;
  col = it.col;
  pos = it.pos;
  }

/**
 * Initialize the row_iterator from another row_iterator.
 */
template<typename eT>
inline
SpMat<eT>::row_iterator::row_iterator(const typename SpMat<eT>::row_iterator& other)
  : M(other.M)
  , row(other.row)
  , col(other.col)
  , pos(other.pos)
  {
  // Nothing to do.
  }

/**
 * We need to return an SpValProxy<eT> because the user might change the value to a zero.
 */
template<typename eT>
inline
SpValProxy<eT>
SpMat<eT>::row_iterator::operator*()
  {
  return M(row, col);
  }

/**
 * Increment the row_iterator.
 */
template<typename eT>
inline
typename SpMat<eT>::row_iterator&
SpMat<eT>::row_iterator::operator++()
  {
  // We just need to find the next nonzero element.
  pos++;

  // If we have exceeded the bounds, update accordingly.
  if (pos >= M.n_nonzero)
    {
    row = 0;
    col = M.n_cols;

    return *this;
    }

  // Otherwise, we need to search.
  uword cur_col = col;
  uword cur_row = row;

  while (true) // This loop is terminated from the inside.
    {
    // Increment the current column and see if we are now on a new row.
    if (++cur_col == M.n_cols)
      {
      cur_col = 0;
      cur_row++;
      }

    // Is there anything in this new column?
    for (uword ind = 0; ((M.col_ptrs[cur_col] + ind < M.col_ptrs[cur_col + 1]) && (M.row_indices[M.col_ptrs[cur_col] + ind] <= cur_row)); ind++)
      {
      if (M.row_indices[M.col_ptrs[cur_col] + ind] == cur_row)
        {
        // We have successfully incremented.
        row = cur_row;
        col = cur_col;

        return *this; // Now we are done.
        }
      }
    }
  }

/**
 * Increment the row_iterator (but do not return anything.
 */
template<typename eT>
inline
void
SpMat<eT>::row_iterator::operator++(int)
  {
  // We just need to find the next nonzero element.
  pos++;

  // Make sure we did not exceed the bounds.
  if (pos >= M.n_nonzero)
    {
    row = 0;
    col = M.n_cols;

    return; // Nothing else to do.
    }

  // Now, we need to search.
  uword cur_col = col;
  uword cur_row = row;

  while (true) // This loop is terminated from the inside.
    {
    // Increment the current column and see if we are now on a new row.
    if (++cur_col == M.n_cols)
      {
      cur_col = 0;
      cur_row++;
      }

    // Is there anything in this new column?
    for (uword ind = 0; ((M.col_ptrs[cur_col] + ind < M.col_ptrs[cur_col + 1]) && (M.row_indices[M.col_ptrs[cur_col] + ind] <= cur_row)); ind++)
      {
      if (M.row_indices[M.col_ptrs[cur_col] + ind] == cur_row)
        {
        // We have successfully incremented.
        row = cur_row;
        col = cur_col;

        return; // Now we are done.
        }
      }
    }
  }

/**
 * Decrement the row_iterator.
 */
template<typename eT>
inline
typename SpMat<eT>::row_iterator&
SpMat<eT>::row_iterator::operator--()
  {
  // We just need to find the previous element.
  if (pos == 0)
    {
    // We cannot decrement.
    return;
    }
  else if (pos == M.n_nonzero)
    {
    // We will be coming off the last element.  We need to reset the row correctly, because we set row = 0 in the last matrix position.
    row = M.n_rows - 1;
    }
  else if (pos > M.n_nonzero)
    {
    // We are in nowhere land...
    pos--;
    return *this;
    }

  pos--;

  // We have to search backwards.
  uword cur_col = col;
  uword cur_row = row;

  while (true) // This loop is terminated from the inside.
    {
    // Decrement the current column and see if we are now on a new row.  cur_col is a uword so a negativity check will not work.
    if (--cur_col > M.n_cols /* this means it underflew */)
      {
      cur_col = M.n_cols - 1;
      cur_row--;
      }

    // Is there anything in this new column?
    for (uword ind = 0; ((M.col_ptrs[cur_col] + ind < M.col_ptrs[cur_col + 1]) && (M.row_indices[M.col_ptrs[cur_col] + ind] <= cur_row)); ind++)
      {
      if (M.row_indices[M.col_ptrs[cur_col] + ind] == cur_row)
        {
        // We have successfully decremented.
        row = cur_row;
        col = cur_col;

        return *this; // Now we are done.
        }
      }
    }
  }

/**
 * Decrement the row_iterator.
 */
template<typename eT>
inline
void
SpMat<eT>::row_iterator::operator--(int)
  {
  // We just need to find the previous element.
  if (pos == 0)
    {
    // We cannot decrement.
    return;
    }
  else if (pos == M.n_nonzero)
    {
    // We will be coming off the last element.  We need to reset the row correctly, because we set row = 0 in the last matrix position.
    row = M.n_rows - 1;
    }
  else if (pos > M.n_nonzero)
    {
    // We are in nowhere land...
    pos--;
    return;
    }

  pos--;

  // We have to search backwards.
  uword cur_col = col;
  uword cur_row = row;

  while (true) // This loop is terminated from the inside.
    {
    // Decrement the current column and see if we are now on a new row.  cur_col is a uword so a negativity check will not work.
    if (--cur_col > M.n_cols /* this means it underflew */)
      {
      cur_col = M.n_cols - 1;
      cur_row--;
      }

    // Is there anything in this new column?
    for (uword ind = 0; ((M.col_ptrs[cur_col] + ind < M.col_ptrs[cur_col + 1]) && (M.row_indices[M.col_ptrs[cur_col] + ind] <= cur_row)); ind++)
      {
      if (M.row_indices[M.col_ptrs[cur_col] + ind] == cur_row)
        {
        // We have successfully decremented.
        row = cur_row;
        col = cur_col;

        return; // Now we are done.
        }
      }
    }
  }

/**
 * Return true if this row_iterator does not represent the same position as the given row_iterator.
 */
template<typename eT>
inline
bool
SpMat<eT>::row_iterator::operator!=(const typename SpMat<eT>::iterator& rhs) const
  {
  return ((row != rhs.row) || (col != rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::row_iterator::operator!=(const typename SpMat<eT>::const_iterator& rhs) const
  {
  return ((row != rhs.row) || (col != rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::row_iterator::operator!=(const typename SpMat<eT>::row_iterator& rhs) const
  {
  return ((row != rhs.row) || (col != rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::row_iterator::operator!=(const typename SpMat<eT>::const_row_iterator& rhs) const
  {
  return ((row != rhs.row) || (col != rhs.col));
  }

/**
 * Return true if this row_iterator does represent the same position as the given row_iterator.
 */
template<typename eT>
inline
bool
SpMat<eT>::row_iterator::operator==(const typename SpMat<eT>::iterator& rhs) const
  {
  return ((row == rhs.row) && (col == rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::row_iterator::operator==(const typename SpMat<eT>::const_iterator& rhs) const
  {
  return ((row == rhs.row) && (col == rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::row_iterator::operator==(const typename SpMat<eT>::row_iterator& rhs) const
  {
  return ((row == rhs.row) && (col == rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::row_iterator::operator==(const typename SpMat<eT>::const_row_iterator& rhs) const
  {
  return ((row == rhs.row) && (col == rhs.col));
  }

///////////////////////////////////////////////////////////////////////////////
// SpMat::const_row_iterator implementation                                  //
///////////////////////////////////////////////////////////////////////////////

/**
 * Initialize the const_row_iterator.
 */
template<typename eT>
inline
SpMat<eT>::const_row_iterator::const_row_iterator(SpMat<eT>& in_M, uword initial_pos)
  : M(in_M)
  , row(0)
  , col(0)
  , pos(initial_pos)
  {
  // We don't count zeroes in our position count, so we have to find the nonzero
  // value corresponding to the given initial position.

  // First check: if the given position is greater than the number of nonzero
  // elements, we are at the end.
  if (pos >= M.n_nonzero)
    {
    pos = M.n_nonzero;
    row = 0;
    col = M.n_cols;
    }
  else
    {
    // This is irritating because we don't know where the elements are in each
    // row.  What we will do is loop across all columns looking for elements in
    // row 0 (and add to our sum), then in row 1, and so forth, until we get to
    // the desired position.
    uword cur_pos = -1;
    uword cur_row = 0;
    uword cur_col = 0;

    while(true) // This loop is terminated from the inside.
      {
      // Is there anything in the column we are looking at?
      for (uword ind = 0; ((M.col_ptrs[cur_col] + ind < M.col_ptrs[cur_col + 1]) && (M.row_indices[M.col_ptrs[cur_col] + ind] <= cur_row)); ind++)
        {
        // There is something in this column.  Is it in the row we are looking
        // at?
        if (M.row_indices[M.col_ptrs[cur_col] + ind] == cur_row)
          {
          // Yes, it is what we are looking for.  Increment our current
          // position.
          if (++cur_pos == pos)
            {
            row = cur_row;
            col = cur_col;

            return;
            }

          // We are done with this column.  Break to the column incrementing code (directly below).
          break;
          }
        }

      cur_col++; // Done with the column.  Move on.
      if (cur_col == M.n_cols)
        {
        // We are out of columns.  Loop back to the beginning and look on the
        // next row.
        cur_col = 0;
        cur_row++;

        }

      }

    row = cur_row;
    col = cur_col;

    }

  }


template<typename eT>
inline
SpMat<eT>::const_row_iterator::const_row_iterator(const SpMat<eT>& in_M, uword in_row, uword in_col)
  : M(in_M)
  , row(in_row)
  , col(in_col)
  , pos(0)
  {
  // So we have a destination we want to be just after, but don't know what position that is.  Make another iterator to find out...
  const_row_iterator it(in_M);
  while((it.row < in_row) || ((it.row == in_row) && (it.col < in_col)))
    {
    it++;
    }

  // Now that it is at the right place, take its position.
  row = it.row;
  col = it.col;
  pos = it.pos;
  }

/**
 * Initialize the const_row_iterator from another const_row_iterator.
 */
template<typename eT>
inline
SpMat<eT>::const_row_iterator::const_row_iterator(const typename SpMat<eT>::const_row_iterator& other)
  : M(other.M)
  , row(other.row)
  , col(other.col)
  , pos(other.pos)
  {
  // Nothing to do.
  }

/**
 * Initialize the const_row_iterator from a row_iterator.
 */
template<typename eT>
inline
SpMat<eT>::const_row_iterator::const_row_iterator(const typename SpMat<eT>::row_iterator& other)
  : M(other.M)
  , row(other.row)
  , col(other.col)
  , pos(other.pos)
  {
  // Nothing to do.
  }

/**
 * Because this is a const iterator, we can return the value by itself.
 */
template<typename eT>
inline
const eT
SpMat<eT>::const_row_iterator::operator*() const
  {
  return M(row, col);
  }

/**
 * Increment the row_iterator.
 */
template<typename eT>
inline
typename SpMat<eT>::const_row_iterator&
SpMat<eT>::const_row_iterator::operator++()
  {
  // We just need to find the next nonzero element.
  pos++;

  // If we have exceeded the bounds, update accordingly.
  if (pos >= M.n_nonzero)
    {
    row = 0;
    col = M.n_cols;

    return *this;
    }

  // Otherwise, we need to search.
  uword cur_col = col;
  uword cur_row = row;

  while (true) // This loop is terminated from the inside.
    {
    // Increment the current column and see if we are now on a new row.
    if (++cur_col == M.n_cols)
      {
      cur_col = 0;
      cur_row++;
      }

    // Is there anything in this new column?
    for (uword ind = 0; ((M.col_ptrs[cur_col] + ind < M.col_ptrs[cur_col + 1]) && (M.row_indices[M.col_ptrs[cur_col] + ind] <= cur_row)); ind++)
      {
      if (M.row_indices[M.col_ptrs[cur_col] + ind] == cur_row)
        {
        // We have successfully incremented.
        row = cur_row;
        col = cur_col;

        return *this; // Now we are done.
        }
      }
    }
  }

/**
 * Increment the row_iterator (but do not return anything.
 */
template<typename eT>
inline
void
SpMat<eT>::const_row_iterator::operator++(int)
  {
  // We just need to find the next nonzero element.
  pos++;

  // Make sure we did not exceed the bounds.
  if (pos >= M.n_nonzero)
    {
    row = 0;
    col = M.n_cols;

    return; // Nothing else to do.
    }

  // Now, we need to search.
  uword cur_col = col;
  uword cur_row = row;

  while (true) // This loop is terminated from the inside.
    {
    // Increment the current column and see if we are now on a new row.
    if (++cur_col == M.n_cols)
      {
      cur_col = 0;
      cur_row++;
      }

    // Is there anything in this new column?
    for (uword ind = 0; ((M.col_ptrs[cur_col] + ind < M.col_ptrs[cur_col + 1]) && (M.row_indices[M.col_ptrs[cur_col] + ind] <= cur_row)); ind++)
      {
      if (M.row_indices[M.col_ptrs[cur_col] + ind] == cur_row)
        {
        // We have successfully incremented.
        row = cur_row;
        col = cur_col;

        return; // Now we are done.
        }
      }
    }
  }

/**
 * Decrement the row_iterator.
 */
template<typename eT>
inline
typename SpMat<eT>::const_row_iterator&
SpMat<eT>::const_row_iterator::operator--()
  {
  // We just need to find the previous element.
  if (pos == 0)
    {
    // We cannot decrement.
    return;
    }
  else if (pos == M.n_nonzero)
    {
    // We will be coming off the last element.  We need to reset the row correctly, because we set row = 0 in the last matrix position.
    row = M.n_rows - 1;
    }
  else if (pos > M.n_nonzero)
    {
    // We are in nowhere land...
    pos--;
    return *this;
    }

  pos--;

  // We have to search backwards.
  uword cur_col = col;
  uword cur_row = row;

  while (true) // This loop is terminated from the inside.
    {
    // Decrement the current column and see if we are now on a new row.  This is a uword so a negativity check won't work.
    if (--cur_col > M.n_cols /* this means it underflew */)
      {
      cur_col = M.n_cols - 1;
      cur_row--;
      }

    // Is there anything in this new column?
    for (uword ind = 0; ((M.col_ptrs[cur_col] + ind < M.col_ptrs[cur_col + 1]) && (M.row_indices[M.col_ptrs[cur_col] + ind] <= cur_row)); ind++)
      {
      if (M.row_indices[M.col_ptrs[cur_col] + ind] == cur_row)
        {
        // We have successfully decremented.
        row = cur_row;
        col = cur_col;

        return *this; // Now we are done.
        }
      }
    }
  }

/**
 * Decrement the row_iterator.
 */
template<typename eT>
inline
void
SpMat<eT>::const_row_iterator::operator--(int)
  {
  // We just need to find the previous element.
  if (pos == 0)
    {
    // We cannot decrement.
    return;
    }
  else if (pos == M.n_nonzero)
    {
    // We will be coming off the last element.  We need to reset the row correctly, because we set row = 0 in the last matrix position.
    row = M.n_rows - 1;
    }
  else if (pos > M.n_nonzero)
    {
    // We are in nowhere land...
    pos--;
    return;
    }

  pos--;

  // We have to search backwards.
  uword cur_col = col;
  uword cur_row = row;

  while (true) // This loop is terminated from the inside.
    {
    // Decrement the current column and see if we are now on a new row.  cur_col is a uword so a negativity check will not work.
    if (--cur_col > M.n_cols)
      {
      cur_col = M.n_cols - 1;
      cur_row--;
      }

    // Is there anything in this new column?
    for (uword ind = 0; ((M.col_ptrs[cur_col] + ind < M.col_ptrs[cur_col + 1]) && (M.row_indices[M.col_ptrs[cur_col] + ind] <= cur_row)); ind++)
      {
      if (M.row_indices[M.col_ptrs[cur_col] + ind] == cur_row)
        {
        // We have successfully decremented.
        row = cur_row;
        col = cur_col;

        return; // Now we are done.
        }
      }
    }
  }

/**
 * Return true if this row_iterator does not represent the same position as the given row_iterator.
 */
template<typename eT>
inline
bool
SpMat<eT>::const_row_iterator::operator!=(const typename SpMat<eT>::iterator& rhs) const
  {
  return ((row != rhs.row) || (col != rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::const_row_iterator::operator!=(const typename SpMat<eT>::const_iterator& rhs) const
  {
  return ((row != rhs.row) || (col != rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::const_row_iterator::operator!=(const typename SpMat<eT>::row_iterator& rhs) const
  {
  return ((row != rhs.row) || (col != rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::const_row_iterator::operator!=(const typename SpMat<eT>::const_row_iterator& rhs) const
  {
  return ((row != rhs.row) || (col != rhs.col));
  }

/**
 * Return true if this row_iterator does represent the same position as the given row_iterator.
 */
template<typename eT>
inline
bool
SpMat<eT>::const_row_iterator::operator==(const typename SpMat<eT>::iterator& rhs) const
  {
  return ((row == rhs.row) && (col == rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::const_row_iterator::operator==(const typename SpMat<eT>::const_iterator& rhs) const
  {
  return ((row == rhs.row) && (col == rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::const_row_iterator::operator==(const typename SpMat<eT>::row_iterator& rhs) const
  {
  return ((row == rhs.row) && (col == rhs.col));
  }

template<typename eT>
inline
bool
SpMat<eT>::const_row_iterator::operator==(const typename SpMat<eT>::const_row_iterator& rhs) const
  {
  return ((row == rhs.row) && (col == rhs.col));
  }

///////////////////////////////////////////////////////////////////////////////
// SpMat implementation                                                      //
///////////////////////////////////////////////////////////////////////////////

/**
 * Initialize a sparse matrix with size 0x0 (empty).
 */
template<typename eT>
inline
SpMat<eT>::SpMat()
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  {
  arma_extra_debug_sigprint_this(this);
  }

/**
 * Clean up the memory of a sparse matrix and destruct it.
 */
template<typename eT>
inline
SpMat<eT>::~SpMat()
  {
  arma_extra_debug_sigprint();
  // The three std::vector objects manage themselves.
  }

/**
 * Constructor with size given.
 */
template<typename eT>
inline
SpMat<eT>::SpMat(const uword in_rows, const uword in_cols)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  {
  arma_extra_debug_sigprint_this(this);

  init(in_rows, in_cols);
  }

/**
 * Assemble from text.
 */
template<typename eT>
inline
SpMat<eT>::SpMat(const char* text)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  {
  arma_extra_debug_sigprint_this(this);

  init(std::string(text));
  }

template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator=(const char* text)
  {
  arma_extra_debug_sigprint();

  init(std::string(text));
  }

template<typename eT>
inline
SpMat<eT>::SpMat(const std::string& text)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  {
  arma_extra_debug_sigprint();

  init(text);
  }

template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator=(const std::string& text)
  {
  arma_extra_debug_sigprint();

  init(text);
  }

template<typename eT>
inline
SpMat<eT>::SpMat(const SpMat<eT>& x)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  {
  arma_extra_debug_sigprint_this(this);

  init(x);
  }

/**
 * Simple operators with plain values.  These operate on every value in the
 * matrix, so a sparse matrix += 1 will turn all those zeroes into ones.  Be
 * careful and make sure that's what you really want!
 */
template<typename eT>
arma_inline
const SpMat<eT>&
SpMat<eT>::operator=(const eT val)
  {
  arma_extra_debug_sigprint();

  // Resize to 1x1 then set that to the right value.
  init(1, 1);
  get_value(0, 0) = val;

  return *this;

  }

template<typename eT>
arma_inline
const SpMat<eT>&
SpMat<eT>::operator+=(const eT val)
  {
  arma_extra_debug_sigprint();

  // This is not likely to be very fast, and could make memory usage explode.
  for (uword i = 0; i < n_elem; i++)
    get_value(i) += val;

  return *this;

  }

template<typename eT>
arma_inline
const SpMat<eT>&
SpMat<eT>::operator-=(const eT val)
  {
  arma_extra_debug_sigprint();

  // This is not likely to be very fast, and could make memory usage explode.
  for (uword i = 0; i < n_elem; i++)
    get_value(i) -= val;

  return *this;

  }

template<typename eT>
arma_inline
const SpMat<eT>&
SpMat<eT>::operator*=(const eT val)
  {
  arma_extra_debug_sigprint();

  if (val == 0)
    {
    // Everything will be zero.
    init(n_rows, n_cols);
    return *this;
    }

  // Let's use iterators over nonzero values, which is a lot faster.
  for (typename std::vector<eT>::iterator it = values.begin(); it != values.end(); it++)
    {
    (*it) *= val;
    }

  return *this;

  }

template<typename eT>
arma_inline
const SpMat<eT>&
SpMat<eT>::operator/=(const eT val)
  {
  arma_extra_debug_sigprint();

  if (val == 0) // I certainly hope not!
    {
    // We have to loop over everything.
    for (uword i = 0; i < n_elem; i++)
      get_value(i) /= val;

    }
  else
    {
    // We only have to loop over nonzero values.
    for (typename std::vector<eT>::iterator it = values.begin(); it != values.end(); it++)
      {
      (*it) /= val;
      }
    }

  return *this;

  }


template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator+=(const SpMat<eT>& x)
  {
  arma_extra_debug_sigprint();

  arma_debug_check((n_cols != x.n_cols) || (n_rows != x.n_rows), "SpMat::operator+=(SpMat): matrices must be the same size");

  // Iterate over nonzero values of other matrix.
  for (const_iterator it = x.begin(); it != x.end(); it++)
    {
    get_value(it.row, it.col) += *it;
    }

  return *this;

  }

template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator-=(const SpMat<eT>& x)
  {
  arma_extra_debug_sigprint();

  arma_debug_check((n_cols != x.n_cols) || (n_rows != x.n_rows), "SpMat::operator-=(SpMat): matrices must be the same size");

  // Iterate over nonzero values of other matrix.
  for (const_iterator it = x.begin(); it != x.end(); it++)
    {
    get_value(it.row, it.col) -= *it;
    }

  return *this;

  }

template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator*=(const SpMat<eT>& y)
  {
  arma_extra_debug_sigprint();

  /**
   * Matrix multiplication with sparse matrices.
   * General matrix multiplication is
   *
   *  z_ij = sum_{k = 0} x_ik y_kj
   *
   * But, we will only have a nonzero value in z_ij if there is a nonzero value
   * in X.row(i) or Y.col(j).
   *
   * Expanding on that, if there are no nonzero values in Y.col(j), then there
   * are no nonzero values in Z.col(j).  If there are no nonzero values in
   * X.row(i), there are no nonzero values in Z.row(i).
   *
   * Now, we can formulate a strategy.  We loop over rows of X and then loop
   * over columns of Y.  We can skip iterations where the row of X contains no
   * nonzero elements and inner iterations where the column of Y contains no
   * nonzero elements.
   *
   * Also, notationally, `this` is X; x is Y; and z is Z.
   *
   * Although this operation is in-place on X, we can't perform it like that and
   * store a temporary matrix z.
   */
  SpMat z(n_rows, y.n_cols);

  for (const_row_iterator x_row_it = begin_row(); x_row_it != end_row(); x_row_it++)
    {
    // The current row is x_row_it.row.
    // Now we iterate over values in y.col(x_row_it.row).
    // Y is used as the inner loop because the regular column iterator is less computationally expensive.
    for (const_iterator y_col_it = y.begin(); y_col_it != y.end(); y_col_it++)
      {
      // At this moment in the loop, we are calculating anything that is contributed to by *x_row_it and *y_col_it.
      // Given that our position is x_ab and y_bc, there will only be a contribution if x.col == y.row, and that
      // contribution will be in location z_ac.
      if (x_row_it.col == y_col_it.row)
        {
        z(x_row_it.row, y_col_it.col) += (*x_row_it) * (*y_col_it);
        }
      }
    }

  // With the calculation done, assign the temporary to ourselves.
  init(z);

  return *this;
  }

// This is in-place element-wise matrix multiplication.
template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator%=(const SpMat<eT>& x)
  {
  arma_extra_debug_sigprint();

  // We can do this with two iterators rather simply.
  iterator it = begin();
  const_iterator x_it = x.begin();

  while (it != end() && x_it != x.end())
    {
    // One of these will be further advanced than the other (or they will be at the same place).
    if ((it.row == x_it.row) && (it.col == x_it.col))
      {
      // There is an element at this place in both matrices.  Multiply.
      (*it) *= (*x_it);

      // Now move on to the next position.
      it++;
      x_it++;
      }

    else if ((it.col <= x_it.col) && (it.row < x_it.row))
      {
      // This case is when our matrix has an element which the other matrix does not.
      // So we must delete this element.
      (*it) = 0;

      // Because we have deleted the element, we now have to manually set the position...
      it.pos--;

      // Now we can increment our iterator.
      it++;
      }

    else /* if our iterator is ahead of the other matrix */
      {
      // In this case we don't need to set anything to 0; our element is already 0.
      // We can just increment the iterator of the other matrix.
      x_it++;
      }

    }

  // If we are not at the end of our matrix, then we must terminate the remaining elements.
  while (it != end())
    {
    (*it) = 0;

    // Hack to manually set the position right...
    it.pos--;
    it++; // ...and then an increment.
    }

  return *this;
  }

template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator/=(const SpMat<eT>& x)
  {
  arma_extra_debug_sigprint();

  // If you use this method, you are probably stupid or misguided, but for compatibility with Mat, we have implemented it anyway.
  // We have to loop over every element, which is not good.  In fact, it makes me physically sad to write this.
  for (uword i = 0; i < n_elem; i++)
    {
    at(i) /= x.at(i);
    }

  return *this;
  }


template<typename eT>
inline
SpMat<eT>::SpMat(const Mat<eT>& x)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  {
  arma_extra_debug_sigprint_this(this);

  (*this).operator=(x);
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator=(const Mat<eT>& x)
  {//No easy init function, will have to generate matrix manually.
  arma_extra_debug_sigprint();

  init(x.n_rows, x.n_cols);

  for(uword i = 0; i < x.n_rows; i++)
    {
    for(uword j = 0; j < x.n_cols; j++)
      {
      at(i, j) = x(i, j); // Let the proxy handle 0's.
      }
    }

  return *this;
  }

template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator+=(const Mat<eT>& x)
  { // This will probably defeat the purpose of using SpMat...
  arma_extra_debug_sigprint();

  arma_debug_check((n_cols != x.n_cols) || (n_rows != x.n_rows), "SpMat::operator+=(Mat): matrices must be the same size");

  for(uword i = 0; i < x.n_rows; i++)
    {
    for(uword j = 0; j < x.n_cols; j++)
      {
      at(i, j) += x(i, j); // Let the proxy handle zeros.
      }
    }

  return *this;
  }

template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator*=(const Mat<eT>& y)
  {
  arma_extra_debug_sigprint();

  arma_debug_check((n_cols != y.n_rows) || (n_rows != y.n_cols), "SpMat::operator*=(Mat): matrices must be the same size");

  /**
   * Matrix multiplication with a sparse matrix and a dense matrix.
   * General matrix multiplication is
   *
   *  z_ij = sum_{k = 0} x_ik y_kj
   *
   * But, we will only have a nonzero value in z_ij if there is a nonzero value
   * in X.row(i).
   *
   * Expanding on that, if there are no nonzero values in X.row(i), then there
   * are no nonzero values in Z.row(i).
   *
   * Now, we can formulate a strategy.  We loop over rows of X and then loop
   * over columns of Y.  We can skip iterations where the row of X contains no
   * nonzero elements and inner iterations where the column of Y contains no
   * nonzero elements.
   *
   * Also, notationally, `this` is X; x is Y; and z is Z.
   *
   * Although this operation is in-place on X, we can't perform it like that and
   * store a temporary matrix z.
   */
  SpMat z(n_rows, y.n_cols);

  for (const_row_iterator x_row_it = begin_row(); x_row_it != end_row(); x_row_it++)
    {
    // The current row is x_row_it.row.
    // Now we just want to use values where y.row = x_row_it.col.
    for (uword col = 0; col < z.n_cols; col++)
      {
      z(x_row_it.row, col) += (*x_row_it) * y.at(x_row_it.col, col);
      }
    }

  // Now copy the temporary to this.
  init(z);

  return *this;
  }

template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator/=(const Mat<eT>& x)
  {
  arma_extra_debug_sigprint();

  /**
   * Don't use this function.  It's not mathematically well-defined and wastes
   * cycles to trash all your data.  This is dumb.
   */
  arma_debug_check((n_cols != x.n_rows) || (n_rows != x.n_cols), "SpMat::operator/=(Mat): matrices must be the same size");

  for(uword i = 0; i < n_rows; i++) // You got a better idea?
    {
    for(uword j = 0; j < n_cols; j++)
      {
      at(i, j) /= x.at(i, j);
      }
    }

  return *this;
  }

template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator%=(const Mat<eT>& x)
  {
  arma_extra_debug_sigprint();

  /**
   * Element wise multiplication between two matrices.
   */
  arma_debug_check((n_cols != x.n_cols) || (n_rows != x.n_rows), "SpMat::operator%=(Mat): matrices must be the same size!");

  // Implement by hand (maybe someone else could do better).
  for(iterator it = begin(); it != end(); it++)
    {
    (*it) *= x(it.row, it.col);
    }

  return *this;
  }

template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator=(const SpMat<eT>& x)
  {
  arma_extra_debug_sigprint();

  init(x);

  return *this;
  }


template<typename eT>
template<typename T1>
inline
SpMat<eT>::SpMat(const BaseCube<eT, T1>& X)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  {
  arma_extra_debug_sigprint_this(this);

  (*this).operator=(X);
  }

template<typename eT>
template<typename T1>
inline
const SpMat<eT>& SpMat<eT>::operator=(const BaseCube<eT, T1>& X)
  {
  arma_extra_debug_sigprint();

  SpMat<eT>& out = *this; // convenience

  const unwrap_cube<T1> tmp(X.get_ref());
  const Cube<eT>& in = tmp.M;

  arma_debug_assert_cube_as_mat(out, in, "copy into matrix", false);

  const uword in_n_rows   = in.n_rows;
  const uword in_n_cols   = in.n_cols;
  const uword in_n_slices = in.n_slices;

  if(in_n_slices == 1)
    {
    out.set_size(in_n_rows, in_n_cols);

    for(uword col = 0; col < in_n_cols; ++col)
      {
      // Manually add each element.  The vectors are all clear because of the call to set_size.
      for(uword row = 0; row < in_n_rows; ++row)
        {
        double value = in(row, col, 0 /* only slice */);
        if (value != 0)
          {
          values.push_back(value);
          row_indices.push_back(row);
          }
        }
      // Now that we have added the elements in the column, set the column pointer correctly.
      col_ptrs[col + 1] = values.size();
      }
    }
  else
    {
    if(in_n_cols == 1)
      {
      out.set_size(in_n_rows, in_n_slices);

      for(uword col = 0; col < in_n_slices; ++col)
        {
        for(uword row = 0; row < in_n_rows; ++row)
          {
          double value = in(row, 0 /* only column */, col);
          if(value != 0)
            {
            values.push_back(value);
            row_indices.push_back(row);
            }
          }
        // Now that we have added the elements in the column, set the column pointer correctly.
        col_ptrs[col + 1] = values.size();
        }
      }
    else if(in_n_rows == 1)
      {
      out.set_size(in_n_cols, in_n_slices);

      for(uword col = 0; col < in_n_slices; ++col)
        {
        for(uword row = 0; row < in_n_cols; ++col)
          {
          double value = in(0 /* only row */, row, col);
          if(value != 0)
            {
            values.push_back(value);
            row_indices.push_back(row);
            }
          }
          col_ptrs[col + 1] = values.size();
        }
      }
    else
      {
      out.set_size(in_n_slices);

      for(uword elem = 0; elem < in_n_slices; ++elem)
        {
        double value = in(0, 0, elem);
        if(value != 0)
          {
          values.push_back(value);
          row_indices.push_back(row);
          }
        }

      col_ptrs[1] = values.size();
      }
    }

  return *this;
  }



template<typename eT>
template<typename T1>
inline
const SpMat<eT>&
SpMat<eT>::operator+=(const BaseCube<eT, T1>& X)
  {
  arma_extra_debug_sigprint();

  SpMat<eT>& out = *this;

  const unwrap_cube<T1> tmp(X.get_ref());
  const Cube<eT>& in = tmp.M;

  arma_debug_assert_cube_as_mat(out, in, "addition", true);

  const uword in_n_rows   = in.n_rows;
  const uword in_n_cols   = in.n_cols;
  const uword in_n_slices = in.n_slices;

  const uword out_n_rows  = out.n_rows;
  const uword out_n_cols  = out.n_cols;

  if(in_n_slices == 1)
    {
    for(uword col = 0; col < in_n_cols; ++col)
      {
      for(uword row = 0; row < in_n_rows; ++row)
        {
        out(row, col) += in(row, col, 0 /* only slice */);
        }
      }
    }
  else
    {
    if((in_n_rows == out_n_rows) && (in_n_cols == 1) && (in_n_slices == out_n_cols))
      {
      for(uword col = 0; col < in_n_slices; ++col)
        {
        for(uword row = 0; row < in_n_rows; ++row)
          {
          out(row, col) += in(row, 0, col);
          }
        }
      }
    else if((in_n_rows == 1) && (in_n_cols == out_n_rows) && (in_n_slices == out_n_cols))
      {
      for(uword col = 0; col < in_n_slices; ++col)
        {
        for(uword row = 0; row < in_n_cols; ++col)
          {
          out(row, col) += in(0, row, col);
          }
        }
      }
    else
      {
      out.set_size(in_n_slices);

      for(uword elem = 0; elem < in_n_slices; ++elem)
        {
        out(elem) += in(elem);
        }
      }
    }

  return *this;
  }




template<typename eT>
template<typename T1>
inline
const SpMat<eT>&
SpMat<eT>::operator-=(const BaseCube<eT, T1>& X)
  {
  arma_extra_debug_sigprint();

  SpMat<eT>& out = *this;

  const unwrap_cube<T1> tmp(X.get_ref());
  const Cube<eT>& in = tmp.M;

  arma_debug_assert_cube_as_mat(out, in, "addition", true);

  const uword in_n_rows   = in.n_rows;
  const uword in_n_cols   = in.n_cols;
  const uword in_n_slices = in.n_slices;

  const uword out_n_rows  = out.n_rows;
  const uword out_n_cols  = out.n_cols;

  if(in_n_slices == 1)
    {
    for(uword col = 0; col < in_n_cols; ++col)
      {
      for(uword row = 0; row < in_n_rows; ++row)
        {
        out(row, col) -= in(row, col, 0 /* only slice */);
        }
      }
    }
  else
    {
    if((in_n_rows == out_n_rows) && (in_n_cols == 1) && (in_n_slices == out_n_cols))
      {
      for(uword col = 0; col < in_n_slices; ++col)
        {
        for(uword row = 0; row < in_n_rows; ++row)
          {
          out(row, col) -= in(row, 0, col);
          }
        }
      }
    else if((in_n_rows == 1) && (in_n_cols == out_n_rows) && (in_n_slices == out_n_cols))
      {
      for(uword col = 0; col < in_n_slices; ++col)
        {
        for(uword row = 0; row < in_n_cols; ++col)
          {
          out(row, col) -= in(0, row, col);
          }
        }
      }
    else
      {
      out.set_size(in_n_slices);

      for(uword elem = 0; elem < in_n_slices; ++elem)
        {
        out(elem) -= in(elem);
        }
      }
    }

  return *this;
  }



template<typename eT>
template<typename T1>
inline
const SpMat<eT>&
SpMat<eT>::operator*=(const BaseCube<eT, T1>& X)
  {
  arma_extra_debug_sigprint();

  const Mat<eT> B(X);

  (*this).operator*=(B);

  return *this;
  }



template<typename eT>
template<typename T1>
inline
const SpMat<eT>&
SpMat<eT>::operator%=(const BaseCube<eT, T1>& X)
  {
  arma_extra_debug_sigprint();

  SpMat<eT>& out = *this;

  const unwrap_cube<T1> tmp(X.get_ref());
  const Cube<eT>& in = tmp.M;

  arma_debug_assert_cube_as_mat(out, in, "addition", true);

  const uword in_n_rows   = in.n_rows;
  const uword in_n_cols   = in.n_cols;
  const uword in_n_slices = in.n_slices;

  const uword out_n_rows  = out.n_rows;
  const uword out_n_cols  = out.n_cols;

  if(in_n_slices == 1)
    {
    for(uword col = 0; col < in_n_cols; ++col)
      {
      for(uword row = 0; row < in_n_rows; ++row)
        {
        out(row, col) *= in(row, col, 0 /* only slice */);
        }
      }
    }
  else
    {
    if((in_n_rows == out_n_rows) && (in_n_cols == 1) && (in_n_slices == out_n_cols))
      {
      for(uword col = 0; col < in_n_slices; ++col)
        {
        for(uword row = 0; row < in_n_rows; ++row)
          {
          out(row, col) *= in(row, 0, col);
          }
        }
      }
    else if((in_n_rows == 1) && (in_n_cols == out_n_rows) && (in_n_slices == out_n_cols))
      {
      for(uword col = 0; col < in_n_slices; ++col)
        {
        for(uword row = 0; row < in_n_cols; ++col)
          {
          out(row, col) *= in(0, row, col);
          }
        }
      }
    else
      {
      out.set_size(in_n_slices);

      for(uword elem = 0; elem < in_n_slices; ++elem)
        {
        out(elem) *= in(elem);
        }
      }
    }
  }



template<typename eT>
template<typename T1>
inline
const SpMat<eT>&
SpMat<eT>::operator/=(const BaseCube<eT, T1>& X)
  {
  arma_extra_debug_sigprint();

  SpMat<eT>& out = *this;

  const unwrap_cube<T1> tmp(X.get_ref());
  const Cube<eT>& in = tmp.M;

  arma_debug_assert_cube_as_mat(out, in, "addition", true);

  const uword in_n_rows   = in.n_rows;
  const uword in_n_cols   = in.n_cols;
  const uword in_n_slices = in.n_slices;

  const uword out_n_rows  = out.n_rows;
  const uword out_n_cols  = out.n_cols;

  if(in_n_slices == 1)
    {
    for(uword col = 0; col < in_n_cols; ++col)
      {
      for(uword row = 0; row < in_n_rows; ++row)
        {
        out(row, col) /= in(row, col, 0 /* only slice */);
        }
      }
    }
  else
    {
    if((in_n_rows == out_n_rows) && (in_n_cols == 1) && (in_n_slices == out_n_cols))
      {
      for(uword col = 0; col < in_n_slices; ++col)
        {
        for(uword row = 0; row < in_n_rows; ++row)
          {
          out(row, col) /= in(row, 0, col);
          }
        }
      }
    else if((in_n_rows == 1) && (in_n_cols == out_n_rows) && (in_n_slices == out_n_cols))
      {
      for(uword col = 0; col < in_n_slices; ++col)
        {
        for(uword row = 0; row < in_n_cols; ++col)
          {
          out(row, col) /= in(0, row, col);
          }
        }
      }
    else
      {
      out.set_size(in_n_slices);

      for(uword elem = 0; elem < in_n_slices; ++elem)
        {
        out(elem) /= in(elem);
        }
      }
    }
  }


/**
 * Functions on subviews.
 */
template<typename eT>
inline
SpMat<eT>::SpMat(const SpSubview<eT>& X)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  {
  arma_extra_debug_sigprint_this(this);

  (*this).operator=(X);
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator=(const SpSubview<eT>& X)
  {
  arma_extra_debug_sigprint();

  const uword in_n_cols = X.n_cols;
  const uword in_n_rows = X.n_rows;

  init(in_n_rows, in_n_cols);

  // Iterate through the matrix using the internal matrix's iterators.
  const_iterator it = const_iterator(X.m, X.aux_row1, X.aux_col1);

  while(it.col < (X.aux_col1 + in_n_cols) && it.row < (X.aux_row1 + in_n_rows))
    {
    // Is it within the proper range?
    if((it.row >= X.aux_row1) && (it.row < (X.aux_row1 + in_n_rows)))
      {
      values.push_back(*it);
      row_indices.push_back(it.row - X.aux_row1);
      ++col_ptrs[(it.col - X.aux_col1) + 1]; // This is not completely correct, we'll fix it later.
      }

    ++it;
    }

  // Now fix column pointers because their counts are not cumulative.
  for(uword col = 2; col <= in_n_cols; ++col)
    {
    col_ptrs[col] += col_ptrs[col - 1];
    }

  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator+=(const SpSubview<eT>& X)
  {
  arma_extra_debug_sigprint();

  arma_debug_check((n_cols != X.n_cols) || (n_rows != X.n_rows), "SpMat::operator+=(SpSubview): matrices must be the same size");

  const uword in_n_cols = X.n_cols;
  const uword in_n_rows = X.n_rows;

  const_iterator it = const_iterator(X.m, X.aux_row1, X.aux_col1);

  while(it.col < (X.aux_col1 + in_n_cols) && it.row < (X.aux_row1 + in_n_rows))
    {
    // Is it within the proper range?
    if((it.row >= X.aux_row1) && (it.row < (X.aux_row1 + in_n_rows)))
      {
      at(it.row - X.aux_row1, it.col - X.aux_col1) += (*it);
      }
    }

  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator-=(const SpSubview<eT>& X)
  {
  arma_extra_debug_sigprint();

  arma_debug_check((n_cols != X.n_cols) || (n_rows != X.n_rows), "SpMat::operator-=(SpSubview): matrices must be the same size");

  const uword in_n_cols = X.n_cols;
  const uword in_n_rows = X.n_rows;

  const_iterator it = const_iterator(X.m, X.aux_row1, X.aux_col1);

  while(it.col < (X.aux_col1 + in_n_cols) && it.row < (X.aux_row1 + in_n_rows))
    {
    // Is it within the proper range?
    if((it.row >= X.aux_row1) && (it.row < (X.aux_row1 + in_n_rows)))
      {
      at(it.row - X.aux_row1, it.col - X.aux_col1) -= (*it);
      }
    }

  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator*=(const SpSubview<eT>& y)
  {
  arma_extra_debug_sigprint();

  arma_debug_check((n_cols != y.n_rows), "SpMat::operator*=(SpSubview): matrices must be the same size");

  // See the documentation for SpMat::operator*=(SpMat).  This is based on that.
  SpMat z(n_rows, y.n_cols);

  for (const_row_iterator x_row_it = begin_row(); x_row_it != end_row(); ++x_row_it)
    {
    for (const_iterator y_col_it = const_iterator(y.m, y.aux_row1, y.aux_col1);
        (y_col_it.col < (y.aux_col1 + y.n_cols) && (y_col_it.row < (y.aux_row1 + y.n_rows))); ++y_col_it)
      {
      // At this moment in the loop, we are calculating anything that is contributed to by *x_row_it and *y_col_it.
      // Given that our position is x_ab and y_bc, there will only be a contribution if x.col == y.row, and that
      // contribution will be in location z_ac.
      if (x_row_it.col == (y_col_it.row - y.aux_row1))
        {
        z(x_row_it.row, (y_col_it.col - y.aux_col1)) += (*x_row_it) * (*y_col_it);
        }
      }
    }

  // With the calculation done, assign the temporary to ourselves.
  init(z);

  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator%=(const SpSubview<eT>& x)
  {
  arma_extra_debug_sigprint();

  arma_debug_check((n_cols != x.n_cols) || (n_rows != x.n_rows), "SpMat::operator%=(SpSubview): matrices must be the same size");

  // We want to iterate over whichever has fewer nonzero points.
  if (n_nonzero <= x.n_nonzero)
    {
    // Use our iterator.
    for (const_iterator it = begin(); it != end(); it++)
      {
      (*it) *= x(it.row, it.col);
      }
    }
  else
    {
    // Use their iterator.  A little more complex...
    const_iterator it = const_iterator(x.m, x.aux_row1, x.aux_col1);
    while((it.col < (x.aux_col1 + x.n_cols)) && (it.row < (x.aux_row1 + x.n_rows)))
      {
      if((it.row >= x.aux_row1) && (it.row < (x.aux_row1 + x.n_rows)))
        {
        at(it.row - x.aux_row1, it.col - x.aux_col1) *= (*it);
        }

      ++it;
      }
    }

  return *this;
  }


template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator/=(const SpSubview<eT>& x)
  {
  arma_extra_debug_sigprint();

  arma_debug_check((n_cols != x.n_cols) || (n_rows != x.n_rows), "SpMat::operator*=(SpSubview): matrices must be the same size");

  // There is no pretty way to do this.
  for(uword elem = 0; elem < n_elem; elem++)
    {
    at(elem) /= x(elem);
    }

  return *this;
  }

/**
 * Operators on regular subviews.
 */
template<typename eT>
inline
SpMat<eT>::SpMat(const subview<eT>& x)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  {
  arma_extra_debug_sigprint_this(this);

  (*this).operator=(x);
  }


template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator=(const subview<eT>& x)
  {
  arma_extra_debug_sigprint();

  // Set the size correctly.
  init(x.n_rows, x.n_cols);

  for(uword col = 0; col < x.n_cols; col++)
    {
    for(uword row = 0; row < x.n_rows; row++)
      {
      // Add any nonzero values.
      double value = x(row, col);
      if(value != 0)
        {
        values.push_back(value);
        row_indices.push_back(row);
        }
      }

    col_ptrs[col + 1] = values.size();
    }

  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator+=(const subview<eT>& x)
  {
  arma_extra_debug_sigprint();

  arma_debug_check((n_cols != x.n_cols) || (n_rows != x.n_rows), "SpMat::operator+=(subview): matrices must be the same size");

  // Loop over every element.
  for(uword elem = 0; elem < n_elem; elem++)
    {
    at(elem) += x(elem);
    }

  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator-=(const subview<eT>& x)
  {
  arma_extra_debug_sigprint();

  arma_debug_check((n_cols != x.n_cols) || (n_rows != x.n_rows), "SpMat::operator+=(subview): matrices must be the same size");

  // Loop over every element.
  for(uword elem = 0; elem < n_elem; elem++)
    {
    at(elem) -= x(elem);
    }

  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator*=(const subview<eT>& y)
  {
  arma_extra_debug_sigprint();

  arma_debug_check((n_cols != y.n_rows), "SpMat::operator*=(subview): matrices must be the same size");

  // Performed in the same fashion as operator*=(SpMat).  We can't use GMM because then we'd have to copy things.
  for (const_row_iterator x_row_it = begin_row(); x_row_it != end_row(); ++x_row_it)
    {
    for (uword col = 0; col < y.n_cols; ++col)
      {
      // At this moment in the loop, we are calculating anything that is contributed to by *x_row_it and *y_col_it.
      // Given that our position is x_ab and y_bc, there will only be a contribution if x.col == y.row, and that
      // contribution will be in location z_ac.
      z(x_row_it.row, col) += (*x_row_it) * y(x_row_it.col, col);
      }
    }

  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator%=(const subview<eT>& x)
  {
  arma_extra_debug_sigprint();

  arma_debug_check((n_cols != x.n_cols) || (n_rows != x.n_rows), "SpMat::operator+=(subview): matrices must be the same size");

  // Loop over every element.
  for(uword elem = 0; elem < n_elem; elem++)
    {
    at(elem) *= x(elem);
    }

  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator/=(const subview<eT>& x)
  {
  arma_extra_debug_sigprint();

  arma_debug_check((n_cols != x.n_cols) || (n_rows != x.n_rows), "SpMat::operator+=(subview): matrices must be the same size");

  // Loop over every element.
  for(uword elem = 0; elem < n_elem; elem++)
    {
    at(elem) /= x(elem);
    }

  return *this;
  }





template<typename eT>
arma_inline
SpSubview<eT>
SpMat<eT>::row(const uword row_num)
  {
  arma_extra_debug_sigprint();

  return SpSubview<eT>(*this, row_num, 0, 1, n_cols);
  }

template<typename eT>
arma_inline
const SpSubview<eT>
SpMat<eT>::row(const uword row_num) const
  {
  arma_extra_debug_sigprint();

  return SpSubview<eT>(*this, row_num, 0, 1, n_cols);
  }

template<typename eT>
inline
SpSubview<eT>
SpMat<eT>::operator()(const uword row_num, const span& col_span)
  {
  arma_extra_debug_sigprint();

  return SpSubview<eT>(*this, row_num, col_span.a, col_span.b - col_span.a + 1);
  }

template<typename eT>
inline
const SpSubview<eT>
SpMat<eT>::operator()(const uword row_num, const span& col_span) const
  {
  arma_extra_debug_sigprint();

  return SpSubview<eT>(*this, row_num, col_span.a, col_span.b - col_span.a + 1);
  }

template<typename eT>
arma_inline
SpSubview<eT>
SpMat<eT>::col(const uword col_num)
  {
  arma_extra_debug_sigprint();

  return SpSubview<eT>(*this, 0, col_num, n_rows, 1);
  }

template<typename eT>
arma_inline
const SpSubview<eT>
SpMat<eT>::col(const uword col_num) const
  {
  arma_extra_debug_sigprint();

  return SpSubview<eT>(*this, 0, col_num, n_rows, 1);
  }

template<typename eT>
inline
SpSubview<eT>
SpMat<eT>::operator()(const span& row_span, const uword col_num)
  {
  arma_extra_debug_sigprint();

  return SpSubview<eT>(*this, row_span.a, col_num, row_span.b - row_span.a + 1, 0);
  }

template<typename eT>
inline
const SpSubview<eT>
SpMat<eT>::operator()(const span& row_span, const uword col_num) const
  {
  arma_extra_debug_sigprint();

  return SpSubview<eT>(*this, row_span.a, col_num, row_span.b - row_span.a + 1, 0);
  }

template<typename eT>
inline
SpCol<eT>
SpMat<eT>::unsafe_col(const uword col_num)
  {
  arma_extra_debug_sigprint();
  }

template<typename eT>
inline
const SpCol<eT>
SpMat<eT>::unsafe_col(const uword col_num) const
  {
  arma_extra_debug_sigprint();
  }

template<typename eT>
arma_inline
SpSubview<eT>
SpMat<eT>::rows(const uword in_row1, const uword in_row2)
  {
  arma_extra_debug_sigprint();

  arma_debug_check
    (
    (in_row1 > in_row2) || (in_row2 >= n_rows),
    "SpMat::rows(): indices out of bounds or incorrectly used"
    );

  const uword subview_n_rows = in_row2 - in_row1 + 1;

  return SpSubview<eT>(*this, in_row1, 0, subview_n_rows, n_cols);
  }

template<typename eT>
arma_inline
const SpSubview<eT>
SpMat<eT>::rows(const uword in_row1, const uword in_row2) const
  {
  arma_extra_debug_sigprint();

  arma_debug_check
    (
    (in_row1 > in_row2) || (in_row2 >= n_rows),
    "SpMat::rows(): indices out of bounds or incorrectly used"
    );

  const uword subview_n_rows = in_row2 - in_row1 + 1;

  return SpSubview<eT>(*this, in_row1, 0, subview_n_rows, n_cols);
  }

template<typename eT>
arma_inline
SpSubview<eT>
SpMat<eT>::cols(const uword in_col1, const uword in_col2)
  {
  arma_extra_debug_sigprint();

  arma_debug_check
    (
    (in_col1 > in_col2) || (in_col2 >= n_cols),
    "SpMat::cols(): indices out of bounds or incorrectly used"
    );

  const uword subview_n_cols = in_col2 - in_col1 + 1;

  return SpSubview<eT>(*this, 0, in_col1, n_rows, subview_n_cols);
  }

template<typename eT>
arma_inline
const SpSubview<eT>
SpMat<eT>::cols(const uword in_col1, const uword in_col2) const
  {
  arma_extra_debug_sigprint();

  arma_debug_check
    (
    (in_col1 > in_col2) || (in_col2 >= n_cols),
    "SpMat::cols(): indices out of bounds or incorrectly used"
    );

  const uword subview_n_cols = in_col2 - in_col1 + 1;

  return SpSubview<eT>(*this, 0, in_col1, n_rows, subview_n_cols);
  }

template<typename eT>
arma_inline
SpSubview<eT>
SpMat<eT>::submat(const uword in_row1, const uword in_col1, const uword in_row2, const uword in_col2)
  {
  arma_extra_debug_sigprint();

  arma_debug_check
    (
    (in_row1 > in_row2) || (in_col1 >  in_col2) || (in_row2 >= n_rows) || (in_col2 >= n_cols),
    "SpMat::submat(): indices out of bounds or incorrectly used"
    );

  const uword subview_n_rows = in_row2 - in_row1 + 1;
  const uword subview_n_cols = in_col2 - in_col1 + 1;

  return SpSubview<eT>(*this, in_row1, in_col1, subview_n_rows, subview_n_cols);
  }

template<typename eT>
arma_inline
const SpSubview<eT>
SpMat<eT>::submat(const uword in_row1, const uword in_col1, const uword in_row2, const uword in_col2) const
  {
  arma_extra_debug_sigprint();

  return submat(in_row1, in_col1, in_row2, in_col2);
  }

template<typename eT>
inline
SpSubview<eT>
SpMat<eT>::submat    (const span& row_span, const span& col_span)
  {
  arma_extra_debug_sigprint();

  const bool row_all = row_span.whole;
  const bool col_all = col_span.whole;
  
  const uword local_n_rows = n_rows;
  const uword local_n_cols = n_cols;
  
  const uword in_row1       = row_all ? 0            : row_span.a;
  const uword in_row2       =                          row_span.b;
  const uword submat_n_rows = row_all ? local_n_rows : in_row2 - in_row1 + 1; 
  
  const uword in_col1       = col_all ? 0            : col_span.a;
  const uword in_col2       =                          col_span.b;
  const uword submat_n_cols = col_all ? local_n_cols : in_col2 - in_col1 + 1; 
  
  arma_debug_check
    (    
    ( row_all ? false : ((in_row1 > in_row2) || (in_row2 >= local_n_rows)) )
    ||   
    ( col_all ? false : ((in_col1 > in_col2) || (in_col2 >= local_n_cols)) )
    ,    
    "SpMat::submat(): indices out of bounds or incorrectly used"
    );   
  
  return SpSubview<eT>(*this, in_row1, in_col1, submat_n_rows, submat_n_cols);

  }

template<typename eT>
inline
const SpSubview<eT>
SpMat<eT>::submat    (const span& row_span, const span& col_span) const
  {
  arma_extra_debug_sigprint();

  return submat(row_span, col_span);
  }

template<typename eT>
inline
SpSubview<eT>
SpMat<eT>::operator()(const span& row_span, const span& col_span)
  {
  arma_extra_debug_sigprint();

  return submat(row_span, col_span);
  }

template<typename eT>
inline
const SpSubview<eT>
SpMat<eT>::operator()(const span& row_span, const span& col_span) const
  {
  arma_extra_debug_sigprint();

  return submat(row_span, col_span);
  }

/**
 * Operators for generated matrices.
 */
template<typename eT>
template<typename gen_type>
inline
SpMat<eT>::SpMat(const Gen<eT, gen_type>& X)
  : n_rows(X.n_rows)
  , n_cols(X.n_cols)
  , n_elem(n_rows * n_cols)
  , n_nonzero(0)
  {
  arma_extra_debug_sigprint_this(this);

  X.apply(*this);
  }



template<typename eT>
template<typename gen_type>
inline
const SpMat<eT>&
SpMat<eT>::operator=(const Gen<eT, gen_type>& X)
  {
  arma_extra_debug_sigprint();

  set_size(X.n_rows, X.n_cols);

  X.apply(*this);

  return *this;
  }



template<typename eT>
template<typename gen_type>
inline
const SpMat<eT>&
SpMat<eT>::operator+=(const Gen<eT, gen_type>& X)
  {
  arma_extra_debug_sigprint();

  X.apply_inplace_plus(*this);

  return *this;
  }



template<typename eT>
template<typename gen_type>
inline
const SpMat<eT>&
SpMat<eT>::operator-=(const Gen<eT, gen_type>& X)
  {
  arma_extra_debug_sigprint();

  X.apply_inplace_minus(*this);

  return *this;
  }



template<typename eT>
template<typename gen_type>
inline
const SpMat<eT>&
SpMat<eT>::operator*=(const Gen<eT, gen_type>& X)
  {
  arma_extra_debug_sigprint();

  const Mat<eT> tmp(X);

  return (*this).operator*=(tmp);
  }



template<typename eT>
template<typename gen_type>
inline
const SpMat<eT>&
SpMat<eT>::operator%=(const Gen<eT, gen_type>& X)
  {
  arma_extra_debug_sigprint();

  X.apply_inplace_schur(*this);

  return *this;
  }



template<typename eT>
template<typename gen_type>
inline
const SpMat<eT>&
SpMat<eT>::operator/=(const Gen<eT, gen_type>& X)
  {
  arma_extra_debug_sigprint();

  X.apply_inplace_div(*this);

  return *this;
  }



/**
 * Basic operators on Op<>.
 */
template<typename eT>
template<typename T1, typename op_type>
inline
SpMat<eT>::SpMat(const Op<T1, op_type>& X)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  {
  arma_extra_debug_sigprint_this(this);

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  op_type::apply(*this, X);
  }

template<typename eT>
template<typename T1, typename op_type>
inline
const SpMat<eT>&
SpMat<eT>::operator=(const Op<T1, op_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  op_type::apply(*this, X);

  return *this;
  }

template<typename eT>
template<typename T1, typename op_type>
inline
const SpMat<eT>&
SpMat<eT>::operator+=(const Op<T1, op_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator+=(m);
  }

template<typename eT>
template<typename T1, typename op_type>
inline
const SpMat<eT>&
SpMat<eT>::operator-=(const Op<T1, op_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator-=(m);
  }

template<typename eT>
template<typename T1, typename op_type>
inline
const SpMat<eT>&
SpMat<eT>::operator*=(const Op<T1, op_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator*=(m);
  }

template<typename eT>
template<typename T1, typename op_type>
inline
const SpMat<eT>&
SpMat<eT>::operator%=(const Op<T1, op_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator%=(m);
  }

template<typename eT>
template<typename T1, typename op_type>
inline
const SpMat<eT>&
SpMat<eT>::operator/=(const Op<T1, op_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator/=(m);
  }

/**
 * Basic operators on eOp<>.
 */
template<typename eT>
template<typename T1, typename eop_type>
inline
SpMat<eT>::SpMat(const eOp<T1, eop_type>& X)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  {
  arma_extra_debug_sigprint_this(this);

  set_size(X.get_n_rows(), X.get_n_cols());

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  eop_type::apply(*this, X);
  }

template<typename eT>
template<typename T1, typename eop_type>
inline
const SpMat<eT>&
SpMat<eT>::operator=(const eOp<T1, eop_type>& X)
  {
  arma_extra_debug_sigprint();

  set_size(X.get_n_rows(), X.get_n_cols());

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  eop_type::apply(*this, X);

  return *this;
  }

template<typename eT>
template<typename T1, typename eop_type>
inline
const SpMat<eT>&
SpMat<eT>::operator+=(const eOp<T1, eop_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  eop_type::apply_inplace_plus(*this, X);

  return *this;
  }

template<typename eT>
template<typename T1, typename eop_type>
inline
const SpMat<eT>&
SpMat<eT>::operator-=(const eOp<T1, eop_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  eop_type::apply_inplace_minus(*this, X);

  return *this;
  }

template<typename eT>
template<typename T1, typename eop_type>
inline
const SpMat<eT>&
SpMat<eT>::operator*=(const eOp<T1, eop_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator*=(m);
  }

template<typename eT>
template<typename T1, typename eop_type>
inline
const SpMat<eT>&
SpMat<eT>::operator%=(const eOp<T1, eop_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  eop_type::apply_inplace_schur(*this, X);

  return *this;
  }

template<typename eT>
template<typename T1, typename eop_type>
inline
const SpMat<eT>&
SpMat<eT>::operator/=(const eOp<T1, eop_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  eop_type::apply_inplace_div(*this, X);

  return *this;
  }

/**
 * Basic operators on mtOp<>.
 */
template<typename eT>
template<typename T1, typename op_type>
inline
SpMat<eT>::SpMat(const mtOp<eT, T1, op_type>& X)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  {
  arma_extra_debug_sigprint_this(this);

  set_size(X.get_n_rows(), X.get_n_cols());

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  op_type::apply(*this, X);
  }

template<typename eT>
template<typename T1, typename op_type>
inline
const SpMat<eT>&
SpMat<eT>::operator=(const mtOp<eT, T1, op_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  set_size(X.get_n_rows(), X.get_n_cols());

  op_type::apply(*this, X);

  return *this;
  }

template<typename eT>
template<typename T1, typename op_type>
inline
const SpMat<eT>&
SpMat<eT>::operator+=(const mtOp<eT, T1, op_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator+=(m);
  }

template<typename eT>
template<typename T1, typename op_type>
inline
const SpMat<eT>&
SpMat<eT>::operator-=(const mtOp<eT, T1, op_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator-=(m);
  }

template<typename eT>
template<typename T1, typename op_type>
inline
const SpMat<eT>&
SpMat<eT>::operator*=(const mtOp<eT, T1, op_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator*=(m);
  }

template<typename eT>
template<typename T1, typename op_type>
inline
const SpMat<eT>&
SpMat<eT>::operator%=(const mtOp<eT, T1, op_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator%=(m);
  }

template<typename eT>
template<typename T1, typename op_type>
inline
const SpMat<eT>&
SpMat<eT>::operator/=(const mtOp<eT, T1, op_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator/=(m);
  }

/**
 * Basic operators on Glue<>.
 */
template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
SpMat<eT>::SpMat(const Glue<T1, T2, glue_type>& X)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  {
  arma_extra_debug_sigprint_this(this);

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  set_size(X.get_n_rows(), X.get_n_cols());

  glue_type::apply(*this, X);
  }

template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator=(const Glue<T1, T2, glue_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  set_size(X.get_n_rows(), X.get_n_cols());

  glue_type::apply(*this, X);

  return *this;
  }

template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator+=(const Glue<T1, T2, glue_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator+=(m);
  }

template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator-=(const Glue<T1, T2, glue_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator-=(m);
  }

template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator*=(const Glue<T1, T2, glue_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator*=(m);
  }

template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator%=(const Glue<T1, T2, glue_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator%=(m);
  }

template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator/=(const Glue<T1, T2, glue_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator/=(m);
  }

/**
 * Basic operators on eGlue<>.
 */
template<typename eT>
template<typename T1, typename T2, typename eglue_type>
inline
SpMat<eT>::SpMat(const eGlue<T1, T2, eglue_type>& X)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  {
  arma_extra_debug_sigprint_this(this);

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  set_size(X.get_n_rows(), X.get_n_cols());

  eglue_type::apply(*this, X);
  }

template<typename eT>
template<typename T1, typename T2, typename eglue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator=(const eGlue<T1, T2, eglue_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  set_size(X.get_n_rows(), X.get_n_cols());

  eglue_type::apply(*this, X);

  return *this;
  }

template<typename eT>
template<typename T1, typename T2, typename eglue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator+=(const eGlue<T1, T2, eglue_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator+=(m);
  }

template<typename eT>
template<typename T1, typename T2, typename eglue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator-=(const eGlue<T1, T2, eglue_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator-=(m);
  }

template<typename eT>
template<typename T1, typename T2, typename eglue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator*=(const eGlue<T1, T2, eglue_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator*=(m);
  }

template<typename eT>
template<typename T1, typename T2, typename eglue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator%=(const eGlue<T1, T2, eglue_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator%=(m);
  }

template<typename eT>
template<typename T1, typename T2, typename eglue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator/=(const eGlue<T1, T2, eglue_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator/=(m);
  }

/**
 * Basic operators on mtGlue<>.
 */
template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
SpMat<eT>::SpMat(const mtGlue<eT, T1, T2, glue_type>& X)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  {
  arma_extra_debug_sigprint_this(this);

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  set_size(X.get_n_cols(), X.get_n_rows());

  glue_type::apply(*this, X);
  }

template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator=(const mtGlue<eT, T1, T2, glue_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  set_size(X.get_n_rows(), X.get_n_cols());

  glue_type::apply(*this, X);

  return *this;
  }

template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator+=(const mtGlue<eT, T1, T2, glue_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator+=(m);
  }

template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator-=(const mtGlue<eT, T1, T2, glue_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator-=(m);
  }

template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator*=(const mtGlue<eT, T1, T2, glue_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator*=(m);
  }

template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator%=(const mtGlue<eT, T1, T2, glue_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator%=(m);
  }

template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator/=(const mtGlue<eT, T1, T2, glue_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::value == false ));

  Mat<eT> m(X);

  return (*this).operator/=(m);
  }


/**
 * Element access; acces the i'th element (works identically to the Mat accessors).
 * If there is nothing at element i, 0 is returned.
 *
 * @param i Element to access.
 */

template<typename eT>
arma_inline
arma_warn_unused
SpValProxy<eT>
SpMat<eT>::operator[](const uword i)
  {
  return get_value(i);
  }

template<typename eT>
arma_inline
arma_warn_unused
eT
SpMat<eT>::operator[](const uword i) const
  {
  return get_value(i);
  }

template<typename eT>
arma_inline
arma_warn_unused
SpValProxy<eT>
SpMat<eT>::at(const uword i)
  {
  return get_value(i);
  }

template<typename eT>
arma_inline
arma_warn_unused
eT
SpMat<eT>::at(const uword i) const
  {
  return get_value(i);
  }

template<typename eT>
arma_inline
arma_warn_unused
SpValProxy<eT>
SpMat<eT>::operator()(const uword i)
  {
  arma_debug_check( (i >= n_elem), "SpMat::operator(): out of bounds");
  return get_value(i);
  }

template<typename eT>
arma_inline
arma_warn_unused
eT
SpMat<eT>::operator()(const uword i) const
  {
  arma_debug_check( (i >= n_elem), "SpMat::operator(): out of bounds");
  return get_value(i);
  }

/**
 * Element access; access the element at row in_rows and column in_col.
 * If there is nothing at that position, 0 is returned.
 */

template<typename eT>
arma_inline
arma_warn_unused
SpValProxy<eT>
SpMat<eT>::at(const uword in_row, const uword in_col)
  {
  return get_value(in_row, in_col);
  }

template<typename eT>
arma_inline
arma_warn_unused
eT
SpMat<eT>::at(const uword in_row, const uword in_col) const
  {
  return get_value(in_row, in_col);
  }

template<typename eT>
arma_inline
arma_warn_unused
SpValProxy<eT>
SpMat<eT>::operator()(const uword in_row, const uword in_col)
  {
  arma_debug_check( ((in_row >= n_rows) || (in_col >= n_cols)), "SpMat::operator(): out of bounds");
  return get_value(in_row, in_col);
  }

template<typename eT>
arma_inline
arma_warn_unused
eT
SpMat<eT>::operator()(const uword in_row, const uword in_col) const
  {
  arma_debug_check( ((in_row >= n_rows) || (in_col >= n_cols)), "SpMat::operator(): out of bounds");
  return get_value(in_row, in_col);
  }


template<typename eT>
arma_inline
const SpMat<eT>&
SpMat<eT>::operator++()
  {
  arma_extra_debug_sigprint();

  // Prefix increment everything...
  uword i, j;

  for(i = 0, j = 1; j < n_elem; i += 2, j += 2)
    {
    ++at(i);
    ++at(j);
    }

  if(i < n_elem)
    {
    ++at(i);
    }

  return *this;
  }


template<typename eT>
arma_inline
void
SpMat<eT>::operator++(int)
  {
  arma_extra_debug_sigprint();

  // Postfix increment everything...
  uword i, j;

  for(i = 0, j = 1; j < n_elem; i += 2, j += 2)
    {
    at(i)++;
    at(j)++;
    }

  if(i < n_elem)
    {
    at(i)++;
    }
  }


template<typename eT>
arma_inline
const SpMat<eT>&
SpMat<eT>::operator--()
  {
  arma_extra_debug_sigprint();

  // Prefix decrement everything...
  uword i, j;

  for(i = 0, j = 1; j < n_elem; i += 2, j += 2)
    {
    --at(i);
    --at(j);
    }

  if(i < n_elem)
    {
    --at(i);
    }

  return *this;
  }


template<typename eT>
arma_inline
void
SpMat<eT>::operator--(int)
  {
  arma_extra_debug_sigprint();

  // Postfix decrement everything...
  uword i, j;

  for(i = 0, j = 1; j < n_elem; i += 2, j += 2)
    {
    at(i)--;
    at(j)--;
    }

  if(i < n_elem)
    {
    at(i)--;
    }
  }


/**
 * Get the minimum or the maximum of the matrix.
 */
template<typename eT>
inline
arma_warn_unused
eT
SpMat<eT>::min() const
  {
  arma_extra_debug_sigprint();

  arma_debug_check((n_elem == 0), "min(): object has no elements");

  if (n_nonzero == 0)
    {
    return 0;
    }

  eT val = op_min::direct_min(values, n_nonzero);

  if ((val > 0) && (n_nonzero < n_elem)) // A sparse 0 is less.
    {
    val = 0;
    }

  return val;

  }

/**
 * Check if matrix is empty (no size, no values).
 */
template<typename eT>
arma_inline
arma_warn_unused
bool
SpMat<eT>::is_empty() const
  {
  return(n_elem == 0);
  }



//! returns true if the object can be interpreted as a column or row vector
template<typename eT>
arma_inline
arma_warn_unused
bool
SpMat<eT>::is_vec() const
  {
  return ( (n_rows == 1) || (n_cols == 1) );
  }



//! returns true if the object can be interpreted as a row vector
template<typename eT>
arma_inline
arma_warn_unused
bool
SpMat<eT>::is_rowvec() const
  {
  return (n_rows == 1);
  }



//! returns true if the object can be interpreted as a column vector
template<typename eT>
arma_inline
arma_warn_unused
bool
SpMat<eT>::is_colvec() const
  {
  return (n_cols == 1);
  }



//! returns true if the object has the same number of non-zero rows and columnns
template<typename eT>
arma_inline
arma_warn_unused
bool
SpMat<eT>::is_square() const
  {
  return (n_rows == n_cols);
  }



//! returns true if all of the elements are finite
template<typename eT>
inline
arma_warn_unused
bool
SpMat<eT>::is_finite() const
  {
  for(uword i = 0; i < values.size(); i++)
    {
    if(arma_isfinite(values[i]) == false)
      {
      return false;
      }
    }

  return true; // No infinite values.
  }



//! returns true if the given index is currently in range
template<typename eT>
arma_inline
arma_warn_unused
bool
SpMat<eT>::in_range(const uword i) const
  {
  return (i < n_elem);
  }


//! returns true if the given start and end indices are currently in range
template<typename eT>
arma_inline
arma_warn_unused
bool
SpMat<eT>::in_range(const span& x) const
  {
  arma_extra_debug_sigprint();

  if(x.whole == true)
    {
    return true;
    }
  else
    {
    const uword a = x.a;
    const uword b = x.b;

    return ( (a <= b) && (b < n_elem) );
    }
  }



//! returns true if the given location is currently in range
template<typename eT>
arma_inline
arma_warn_unused
bool
SpMat<eT>::in_range(const uword in_row, const uword in_col) const
  {
  return ( (in_row < n_rows) && (in_col < n_cols) );
  }



template<typename eT>
arma_inline
arma_warn_unused
bool
SpMat<eT>::in_range(const span& row_span, const uword in_col) const
  {
  arma_extra_debug_sigprint();

  if(row_span.whole == true)
    {
    return (in_col < n_cols);
    }
  else
    {
    const uword in_row1 = row_span.a;
    const uword in_row2 = row_span.b;

    return ( (in_row1 <= in_row2) && (in_row2 < n_rows) && (in_col < n_cols) );
    }
  }


template<typename eT>
arma_inline
arma_warn_unused
bool
SpMat<eT>::in_range(const uword in_row, const span& col_span) const
  {
  arma_extra_debug_sigprint();

  if(col_span.whole == true)
    {
    return (in_row < n_rows);
    }
  else
    {
    const uword in_col1 = col_span.a;
    const uword in_col2 = col_span.b;

    return ( (in_row < n_rows) && (in_col1 <= in_col2) && (in_col2 < n_cols) );
    }
  }



template<typename eT>
arma_inline
arma_warn_unused
bool
SpMat<eT>::in_range(const span& row_span, const span& col_span) const
  {
  arma_extra_debug_sigprint();

  const uword in_row1 = row_span.a;
  const uword in_row2 = row_span.b;

  const uword in_col1 = col_span.a;
  const uword in_col2 = col_span.b;

  const bool rows_ok = row_span.whole ? true : ( (in_row1 <= in_row2) && (in_row2 < n_rows) );
  const bool cols_ok = col_span.whole ? true : ( (in_col1 <= in_col2) && (in_col2 < n_cols) );

  return ( (rows_ok == true) && (cols_ok == true) );
  }



/**
 * Matrix printing, prepends supplied text.
 * Prints 0 wherever no element exists.
 */
template<typename eT>
inline
void
SpMat<eT>::impl_print(const std::string& extra_text) const
  {
  arma_extra_debug_sigprint();

  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = cout.width();

    cout << extra_text << '\n';

    cout.width(orig_width);
    }

  arma_ostream_new::print(cout, *this, true);

  }


template<typename eT>
inline
void
SpMat<eT>::impl_print(std::ostream& user_stream, const std::string& extra_text) const
  {
  arma_extra_debug_sigprint();

  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = cout.width();

    cout << extra_text << '\n';

    cout.width(orig_width);
    }

  arma_ostream_new::print(user_stream, *this, true);

  }

template<typename eT>
inline
void
SpMat<eT>::impl_print_trans(const std::string& extra_text) const
  {
  arma_extra_debug_sigprint();

  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = cout.width();

    cout << extra_text << '\n';

    cout.width(orig_width);
    }

  arma_ostream_new::print_trans(cout, *this, true);

  }

template<typename eT>
inline
void
SpMat<eT>::impl_print_trans(std::ostream& user_stream, const std::string& extra_text) const
  {
  arma_extra_debug_sigprint();
  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = cout.width();

    cout << extra_text << '\n';

    cout.width(orig_width);
    }

  arma_ostream_new::print_trans(user_stream, *this, true);
  }



template<typename eT>
inline
void
SpMat<eT>::impl_raw_print(const std::string& extra_text) const
  {
  arma_extra_debug_sigprint();

  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = cout.width();

    cout << extra_text << '\n';

    cout.width(orig_width);
    }

  arma_ostream_new::print(cout, *this, false);

  }


template<typename eT>
inline
void
SpMat<eT>::impl_raw_print(std::ostream& user_stream, const std::string& extra_text) const
  {
  arma_extra_debug_sigprint();

  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = cout.width();

    cout << extra_text << '\n';

    cout.width(orig_width);
    }

  arma_ostream_new::print(user_stream, *this, false);

  }

template<typename eT>
inline
void
SpMat<eT>::impl_raw_print_trans(const std::string& extra_text) const
  {
  arma_extra_debug_sigprint();

  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = cout.width();

    cout << extra_text << '\n';

    cout.width(orig_width);
    }

  arma_ostream_new::print_trans(cout, *this, false);

  }

template<typename eT>
inline
void
SpMat<eT>::impl_raw_print_trans(std::ostream& user_stream, const std::string& extra_text) const
  {
  arma_extra_debug_sigprint();
  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = cout.width();

    cout << extra_text << '\n';

    cout.width(orig_width);
    }

  arma_ostream_new::print_trans(user_stream, *this, false);
  }


//! Set the size to the size of another matrix.
template<typename eT>
template<typename eT2>
inline
void
SpMat<eT>::copy_size(const SpMat<eT2>& m)
  {
  arma_extra_debug_sigprint();

  init(m.n_rows, m.n_cols);
  }

template<typename eT>
template<typename eT2>
inline
void
SpMat<eT>::copy_size(const Mat<eT2>& m)
  {
  arma_extra_debug_sigprint();

  init(m.n_rows, m.n_cols);
  }

/**
 * Resize the matrix to a given size.  The matrix will be resized to be a column vector (i.e. in_elem columns, 1 row).
 *
 * @param in_elem Number of elements to allow.
 */
template<typename eT>
inline
void
SpMat<eT>::set_size(const uword in_elem)
  {
  arma_extra_debug_sigprint();

  init(1, in_elem);
  }

/**
 * Resize the matrix to a given size.
 *
 * @param in_rows Number of rows to allow.
 * @param in_cols Number of columns to allow.
 */
template<typename eT>
inline
void
SpMat<eT>::set_size(const uword in_rows, const uword in_cols)
  {
  arma_extra_debug_sigprint();

  init(in_rows, in_cols);
  }



template<typename eT>
inline
void
SpMat<eT>::reshape(const uword in_rows, const uword in_cols, const uword dim)
  {
  arma_extra_debug_sigprint();

  if (dim == 0)
    {
    // We have to modify all of the relevant row indices and the relevant column pointers.
    // Iterate over all the points to do this.  We won't be deleting any points, but we will be modifying
    // columns and rows. We'll have to store a new set of column vectors.
    std::vector<uword> col_counts(in_cols);
    std::vector<uword> new_row_indices(row_indices);
    for(const_iterator it = begin(); it != end(); it++)
      {
      uword vector_position = (it.col * n_rows) + it.row;
      new_row_indices[it.pos] = vector_position % in_rows;
      ++col_counts[vector_position / in_rows];
      }

    // Now sum the column counts to get the new column pointers.
    for(uword i = 1; i <= in_cols; i++)
      {
      col_ptrs[i] = col_ptrs[i - 1] + col_counts[i - 1];
      }

    // Copy the new row indices.
    row_indices = new_row_indices;

    // Now set the size.
    access::rw(n_rows) = in_rows;
    access::rw(n_cols) = in_cols;
    }
  else
    {
    // Row-wise reshaping.  This is more tedious and we will use a separate sparse matrix to do it.
    SpMat<eT> tmp(in_rows, in_cols);

    for(const_row_iterator it = begin_row(); it != end_row(); it++)
      {
      uword vector_position = (it.row * n_cols) + it.col;

      tmp((vector_position / in_cols), (vector_position % in_cols)) = (*it);
      }

    (*this).operator=(tmp);
    }
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::fill(const eT val)
  {
  arma_extra_debug_sigprint();

  if (val == 0)
    {
    values.clear();
    row_indices.clear();

    access::rw(n_nonzero) = 0;

    col_ptrs.clear();
    col_ptrs.resize(n_cols + 1, /* fill with 0 */ 0);
    }
  else
    {
    values.clear();
    values.resize(n_elem, /* fill with the value */ val);

    access::rw(n_nonzero) = n_elem;

    // Set the row indices properly.
    row_indices.resize(n_elem);
    for(uword elem = 0; elem < n_elem; elem++)
      {
      row_indices[elem] = (elem % n_rows);
      }

    // Set the column pointers correctly.
    for(uword col = 1; col <= n_cols; col++)
      {
      col_ptrs[col] = col_ptrs[col - 1] + n_rows;
      }
    }

  return *this;
  }


template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::zeros()
  {
  arma_extra_debug_sigprint();

  // If everything is zero, clear the matrix.
  values.clear();
  row_indices.clear();

  col_ptrs.clear();
  col_ptrs.resize(n_cols + 1, 0); // All are 0.

  access::rw(n_nonzero) = 0;

  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::zeros(const uword in_elem)
  {
  arma_extra_debug_sigprint();

  // Clear the matrix.
  values.clear();
  row_indices.clear();

  col_ptrs.clear();
  col_ptrs.resize(2, 0);

  // Now resize.
  access::rw(n_cols) = 1;
  access::rw(n_rows) = in_elem;
  access::rw(n_elem) = in_elem;
  access::rw(n_nonzero) = 0;

  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::zeros(const uword in_rows, const uword in_cols)
  {
  arma_extra_debug_sigprint();

  // Clear the matrix.
  values.clear();
  row_indices.clear();

  col_ptrs.clear();
  col_ptrs.resize(in_cols + 1, 0);

  // Now resize.
  access::rw(n_cols) = in_cols;
  access::rw(n_rows) = in_rows;
  access::rw(n_elem) = (in_rows * in_cols);
  access::rw(n_nonzero) = 0;

  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::ones()
  {
  arma_extra_debug_sigprint();

  return fill(eT(1));
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::ones(const uword in_elem)
  {
  arma_extra_debug_sigprint();

  set_size(in_elem, 1);

  return fill(eT(1));
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::ones(const uword in_rows, const uword in_cols)
  {
  arma_extra_debug_sigprint();

  set_size(in_rows, in_cols);

  return fill(eT(1));
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::randu()
  {
  arma_extra_debug_sigprint();

  zeros(); // Clear the matrix.

  for(uword elem = 0; elem < n_elem; ++elem)
    {
    // It's possible that a value could be zero...
    eT rand_val = eT(eop_aux_randu<eT>());
    if (rand_val != 0)
      {
      values.push_back(rand_val);
      row_indices.push_back(elem % n_rows);
      ++col_ptrs[elem / n_rows];
      }
    }

  // Sum all the column counts to make the column pointers.
  for(uword col = 1; col <= n_cols; ++col)
    {
    col_ptrs[col] += col_ptrs[col - 1];
    }

  access::rw(n_nonzero) = col_ptrs[n_cols];

  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::randu(const uword in_elem)
  {
  arma_extra_debug_sigprint();

  zeros(in_elem);

  return (*this).randu();
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::randu(const uword in_rows, const uword in_cols)
  {
  arma_extra_debug_sigprint();

  zeros(in_rows, in_cols);

  return (*this).randu();
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::randn()
  {
  arma_extra_debug_sigprint();

  zeros(); // Clear the matrix.

  for(uword elem = 0; elem < n_elem; ++elem)
    {
    eT rand_val = eT(eop_aux_randn<eT>());
    if(rand_val != 0)
      {
      values.push_back(rand_val);
      row_indices.push_back(elem % n_rows);
      ++col_ptrs[elem / n_rows];
      }
    }

  // Sum all the column counts to make the column pointers.
  for(uword col = 1; col <= n_cols; ++col)
    {
    col_ptrs[col] += col_ptrs[col - 1];
    }

  access::rw(n_nonzero) = col_ptrs[n_cols];

  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::randn(const uword in_elem)
  {
  arma_extra_debug_sigprint();

  zeros(in_elem);

  return (*this).randn();
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::randn(const uword in_rows, const uword in_cols)
  {
  arma_extra_debug_sigprint();

  zeros(in_rows, in_cols);

  return (*this).randn();
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::eye()
  {
  arma_extra_debug_sigprint();

  return eye(n_rows, n_cols);
  }

template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::eye(const uword in_rows, const uword in_cols)
  {
  arma_extra_debug_sigprint();

  const uword N = std::min(in_rows, in_cols);

  set_size(in_rows, in_cols);

  if(n_nonzero != N)
    {
    values.resize(N, 1);
    row_indices.resize(N, 0);
    }

  values.assign(N, 1);

  col_ptrs.back() = N;

  uword i;
  for(i = 0; i < N; ++i)
    row_indices[i] = i;

  for(i = 0; i < N; ++i)
    col_ptrs[i] = i;

  access::rw(n_nonzero) = N;

  return *this;
  }



template<typename eT>
inline
void
SpMat<eT>::reset()
  {
  arma_extra_debug_sigprint();

  set_size(0, 0);
  }



template<typename eT>
inline
eT
SpMat<eT>::min(uword& index_of_min_val) const
{
  arma_extra_debug_sigprint();

  arma_debug_check((n_elem == 0), "min(): object has no elements");

  eT val = 0;

  if (n_nonzero == 0) // There are no other elements.  It must be 0.
    {
    index_of_min_val = 0;
    }
  else
    {
    uword location;
    val = op_min::direct_min(values, n_nonzero, location);

    if ((val > 0) && (n_nonzero < n_elem)) // A sparse 0 is less.
      {
      val = 0;

      // Give back the index to the first zero position.
      index_of_min_val = 0;
      while (get_position(index_of_min_val) == index_of_min_val) // An element exists at that position.
        {
        index_of_min_val++;
        }

      }
    else
      {
      index_of_min_val = get_position(location);
      }
    }

  return val;

  }

template<typename eT>
inline
eT
SpMat<eT>::min(uword& row_of_min_val, uword& col_of_min_val) const
  {
  arma_extra_debug_sigprint();

  arma_debug_check((n_elem == 0), "min(): object has no elements");

  eT val = 0;

  if (n_nonzero == 0) // There are no other elements.  It must be 0.
    {
    row_of_min_val = 0;
    col_of_min_val = 0;
    }
  else
    {
    uword location;
    val = op_min::direct_min(values, n_nonzero, location);

    if ((val > 0) && (n_nonzero < n_elem)) // A sparse 0 is less.
      {
      val = 0;

      location = 0;
      while (get_position(location) == location) // An element exists at that position.
        {
        location++;
        }

      row_of_min_val = location % n_rows;
      col_of_min_val = location / n_rows;
      }
    else
      {
      get_position(location, row_of_min_val, col_of_min_val);
      }
    }

  return val;

  }

template<typename eT>
inline
arma_warn_unused
eT
SpMat<eT>::max() const
  {
  arma_extra_debug_sigprint();

  arma_debug_check((n_elem == 0), "max(): object has no elements");

  if (n_nonzero == 0)
    {
    return 0;
    }

  eT val = op_max::direct_max(values, n_nonzero);

  if ((val < 0) && (n_nonzero < n_elem)) // A sparse 0 is more.
    {
    return 0;
    }

  return val;

  }

template<typename eT>
inline
eT
SpMat<eT>::max(uword& index_of_max_val) const
  {
  arma_extra_debug_sigprint();

  arma_debug_check((n_elem == 0), "max(): object has no elements");

  eT val = 0;

  if (n_nonzero == 0)
    {
    index_of_max_val = 0;
    }
  else
    {
    uword location;
    val = op_max::direct_max(values, n_nonzero, location);

    if ((val < 0) && (n_nonzero < n_elem)) // A sparse 0 is more.
      {
      val = 0;

      location = 0;
      while (get_position(location) == location) // An element exists at that position.
        {
        location++;
        }

      }
    else
      {
      index_of_max_val = get_position(location);
      }

    }

  return val;

  }

template<typename eT>
inline
eT
SpMat<eT>::max(uword& row_of_max_val, uword& col_of_max_val) const
  {
  arma_extra_debug_sigprint();

  arma_debug_check((n_elem == 0), "max(): object has no elements");

  eT val = 0;

  if (n_nonzero == 0)
    {
    row_of_max_val = 0;
    col_of_max_val = 0;
    }
  else
    {
    uword location;
    val = op_max::direct_max(values, n_nonzero, location);

    if ((val < 0) && (n_nonzero < n_elem)) // A sparse 0 is more.
      {
      val = 0;

      location = 0;
      while (get_position(location) == location) // An element exists at that position.
        {
        location++;
        }

      row_of_max_val = location % n_rows;
      col_of_max_val = location / n_rows;

      }
    else
      {
      get_position(location, row_of_max_val, col_of_max_val);
      }

    }

  return val;

  }



/**
 * Swap in_row1 with in_row2.
 */
template<typename eT>
inline
void
SpMat<eT>::swap_rows(const uword in_row1, const uword in_row2)
  {
  arma_extra_debug_sigprint();

  arma_debug_check
    (
    (in_row1 >= n_rows) || (in_row2 >= n_rows),
    "SpMat::swap_rows(): out of bounds"
    );

  // Sanity check.
  if (in_row1 == in_row2)
    {
    return;
    }

  // The easier way to do this, instead of collecting all the elements in one row and then swapping with the other, will be
  // to iterate over each column of the matrix (since we store in column-major format) and then swap the two elements in the two rows at that time.
  // We will try to avoid using the at() call since it is expensive, instead preferring to use an iterator to track our position.
  uword col1 = (in_row1 < in_row2) ? in_row1 : in_row2;
  uword col2 = (in_row1 < in_row2) ? in_row2 : in_row1;

  for (uword col = 0; col < n_cols; col++)
    {
    // If there is nothing in this column we can ignore it.
    if (col_ptrs[col] == col_ptrs[col + 1])
      {
      continue;
      }

    // These will represent the positions of the items themselves.
    uword loc1 = n_nonzero + 1;
    uword loc2 = n_nonzero + 1;

    for (uword search_pos = col_ptrs[col]; search_pos < col_ptrs[col + 1]; search_pos++)
      {
      if (row_indices[search_pos] == col1)
        {
        loc1 = search_pos;
        }

      if (row_indices[search_pos] == col2)
        {
        loc2 = search_pos;
        break; // No need to look any further.
        }
      }

    // There are four cases: we found both elements; we found one element (loc1); we found one element (loc2); we found zero elements.
    // If we found zero elements no work needs to be done and we can continue to the next column.
    if ((loc1 != (n_nonzero + 1)) && (loc2 != (n_nonzero + 1)))
      {
      // This is an easy case: just swap the values.  No index modifying necessary.
      eT tmp = values[loc1];
      values[loc1] = values[loc2];
      values[loc2] = tmp;
      }
    else if (loc1 != (n_nonzero + 1)) // We only found loc1 and not loc2.
      {
      // We need to find the correct place to move our value to.  It will be forward (not backwards) because in_row2 > in_row1.
      // Each iteration of the loop swaps the current value (loc1) with (loc1 + 1); in this manner we move our value down to where it should be.
      while (((loc1 + 1) < col_ptrs[col + 1]) && (row_indices[loc1 + 1] < in_row2))
        {
        // Swap both the values and the indices.  The column should not change.
        eT tmp = values[loc1];
        values[loc1] = values[loc1 + 1];
        values[loc1 + 1] = tmp;

        uword tmp_index = row_indices[loc1];
        row_indices[loc1] = row_indices[loc1 + 1];
        row_indices[loc1 + 1] = tmp_index;

        loc1++; // And increment the counter.
        }

      // Now set the row index correctly.
      row_indices[loc1] = in_row2;

      }
    else if (loc2 != (n_nonzero + 1))
      {
      // We need to find the correct place to move our value to.  It will be backwards (not forwards) because in_row1 < in_row2.
      // Each iteration of the loop swaps the current value (loc2) with (loc2 - 1); in this manner we move our value up to where it should be.
      while (((loc2 - 1) >= col_ptrs[col]) && (row_indices[loc2 - 1] > in_row1))
        {
        // Swap both the values and the indices.  The column should not change.
        eT tmp = values[loc2];
        values[loc2] = values[loc2 - 1];
        values[loc2 - 1] = tmp;

        uword tmp_index = row_indices[loc2];
        row_indices[loc2] = row_indices[loc2 - 1];
        row_indices[loc2 - 1] = tmp_index;

        loc2--; // And decrement the counter.
        }

      // Now set the row index correctly.
      row_indices[loc2] = in_row1;

      }
    /* else: no need to swap anything; both values are zero */
    }
  }

/**
 * Swap in_col1 with in_col2.
 */
template<typename eT>
inline
void
SpMat<eT>::swap_cols(const uword in_col1, const uword in_col2)
  {
  arma_extra_debug_sigprint();

  // slow but works
  for(uword row = 0; row < n_rows; ++row)
    {
    eT tmp = at(row, in_col1);
    at(row, in_col1) = at(row, in_col2);
    at(row, in_col2) = tmp;
    }
/*
  arma_debug_check
    (
    (in_col1 > in_col2) || (in_col2 >= n_cols),
    "SpMat::swap_cols(): out of bounds"
    );

  // Create pointers to the beginning and end of each column for values and row_indices.
  typename std::vector<eT>::iterator v_c1_beg = values.begin() + col_ptrs[in_col1];
  typename std::vector<eT>::iterator v_c1_end = values.begin() + col_ptrs[in_col1 + 1];
  typename std::vector<eT>::iterator v_c2_beg = values.begin() + col_ptrs[in_col2];
  typename std::vector<eT>::iterator v_c2_end = values.begin() + col_ptrs[in_col2 + 1];

  typename std::vector<uword>::iterator r_c1_beg = row_indices.begin() + col_ptrs[in_col1];
  typename std::vector<uword>::iterator r_c1_end = row_indices.begin() + col_ptrs[in_col1 + 1];
  typename std::vector<uword>::iterator r_c2_beg = row_indices.begin() + col_ptrs[in_col2];
  typename std::vector<uword>::iterator r_c2_end = row_indices.begin() + col_ptrs[in_col2 + 1];

  // Calculate the difference in column sizes.
  long diff = (v_c1_end - v_c1_beg) - (v_c2_end - v_c2_beg);

  if (diff == 0) // The columns are the same size, just swap them.
    {
    std::swap_ranges (v_c1_beg, v_c1_end, v_c2_beg);
    std::swap_ranges (r_c1_beg, r_c1_end, r_c2_beg);
    }
  else if (diff > 0) // Column one is larger than column two.
    {
    // Since an in-place swap would be more complex, these hold the extra elements.
    std::vector<eT> vtmp;
    std::vector<uword> rtmp;

    // Make them the correct size.
    vtmp.resize (diff);
    rtmp.resize (diff);

    // And copy the extra elements into them.
    std::copy (v_c1_end - diff, v_c1_end, vtmp.begin());
    std::copy (r_c1_end - diff, r_c1_end, rtmp.begin());

    // Now, we shift all the elements between the columns to the left.
    std::copy (v_c1_end, v_c2_beg, v_c1_end - diff);
    std::copy (r_c1_end, r_c2_beg, r_c1_end - diff);

    // Copy the first elements of the first column into the space made by shifting.
    std::copy (v_c1_beg, v_c1_end - diff, v_c2_beg - diff);
    std::copy (r_c1_beg, r_c1_end - diff, r_c2_beg - diff);

    // Copy the elements in the second column into the first column's space.
    std::copy (v_c2_beg, v_c2_end, v_c1_beg);
    std::copy (r_c2_beg, r_c2_end, r_c1_beg);

    // Last, copy the elements from the temporary variables into the end of the second column's space.
    std::copy (vtmp.begin(), vtmp.end(), v_c2_beg);
    std::copy (rtmp.begin(), rtmp.end(), r_c2_beg);

    // Finally, we adjust the col_ptrs values to account for the difference in sizes.
    std::vector<uword>::iterator col_ptr_beg = col_ptrs.begin() + in_col1 + 1;
    std::vector<uword>::iterator col_ptr_end = col_ptrs.begin() + in_col2 + 1;
    for (; col_ptr_beg < col_ptr_end; ++col_ptr_beg)
      {
      *col_ptr_beg -= diff;
      }
    }
  else // Column two is larger than column one.
    {
    // Since column two was larger, diff is negative, we need to fix that.
    diff *= -1;

    // Since an in-place swap would be more complex, these hold the extra elements.
    std::vector<eT> vtmp;
    std::vector<uword> rtmp;

    // Make them the correct size.
    vtmp.resize (diff);
    rtmp.resize (diff);

    // And copy the extra elements into them.
    std::copy (v_c2_beg, v_c2_beg + diff, vtmp.begin());
    std::copy (r_c2_beg, r_c2_beg + diff, rtmp.begin());

    // Now, we shift all the elements between the columns to the right.
    std::copy_backward (v_c1_end, v_c2_beg, v_c2_beg + diff);
    std::copy_backward (r_c1_end, r_c2_beg, r_c2_beg + diff);

    // Copy the last elements of column two into the space made by shifting.
    std::copy (v_c2_end - diff , v_c2_end, v_c1_end);
    std::copy (r_c2_end - diff , r_c2_end, r_c1_end);

    // Copy the elements in the first column into the second column's space.
    std::copy (v_c1_beg, v_c1_end, v_c2_beg + diff);
    std::copy (r_c1_beg, r_c1_end, r_c2_beg + diff);

    // Last, copy the elements from the temporary variables into the beginning of the first column's space.
    std::copy (vtmp.begin(), vtmp.end(), v_c1_beg);
    std::copy (rtmp.begin(), rtmp.end(), r_c1_beg);

    // Finally, we adjust the col_ptrs values to account for the difference in sizes.
    std::vector<uword>::iterator col_ptr_beg = col_ptrs.begin() + in_col1 + 1;
    std::vector<uword>::iterator col_ptr_end = col_ptrs.begin() + in_col2 + 1;
    for (; col_ptr_beg < col_ptr_end; ++col_ptr_beg)
      {
      *col_ptr_beg += diff;
      }

    }
*/
  }

/**
 * Remove the row row_num.
 */
template<typename eT>
inline
void
SpMat<eT>::shed_row(const uword row_num)
  {
  arma_extra_debug_sigprint();
  arma_debug_check (row_num >= n_rows, "SpMat::shed_row(): out of bounds");

  shed_rows (row_num, row_num);
  }

/**
 * Remove the column col_num.
 */
template<typename eT>
inline
void
SpMat<eT>::shed_col(const uword col_num)
  {
  arma_extra_debug_sigprint();
  arma_debug_check (col_num >= n_cols, "SpMat::shed_col(): out of bounds");

  shed_cols(col_num, col_num);
  }

/**
 * Remove all rows between (and including) in_row1 and in_row2.
 */
template<typename eT>
inline
void
SpMat<eT>::shed_rows(const uword in_row1, const uword in_row2)
  {
  arma_extra_debug_sigprint();

  arma_debug_check
    (
    (in_row1 > in_row2) || (in_row2 >= n_rows),
    "SpMat::shed_cols(): indices out of bounds or incorectly used"
    );

  uword i, j;
  // Store the length of values
  uword vlength = values.size();
  // Store the length of col_ptrs
  uword clength = col_ptrs.size();

  // This is O(n * n_cols) and inplace, there may be a faster way, though.
  for (i = 0, j = 0; i < vlength; ++i, ++j)
    {
    // Store the row of the ith element.
    uword row = row_indices[i];
    // Is the ith element in the range of rows we want to remove?
    if (row >= in_row1 && row <= in_row2)
      {

      // We shift the element we checked to the left by how many elements
      // we have already removed.
      // j=i until we remove the first element, then j < i.
      if (i != j)
        {
        row_indices[j] = row;
        values[j] = values[i];
        }
      // When we find the first element to remove, reduce j by one so that on
      // the next iteration, j < i.
      else 
        --j;

      // Adjust the values of col_ptrs each time we remove an alement.
      // Basically, the length of one column reduces by one, and everything to
      // its right gets reduced by one to represent all the elements being
      // shifted to the left by one.
      for(uword k = 0; k < clength; ++k)
        {
        if (col_ptrs[k] > i)
          {
          --col_ptrs[k];
          }
        }
      }
    }

  // values.size() - (j - 1) is the number of elements removed.
  --j;
  // Shrink the vectors.
  values.erase (values.end() - j, values.end());
  row_indices.erase (row_indices.end() - j, row_indices.end());

  // Adjust row and element counts.
  access::rw(n_rows)    = n_rows - (in_row2 - in_row1) - 1;
  access::rw(n_elem)    = n_rows * n_cols;
  access::rw(n_nonzero) = values.size();
  }

/**
 * Remove all columns between (and including) in_col1 and in_col2.
 */
template<typename eT>
inline
void
SpMat<eT>::shed_cols(const uword in_col1, const uword in_col2)
  {
  arma_extra_debug_sigprint();

  arma_debug_check
    (
    (in_col1 > in_col2) || (in_col2 >= n_cols),
    "SpMat::shed_cols(): indices out of bounds or incorectly used"
    );

  // First we find the locations in values and row_indices for the column entries.
  uword col_beg = col_ptrs[in_col1];
  uword col_end = col_ptrs[in_col2+1];

  // Then we find the number of entries in the column.
  uword diff = col_end - col_beg;

  // Now, we erase the column entries from values and row_indices.
  values.erase (values.begin() + col_beg, values.begin() + col_end);
  row_indices.erase (row_indices.begin() + col_beg, row_indices.begin() + col_end);

  // Last, we erase the column entries from col_ptrs and reduce the values
  // of the other col_ptrs by the number of elements in the removed columns.
  for (uword i = in_col2+1; i < col_ptrs.size(); ++i)
    {
    col_ptrs[i] -= diff;
    }
  col_ptrs.erase (col_ptrs.begin() + in_col1, col_ptrs.begin() + in_col2+1);

  // We update the element and column counts, and we're done.
  access::rw(n_cols)    = n_cols - ((in_col2 - in_col1) + 1);
  access::rw(n_elem)    = n_cols * n_rows;
  access::rw(n_nonzero) = values.size();

  }

/**
 * Initialize the matrix to the specified size.  Data is not preserved, so the matrix is assumed to be entirely sparse (empty).
 */
template<typename eT>
inline
void
SpMat<eT>::init(const uword in_rows, const uword in_cols)
  {
  arma_extra_debug_sigprint();

  // Clean out the existing memory.
  values.clear();
  row_indices.clear();
  col_ptrs.clear();

  // Set the new size accordingly.
  access::rw(n_rows) = in_rows;
  access::rw(n_cols) = in_cols;
  access::rw(n_elem) = (in_rows * in_cols);
  access::rw(n_nonzero) = 0;

  // Try to allocate the column pointers, filling them with 0.
  col_ptrs.resize(n_cols + 1, 0);
  }

/**
 * Initialize the matrix from a string.
 */
template<typename eT>
inline
void
SpMat<eT>::init(const std::string& text)
  {
  arma_extra_debug_sigprint();

  // Figure out the size first.
  uword t_n_rows = 0;
  uword t_n_cols = 0;

  bool t_n_cols_found = false;

  std::string token;

  std::string::size_type line_start = 0;
  std::string::size_type   line_end = 0;

  while (line_start < text.length())
    {

    line_end = text.find(';', line_start);

    if (line_end == std::string::npos)
      line_end = text.length() - 1;

    std::string::size_type line_len = line_end - line_start + 1;
    std::stringstream line_stream(text.substr(line_start, line_len));

    // Step through each column.
    uword line_n_cols = 0;

    while (line_stream >> token)
      {
      ++line_n_cols;
      }

    if (line_n_cols > 0)
      {
      if (t_n_cols_found == false)
        {
        t_n_cols = line_n_cols;
        t_n_cols_found = true;
        }
      else // Check it each time through, just to make sure.
        arma_check((line_n_cols != t_n_cols), "SpMat::init(): inconsistent number of columns in given string");

      ++t_n_rows;
      }

    line_start = line_end + 1;

    }

  set_size(t_n_rows, t_n_cols);

  // Second time through will pick up all the values.
  line_start = 0;
  line_end = 0;

  uword row = 0;

  while (line_start < text.length())
    {

    line_end = text.find(';', line_start);

    if (line_end == std::string::npos)
      line_end = text.length() - 1;

    std::string::size_type line_len = line_end - line_start + 1;
    std::stringstream line_stream(text.substr(line_start, line_len));

    uword col = 0;
    eT val;

    while (line_stream >> val)
      {
      // Only add nonzero elements.
      if (val != 0)
        {
        get_value(row, col) = val;
        }

      ++col;
      }

    ++row;
    line_start = line_end + 1;

    }

  }

/**
 * Copy from another matrix.
 */
template<typename eT>
inline
void
SpMat<eT>::init(const SpMat<eT>& x)
  {
  arma_extra_debug_sigprint();

  // Ensure we are not initializing to ourselves.
  if (this != &x)
    {
    init(x.n_rows, x.n_cols);

    // Now copy over the elements.
    values = x.values;
    row_indices = x.row_indices;
    col_ptrs = x.col_ptrs;
    
    access::rw(n_nonzero) = x.n_nonzero;
    }
  }

/**
 * Return the row of the given element index.
 */
template<typename eT>
arma_inline
arma_warn_unused
SpValProxy<eT>
SpMat<eT>::get_value(const uword i)
  {
  // First convert to the actual location.
  uword col = i / n_rows; // Integer division.
  uword row = i % n_rows;

  return get_value(row, col);
  }

template<typename eT>
arma_inline
arma_warn_unused
eT
SpMat<eT>::get_value(const uword i) const
  {
  // First convert to the actual location.
  uword col = i / n_rows; // Integer division.
  uword row = i % n_rows;

  return get_value(row, col);
  }



template<typename eT>
inline
typename SpMat<eT>::iterator
SpMat<eT>::begin()
  {
  return iterator(*this);
  }



template<typename eT>
inline
typename SpMat<eT>::const_iterator
SpMat<eT>::begin() const
  {
  return const_iterator(*this);
  }



template<typename eT>
inline
typename SpMat<eT>::iterator
SpMat<eT>::end()
  {
  return iterator(*this, n_nonzero);
  }



template<typename eT>
inline
typename SpMat<eT>::const_iterator
SpMat<eT>::end() const
  {
  return const_iterator(*this, n_nonzero);
  }



template<typename eT>
inline
typename SpMat<eT>::iterator
SpMat<eT>::begin_col(const uword col_num)
  {
  return iterator(*this, 0, col_num);
  }



template<typename eT>
inline
typename SpMat<eT>::const_iterator
SpMat<eT>::begin_col(const uword col_num) const
  {
  return const_iterator(*this, 0, col_num);
  }



template<typename eT>
inline
typename SpMat<eT>::iterator
SpMat<eT>::end_col(const uword col_num)
  {
  return iterator(*this, 0, col_num + 1);
  }



template<typename eT>
inline
typename SpMat<eT>::const_iterator
SpMat<eT>::end_col(const uword col_num) const
  {
  return const_iterator(*this, 0, col_num + 1);
  }



template<typename eT>
inline
typename SpMat<eT>::row_iterator
SpMat<eT>::begin_row(const uword row_num)
  {
  return row_iterator(*this, row_num, 0);
  }



template<typename eT>
inline
typename SpMat<eT>::const_row_iterator
SpMat<eT>::begin_row(const uword row_num) const
  {
  return const_row_iterator(*this, row_num, 0);
  }



template<typename eT>
inline
typename SpMat<eT>::row_iterator
SpMat<eT>::end_row()
  {
  return row_iterator(*this, n_nonzero);
  }



template<typename eT>
inline
typename SpMat<eT>::const_row_iterator
SpMat<eT>::end_row() const
  {
  return row_iterator(*this, n_nonzero);
  }



template<typename eT>
inline
typename SpMat<eT>::row_iterator
SpMat<eT>::end_row(const uword row_num)
  {
  return row_iterator(*this, row_num + 1, 0);
  }



template<typename eT>
inline
typename SpMat<eT>::const_row_iterator
SpMat<eT>::end_row(const uword row_num) const
  {
  return const_row_iterator(*this, row_num + 1, 0);
  }



template<typename eT>
inline
void
SpMat<eT>::clear()
  {
  values.clear();
  row_indices.clear();
  col_ptrs.clear();
  col_ptrs.resize(n_cols + 1, 0);

  access::rw(n_nonzero) = 0;
  }



template<typename eT>
inline
bool
SpMat<eT>::empty() const
  {
  return (n_elem == 0);
  }



template<typename eT>
inline
uword
SpMat<eT>::size() const
  {
  return n_elem;
  }

template<typename eT>
arma_warn_unused
SpValProxy<eT>
SpMat<eT>::get_value(const uword in_row, const uword in_col)
  {
  uword colptr = col_ptrs[in_col];
  uword next_colptr = col_ptrs[in_col + 1];

  if (colptr == next_colptr) // Column is empty.
    {
    return SpValProxy<eT>(in_row, in_col, *this); // Proxy for a zero value.
    }

  // Step through the row indices to see if our element exists.
  typename std::vector<eT>::iterator it_val = values.begin() + colptr;
  for (typename std::vector<uword>::iterator it = row_indices.begin() + colptr; it != row_indices.begin() + next_colptr; ++it, ++it_val)
    {
    // First check that we have not stepped past it.
    if (in_row < *it) // If we have, then it doesn't exist: return 0.
      {
      return SpValProxy<eT>(in_row, in_col, *this); // Proxy for a zero value.
      }

    // Now check if we are at the correct place.
    if (in_row == *it) // If we are, return a reference to the value.
      {
      return SpValProxy<eT>(in_row, in_col, *this, &(*it_val));
      }

    }

  // We did not find it, so it does not exist: return 0.
  return SpValProxy<eT>(in_row, in_col, *this);

  }

template<typename eT>
arma_warn_unused
eT
SpMat<eT>::get_value(const uword in_row, const uword in_col) const
  {
  uword colptr = col_ptrs[in_col];
  uword next_colptr = col_ptrs[in_col + 1];

  if (colptr == next_colptr) // Column is empty.  Value must be 0.
    {
    return 0.0;
    }

  // Step through the row indices to see if our element exists.
  typename std::vector<eT>::const_iterator it_val = values.begin() + colptr;
  for (typename std::vector<uword>::const_iterator it = row_indices.begin() + colptr; it != row_indices.begin() + next_colptr; ++it, ++it_val)
    {
    // First check that we have not stepped past it.
    if (in_row < *it) // If we have, then it doesn't exist: return 0.
      {
      return 0.0;
      }

    // Now check if we are at the correct place.
    if (in_row == *it) // If we are, return the value.
      {
      return *it_val;
      }

    }

  // We did not find it, so it does not exist: return 0.
  return 0.0;

  }

/**
 * Given the index representing which of the nonzero values this is, return its
 * actual location, either in row/col or just the index.
 */
template<typename eT>
arma_inline
arma_warn_unused
uword
SpMat<eT>::get_position(const uword i) const
  {
  uword row, col;

  get_position(i, row, col);

  // Assemble the row/col into the element's location in the matrix.
  return (row + n_rows * col);
  }

template<typename eT>
arma_inline
void
SpMat<eT>::get_position(const uword i, uword& row_of_i, uword& col_of_i) const
  {
  arma_debug_check((i >= n_nonzero), "SpMat::get_position(): index out of bounds");

  col_of_i = 0;
  while (col_ptrs[col_of_i + 1] <= i)
    {
    col_of_i++;
    }

  row_of_i = row_indices[i];

  return;
  }

/**
 * Add an element at the given position, and return a reference to it.  The
 * element will be set to 0 (unless otherwise specified).  If the element
 * already exists, its value will be overwritten.
 *
 * @param in_row Row of new element.
 * @param in_col Column of new element.
 * @param in_val Value to set new element to (default 0.0).
 */
template<typename eT>
arma_inline
arma_warn_unused
eT&
SpMat<eT>::add_element(const uword in_row, const uword in_col, const eT val)
  {
  arma_extra_debug_sigprint();

  // We will assume the new element does not exist and begin the search for
  // where to insert it.  If we find that it already exists, we will then
  // overwrite it.
  uword colptr = col_ptrs[in_col];
  uword next_colptr = col_ptrs[in_col + 1];

  uword pos = colptr; // The position in the matrix of this value.

  if (colptr != next_colptr)
    {
    // There are other elements in this column, so we must find where this
    // element will fit as compared to those.
    while (pos < next_colptr && in_row > row_indices[pos])
      {
      pos++;
      }

    // We aren't inserting into the last position, so it is still possible
    // that the element may exist.
    if (pos != next_colptr && row_indices[pos] == in_row)
      {
      // It already exists.  Then, just overwrite it.
      values[pos] = val;

      return values[pos];

      }

    }

  // First, we have to update the rest of the column pointers.
  for (uword i = in_col + 1; i < n_cols + 1; i++)
    {
    access::rw(col_ptrs[i])++; // We are only inserting one new element.
    }

  // Insert the new element correctly.
  values.insert(values.begin() + pos, val);
  row_indices.insert(row_indices.begin() + pos, in_row);
  access::rw(n_nonzero)++; // Add to count of nonzero elements.

  return values[pos];

  }

/**
 * Delete an element at the given position.
 *
 * @param in_row Row of element to be deleted.
 * @param in_col Column of element to be deleted.
 */
template<typename eT>
arma_inline
void
SpMat<eT>::delete_element(const uword in_row, const uword in_col)
  {
  arma_extra_debug_sigprint();

  // We assume the element exists (although... it may not) and look for its
  // exact position.  If it doesn't exist... well, we don't need to do anything.
  uword colptr = col_ptrs[in_col];
  uword next_colptr = col_ptrs[in_col + 1];

  if (colptr != next_colptr)
    {
    // There's at least one element in this column.  Let's see if we are one of
    // them.
    for (uword pos = colptr; pos < next_colptr; pos++)
      {
      if (in_row == row_indices[pos])
        {
        // Found it.  Now remove it.
        values.erase(values.begin() + pos);
        row_indices.erase(row_indices.begin() + pos);
        access::rw(n_nonzero)--; // Remove one from the count of nonzero elements.

        // And lastly, update all the column pointers (decrement by one).
        for (uword i = in_col + 1; i < n_cols + 1; i++)
          {
          access::rw(col_ptrs[i])--; // We only removed one element.
          }

        return; // There is nothing left to do.

        }
      }
    }

  return; // The element does not exist, so there's nothing for us to do.

  }



#ifdef ARMA_EXTRA_SPMAT_MEAT
  #include ARMA_INCFILE_WRAP(ARMA_EXTRA_SPMAT_MEAT)
#endif

//! @}
