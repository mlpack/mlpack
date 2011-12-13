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

//! \addtogroup SpMat
//! @{

//! SpValProxy implementation.
template<typename eT>
SpValProxy<eT>::SpValProxy(uword in_row, uword in_col, SpMat<eT>& in_parent, eT* in_val_ptr)
  : row(in_row)
  , col(in_col)
  , val_ptr(in_val_ptr)
  , parent(in_parent)
  {
  // Nothing to do.
  }

template<typename eT>
arma_inline
SpValProxy<eT>&
SpValProxy<eT>::operator=(const SpValProxy<eT>& rhs)
  {
  return (*this).operator=(eT(rhs));
  }

template<typename eT>
arma_inline
SpValProxy<eT>&
SpValProxy<eT>::operator=(const eT rhs)
  {
  if (rhs != 0) // A nonzero element is being assigned.
    {

    if (val_ptr)
      {
      // The value exists and merely needs to be updated.
      *val_ptr = rhs;
      }

    else
      {
      // The value is nonzero and must be added.
      val_ptr = &parent.add_element(row, col, rhs);
      }

    }
  else // A zero is being assigned.~
    {

    if (val_ptr)
      {
      // The element exists, but we need to remove it, because it is being set to 0.
      parent.delete_element(row, col);
      val_ptr = NULL;
      }

    // If the element does not exist, we do not need to do anything at all.

    }

  return *this;

  }

template<typename eT>
arma_inline
SpValProxy<eT>&
SpValProxy<eT>::operator+=(const eT rhs)
  {
  if (rhs != 0)
    {

    if (val_ptr)
      {
      // The value already exists and merely needs to be updated.
      *val_ptr += rhs;
      check_zero();
      }

    else
      {
      // The value does not exist and must be added.
      val_ptr = &parent.add_element(row, col, rhs);
      }

    }

    return *this;

  }

template<typename eT>
arma_inline
SpValProxy<eT>&
SpValProxy<eT>::operator-=(const eT rhs)
  {
  if (rhs != 0)
    {

    if (val_ptr)
      {
      // The value already exists and merely needs to be updated.
      *val_ptr -= rhs;
      check_zero();
      }

    else
      {
      // The value does not exist and must be added.
      val_ptr = &parent.add_element(row, col, -rhs);
      }

    }

    return *this;

  }

template<typename eT>
arma_inline
SpValProxy<eT>&
SpValProxy<eT>::operator*=(const eT rhs)
  {
  if (rhs != 0)
    {

    if (val_ptr)
      {
      // The value already exists and merely needs to be updated.
      *val_ptr *= rhs;
      check_zero();
      }

    }
  else
    {

    if (val_ptr)
      {
      // Since we are multiplying by zero, the value can be deleted.
      parent.delete_element(row, col);
      val_ptr = NULL;
      }

    }

  return *this;

  }

template<typename eT>
arma_inline
SpValProxy<eT>&
SpValProxy<eT>::operator/=(const eT rhs)
  {
  if (rhs != 0) // I hope this is true!
    {

    if (val_ptr)
      {
      *val_ptr /= rhs;
      check_zero();
      }

    }
  else
    {

    if (val_ptr)
      {
      *val_ptr /= rhs; // That is where it gets ugly.
      // Now check if it's 0.
      if (*val_ptr == 0)
        {
        parent.delete_element(row, col);
        val_ptr = NULL;
        }
      }

    else
      {
      eT val = 0 / rhs; // This may vary depending on type and implementation.

      if (val != 0)
        {
        // Ok, now we have to add it.
        val_ptr = &parent.add_element(row, col, val);
        }

      }
    }

  return *this;

  }

template<typename eT>
arma_inline
SpValProxy<eT>&
SpValProxy<eT>::operator%=(const eT rhs)
  {
  if (val_ptr)
    {
    *val_ptr %= rhs;
    check_zero();
    }

  return *this;

  }

template<typename eT>
arma_inline
SpValProxy<eT>&
SpValProxy<eT>::operator<<=(const eT rhs)
  {
  // Shifting 0 by any amount should do nothing.
  if (val_ptr)
    {
    *val_ptr <<= rhs;
    check_zero();
    }

  return *this;

  }

template<typename eT>
arma_inline
SpValProxy<eT>&
SpValProxy<eT>::operator>>=(const eT rhs)
  {
  // Shifting 0 by any amount should do nothing.
  if (val_ptr)
    {
    *val_ptr <<= rhs;
    check_zero();
    }

  return *this;

  }

template<typename eT>
arma_inline
SpValProxy<eT>&
SpValProxy<eT>::operator&=(const eT rhs)
  {
  // Anding with 0 should do nothing.
  if (val_ptr)
    {
    *val_ptr &= rhs;
    check_zero();
    }

  return *this;

  }

template<typename eT>
arma_inline
SpValProxy<eT>&
SpValProxy<eT>::operator|=(const eT rhs)
  {
  // Oring with 0 means we assign rhs.
  if (rhs != 0)
    {

    if (val_ptr)
      {
      *val_ptr |= rhs;
      }
    else
      {
      val_ptr = &parent.add_element(row, col, rhs);
      }

    }

  return *this;

  }

template<typename eT>
arma_inline
SpValProxy<eT>&
SpValProxy<eT>::operator^=(const eT rhs)
  {
  // Xoring 0 with 0 is still 0.
  if (rhs != 0)
    {

    if (val_ptr)
      {
      *val_ptr ^= rhs;
      check_zero();
      }
    else
      {
      val_ptr = &parent.add_element(row, col, rhs ^ 0);
      }

    }
  else
    {

    if (val_ptr)
      {
      *val_ptr ^= rhs;
      check_zero();
      }

    }

  return *this;

  }

template<typename eT>
arma_inline
SpValProxy<eT>&
SpValProxy<eT>::operator++()
  {
  if (val_ptr)
    {
    ++(*val_ptr);
    check_zero();
    }

  else
    {
    val_ptr = &parent.add_element(row, col, 1);
    }

  return *this;

  }

template<typename eT>
arma_inline
SpValProxy<eT>&
SpValProxy<eT>::operator--()
  {
  if (val_ptr)
    {
    --(*val_ptr);
    check_zero();
    }

  else
    {
    val_ptr = &parent.add_element(row, col, -1);
    }

  return *this;

  }

template<typename eT>
arma_inline
eT
SpValProxy<eT>::operator++(const int unused)
  {
  if (val_ptr)
    {
    ++(*val_ptr);
    check_zero();
    }

  else
    {
    val_ptr = &parent.add_element(row, col, 1);
    }

  if (val_ptr) // It may have changed to now be 0.
    {
    return *(val_ptr - 1);
    }
  else
    {
    return 0;
    }

  }

template<typename eT>
arma_inline
eT
SpValProxy<eT>::operator--(const int unused)
  {
  if (val_ptr)
    {
    --(*val_ptr);
    check_zero();
    }

  else
    {
    val_ptr = &parent.add_element(row, col, -1);
    }

  if (val_ptr) // It may have changed to now be 0.
    {
    return *(val_ptr + 1);
    }
  else
    {
    return 0;
    }

  }

template<typename eT>
arma_inline
SpValProxy<eT>::operator eT() const
  {
  if (val_ptr)
    {
    return *val_ptr;
    }
  else
    {
    return eT(0);
    }
  }

template<typename eT>
arma_inline
void
SpValProxy<eT>::check_zero()
  {
  if (*val_ptr == 0)
    {
    parent.delete_element(row, col);
    val_ptr = NULL;
    }
  }
