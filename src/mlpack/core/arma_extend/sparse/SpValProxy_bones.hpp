// Copyright (C) 2011 Ryan Curtin <ryan@igglybob.com>
//
// This file is part of the Armadillo C++ library.
// It is provided without any warranty of fitness
// for any purpose.  You can redistribute this file
// and/or modify it under the terms of the GNU
// Lesser General Public License (LGPL) as published
// by the Free Software Foundation, either version 3
// of the License or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

//! \addtogroup SpMat
//! @{

/**
 * Sparse value proxy class, meant to prevent 0s from being added to sparse
 * matrices.
 */
template<typename eT>
class SpValProxy
{
  public:

  friend class SpMat<eT>;

  /**
   * Create the sparse value proxy.
   * Otherwise, pass a pointer to a reference of the value.
   */
  SpValProxy(uword row, uword col, SpMat<eT>& in_parent, eT* in_val_ptr = NULL);

  //! For swapping operations.
  arma_inline SpValProxy& operator=(const SpValProxy& rhs);

  //! Overload all of the potential operators.

  //! First, the ones that could modify a value.
  arma_inline SpValProxy& operator=(const eT rhs);
  arma_inline SpValProxy& operator+=(const eT rhs);
  arma_inline SpValProxy& operator-=(const eT rhs);
  arma_inline SpValProxy& operator*=(const eT rhs);
  arma_inline SpValProxy& operator/=(const eT rhs);
  arma_inline SpValProxy& operator%=(const eT rhs);
  arma_inline SpValProxy& operator<<=(const eT rhs);
  arma_inline SpValProxy& operator>>=(const eT rhs);
  arma_inline SpValProxy& operator&=(const eT rhs);
  arma_inline SpValProxy& operator|=(const eT rhs);
  arma_inline SpValProxy& operator^=(const eT rhs);

  arma_inline SpValProxy& operator++();
  arma_inline SpValProxy& operator--();
  arma_inline eT operator++(const int unused);
  arma_inline eT operator--(const int unused);

  //! This will work for any other operations that do not modify a value.
  operator eT() const;

  private:

  // Deletes the element if it is zero.  Does not check if val_ptr == NULL!
  arma_inline void check_zero();

  uword row;
  uword col;
  eT* val_ptr;

  SpMat<eT>& parent; // We will call this object if we need to insert or delete an element.
};
