// Copyright (C) 2008-2016 National ICT Australia (NICTA)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
// -------------------------------------------------------------------
//
// Written by Conrad Sanderson - http://conradsanderson.id.au
// Written by Ryan Curtin

//! Add a serialization operator.
template<typename Archive>
void serialize(Archive& ar, const unsigned int version);

/**
 * These will help us refer the proper vector / column types, only with
 * specifying the matrix type we want to use.
 */

typedef Col<elem_type>   vec_type;
typedef Col<elem_type>   col_type;
typedef Row<elem_type>   row_type;

/*
 * Add row_col_iterator and row_col_const_iterator to arma::Mat.
 */

/*
 * row_col_iterator for Mat<eT>. This iterator can return row and column index
 * of the entry its pointing too. The functionality of this iterator is similar
 * to sparse matrix iterators.
 */

#if ARMA_VERSION_MAJOR < 4 || \
    (ARMA_VERSION_MAJOR == 4 && ARMA_VERSION_MINOR < 349)
class row_col_iterator;

class const_row_col_iterator
  {
  public:

  // empty constructor
  inline const_row_col_iterator();
  // constructs const iterator from other iterators
  inline const_row_col_iterator(const row_col_iterator& it);
  inline const_row_col_iterator(const const_row_iterator& it);
  inline const_row_col_iterator(const row_iterator& it);
  // constructs iterator with given row and col index
  inline const_row_col_iterator(const Mat<eT>& in_M, const uword row = 0, const uword col = 0);

  /*
   * Returns the value of the current position.
   */
  inline arma_hot const eT& operator*() const { return *current_pos; }

  /*
   * Increment and decrement operators for this iterator.
   */
  inline arma_hot const_row_col_iterator& operator++();
  inline arma_hot const_row_col_iterator  operator++(int);
  inline arma_hot const_row_col_iterator& operator--();
  inline arma_hot const_row_col_iterator  operator--(int);

  /*
   * Comparison operator with itself and other relevant iterators.
   */
  inline arma_hot bool operator==(const const_row_col_iterator& rhs) const;
  inline arma_hot bool operator!=(const const_row_col_iterator& rhs) const;
  inline arma_hot bool operator==(const row_col_iterator& rhs) const;
  inline arma_hot bool operator!=(const row_col_iterator& rhs) const;
  inline arma_hot bool operator==(const const_iterator& rhs) const;
  inline arma_hot bool operator!=(const const_iterator& rhs) const;
  inline arma_hot bool operator==(const iterator& rhs) const;
  inline arma_hot bool operator!=(const iterator& rhs) const;
  inline arma_hot bool operator==(const const_row_iterator& rhs) const;
  inline arma_hot bool operator!=(const const_row_iterator& rhs) const;
  inline arma_hot bool operator==(const row_iterator& rhs) const;
  inline arma_hot bool operator!=(const row_iterator& rhs) const;

  arma_inline uword row() const { return internal_row; }
  arma_inline uword col() const { return internal_col; }

  // So that we satisfy the STL iterator types.
  typedef std::bidirectional_iterator_tag iterator_category;
  typedef eT                              value_type;
  typedef uword                           difference_type; // not certain on this one
  typedef const eT*                       pointer;
  typedef const eT&                       reference;

  arma_aligned const Mat<eT>* M;

  arma_aligned const eT* current_pos;
  arma_aligned       uword  internal_col;
  arma_aligned       uword  internal_row;
  };

class row_col_iterator
  {
  public:

  // empty constructor
  inline row_col_iterator();
  // constructs const iterator from other iterators
  inline row_col_iterator(const row_iterator& it);
  // constructs iterator with given row and col index
  inline row_col_iterator(Mat<eT>& in_M, const uword row = 0, const uword col = 0);

  /*
   * Returns the value of the current position.
   */
  inline arma_hot eT& operator*() const { return *current_pos; }

  /*
   * Increment and decrement operators for this iterator.
   */
  inline arma_hot row_col_iterator& operator++();
  inline arma_hot row_col_iterator  operator++(int);
  inline arma_hot row_col_iterator& operator--();
  inline arma_hot row_col_iterator  operator--(int);

  /*
   * Comparison operator with itself and other relevant iterators.
   */
  inline arma_hot bool operator==(const const_row_col_iterator& rhs) const;
  inline arma_hot bool operator!=(const const_row_col_iterator& rhs) const;
  inline arma_hot bool operator==(const row_col_iterator& rhs) const;
  inline arma_hot bool operator!=(const row_col_iterator& rhs) const;
  inline arma_hot bool operator==(const const_iterator& rhs) const;
  inline arma_hot bool operator!=(const const_iterator& rhs) const;
  inline arma_hot bool operator==(const iterator& rhs) const;
  inline arma_hot bool operator!=(const iterator& rhs) const;
  inline arma_hot bool operator==(const const_row_iterator& rhs) const;
  inline arma_hot bool operator!=(const const_row_iterator& rhs) const;
  inline arma_hot bool operator==(const row_iterator& rhs) const;
  inline arma_hot bool operator!=(const row_iterator& rhs) const;

  arma_inline uword row() const { return internal_row; }
  arma_inline uword col() const { return internal_col; }

  // So that we satisfy the STL iterator types.
  typedef std::bidirectional_iterator_tag iterator_category;
  typedef eT                              value_type;
  typedef uword                           difference_type; // not certain on this one
  typedef const eT*                       pointer;
  typedef const eT&                       reference;

  arma_aligned const Mat<eT>* M;

  arma_aligned       eT* current_pos;
  arma_aligned       uword  internal_col;
  arma_aligned       uword  internal_row;
  };

/*
 * Extra functions for Mat<eT>
 */
// begin for iterator row_col_iterator
inline const_row_col_iterator begin_row_col() const;
inline row_col_iterator begin_row_col();

// end for iterator row_col_iterator
inline const_row_col_iterator end_row_col() const;
inline row_col_iterator end_row_col();
#endif
