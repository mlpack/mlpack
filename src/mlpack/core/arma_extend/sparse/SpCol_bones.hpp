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


//! \addtogroup SpCol
//! @{

//! Class for sparse column vectors (matrices with only one column)

template<typename eT>
class SpCol : public SpMat<eT>
  {
  public:

  typedef eT                                elem_type;
  typedef typename get_pod_type<eT>::result pod_type;

#if (ARMA_VERSION_MAJOR) >= 3
  static const bool is_row = false;
  static const bool is_col = true;
#endif

  inline          SpCol();
  inline explicit SpCol(const uword n_elem);
  inline          SpCol(const uword in_rows, const uword in_cols);

  inline                  SpCol(const char*        text);
  inline const SpCol& operator=(const char*        text);

  inline                  SpCol(const std::string& text);
  inline const SpCol& operator=(const std::string& text);

  inline const SpCol& operator=(const eT val);

  template<typename T1> inline                  SpCol(const Base<eT,T1>& X);
  template<typename T1> inline const SpCol& operator=(const Base<eT,T1>& X);

  template<typename T1, typename T2>
  inline explicit SpCol(const Base<pod_type,T1>& A, const Base<pod_type,T2>& B);

  template<typename T1> inline                  SpCol(const BaseCube<eT,T1>& X);
  template<typename T1> inline const SpCol& operator=(const BaseCube<eT,T1>& X);

  inline                  SpCol(const subview_cube<eT>& X);
  inline const SpCol& operator=(const subview_cube<eT>& X);

  arma_inline SpValProxy<eT>& row(const uword row_num);
  arma_inline eT              row(const uword row_num) const;

//  arma_inline       subview_col<eT> rows(const uword in_row1, const uword in_row2);
//  arma_inline const subview_col<eT> rows(const uword in_row1, const uword in_row2) const;

//  arma_inline       subview_col<eT> subvec(const uword in_row1, const uword in_row2);
//  arma_inline const subview_col<eT> subvec(const uword in_row1, const uword in_row2) const;

//  arma_inline       subview_col<eT> subvec(const span& row_span);
//  arma_inline const subview_col<eT> subvec(const span& row_span) const;

  inline void shed_row (const uword row_num);
  inline void shed_rows(const uword in_row1, const uword in_row2);

                        inline void insert_rows(const uword row_num, const uword N, const bool set_to_zero = true);
  template<typename T1> inline void insert_rows(const uword row_num, const Base<eT,T1>& X);


  typedef typename SpMat<eT>::iterator       row_iterator;
  typedef typename SpMat<eT>::const_iterator const_row_iterator;

  inline       row_iterator begin_row(const uword row_num);
  inline const_row_iterator begin_row(const uword row_num) const;

  inline       row_iterator end_row  (const uword row_num);
  inline const_row_iterator end_row  (const uword row_num) const;

  #ifdef ARMA_EXTRA_SPCOL_PROTO
    #include ARMA_INCFILE_WRAP(ARMA_EXTRA_SPCOL_PROTO)
  #endif
  };
