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


//! \addtogroup Proxy
//! @{

template<typename eT>
class Proxy< SpMat<eT> >
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef SpMat<eT>                                stored_type;
  typedef const SpMat<eT>&                         ea_type;

  static const bool prefer_at_accessor = false;
  static const bool has_subview        = false;

#if (ARMA_VERSION_MAJOR >= 3)
  static const bool is_row = false;
  static const bool is_col = false;
#endif

  arma_aligned const SpMat<eT>& Q;

  inline explicit Proxy(const SpMat<eT>& A)
    : Q(A)
    {
    arma_extra_debug_sigprint();
    }

  arma_inline uword get_n_rows() const { return Q.n_rows; }
  arma_inline uword get_n_cols() const { return Q.n_cols; }
  arma_inline uword get_n_elem() const { return Q.n_elem; }

  arma_inline elem_type operator[] (const uword i)                    const { return Q[i];           }
  arma_inline elem_type at         (const uword row, const uword col) const { return Q.at(row, col); }

  arma_inline ea_type get_ea()                   const { return Q;     }
  arma_inline bool    is_alias(const Mat<eT>& X) const { return false; }
  };



template<typename eT>
class Proxy< SpCol<eT> >
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef SpCol<eT>                                stored_type;
  typedef const SpCol<eT>&                         ea_type;

  static const bool prefer_at_accessor = false;
  static const bool has_subview        = false;

#if (ARMA_VERSION_MAJOR >= 3)
  static const bool is_row = false;
  static const bool is_col = true;
#endif

  arma_aligned const SpCol<eT>& Q;

  inline explicit Proxy(const SpCol<eT>& A)
    : Q(A)
    {
    arma_extra_debug_sigprint();
    }

  arma_inline uword get_n_rows() const { return Q.n_rows; }
  arma_inline uword get_n_cols() const { return 1;        }
  arma_inline uword get_n_elem() const { return Q.n_elem; }

  arma_inline elem_type operator[] (const uword i)                    const { return Q[i];           }
  arma_inline elem_type at         (const uword row, const uword col) const { return Q.at(row, col); }

  arma_inline ea_type get_ea()                   const { return Q;     }
  arma_inline bool    is_alias(const Mat<eT>& X) const { return false; }
  };



template<typename eT>
class Proxy< SpRow<eT> >
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef SpRow<eT>                                stored_type;
  typedef const SpRow<eT>&                         ea_type;

  static const bool prefer_at_accessor = false;
  static const bool has_subview        = false;

#if (ARMA_VERSION_MAJOR >= 3)
  static const bool is_row = true;
  static const bool is_col = false;
#endif

  arma_aligned const SpRow<eT>& Q;

  inline explicit Proxy(const SpRow<eT>& A)
    : Q(A)
    {
    arma_extra_debug_sigprint();
    }

  arma_inline uword get_n_rows() const { return 1;        }
  arma_inline uword get_n_cols() const { return Q.n_cols; }
  arma_inline uword get_n_elem() const { return Q.n_elem; }

  arma_inline elem_type operator[] (const uword i)                    const { return Q[i]; }
  arma_inline elem_type at         (const uword row, const uword col) const { return Q.at(row, col); }

  arma_inline ea_type get_ea()                   const { return Q.memptr(); }
  arma_inline bool    is_alias(const Mat<eT>& X) const { return false;      }
  };



template<typename eT>
class Proxy< SpSubview<eT> >
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef SpSubview<eT>                            stored_type;
  typedef const SpSubview<eT>&                     ea_type;

  static const bool prefer_at_accessor = true;
  static const bool has_subview        = true;

#if (ARMA_VERSION_MAJOR >= 3)
  static const bool is_row = false;
  static const bool is_col = false;
#endif

  arma_aligned const SpSubview<eT>& Q;

  inline explicit Proxy(const SpSubview<eT>& A)
    : Q(A)
    {
    arma_extra_debug_sigprint();
    }

  arma_inline uword get_n_rows() const { return Q.n_rows; }
  arma_inline uword get_n_cols() const { return Q.n_cols; }
  arma_inline uword get_n_elem() const { return Q.n_elem; }

  arma_inline elem_type operator[] (const uword i)                    const { return Q[i];           }
  arma_inline elem_type at         (const uword row, const uword col) const { return Q.at(row, col); }

  arma_inline ea_type get_ea()                     const { return Q;              }
  arma_inline bool    is_alias(const SpMat<eT>& X) const { return (&(Q.m) == &X); }
  };

//! @}
