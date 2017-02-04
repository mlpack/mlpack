// Copyright (C) 2008-2015 National ICT Australia (NICTA)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
// -------------------------------------------------------------------
//
// Written by Conrad Sanderson - http://conradsanderson.id.au
// Written by Ryan Curtin

// Backport unary minus operator for sparse matrices to Armadillo 4.000 and
// older.

#if (ARMA_VERSION_MAJOR < 4) || \
    (ARMA_VERSION_MAJOR == 4 && ARMA_VERSION_MINOR <= 0)

template<typename T1>
inline
typename
enable_if2
  <
  is_arma_sparse_type<T1>::value && is_signed<typename T1::elem_type>::value,
  SpOp<T1,spop_scalar_times>
  >::result
operator-
(const T1& X)
  {
  arma_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  return SpOp<T1,spop_scalar_times>(X, eT(-1));
  }

#endif
