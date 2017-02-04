// Copyright (C) 2008-2015 National ICT Australia (NICTA)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
// -------------------------------------------------------------------
//
// Written by Conrad Sanderson - http://conradsanderson.id.au
// Written by Ryan Curtin
// Written by Matthew Amidon

/**
 * Add a batch constructor for SpMat, if the version is older than 3.810.0, and
 * also a serialize() function for Armadillo.
 */
template<typename Archive>
void serialize(Archive& ar, const unsigned int version);

/**
 * These will help us refer the proper vector / column types, only with
 * specifying the matrix type we want to use.
 */
typedef SpCol<elem_type>   vec_type;
typedef SpCol<elem_type>   col_type;
typedef SpRow<elem_type>   row_type;

/*
 * Extra functions for SpMat<eT>
 * Adding definition of row_col_iterator to generalize with Mat<eT>::row_col_iterator
 */
#if ARMA_VERSION_MAJOR < 4 || \
    (ARMA_VERSION_MAJOR == 4 && ARMA_VERSION_MINOR < 349)
typedef iterator row_col_iterator;
typedef const_iterator const_row_col_iterator;

// begin for iterator row_col_iterator
inline const_row_col_iterator begin_row_col() const;
inline row_col_iterator begin_row_col();

// end for iterator row_col_iterator
inline const_row_col_iterator end_row_col() const;
inline row_col_iterator end_row_col();
#endif
