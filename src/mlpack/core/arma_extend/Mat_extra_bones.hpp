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