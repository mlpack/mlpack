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
 * These will help us refer the proper vector / column types, only with
 * specifying the matrix type we want to use.
 */
typedef SpCol<elem_type>   vec_type;
typedef SpCol<elem_type>   col_type;
typedef SpRow<elem_type>   row_type;
