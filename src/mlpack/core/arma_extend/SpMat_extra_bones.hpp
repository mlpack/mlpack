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
template<typename Archive, typename eT>
void serialize(Archive& ar, arma::SpMat<eT>& SpaMat);

/**
 * These will help us refer the proper vector / column types, only with
 * specifying the matrix type we want to use.
 */
// template<typename eT>
// using vec_type = arma::SpCol<eT>;

// template<typename eT>
// using col_type = arma::SpCol<eT>;

// template<typename eT>
// using row_type = arma::SpRow<eT>;
