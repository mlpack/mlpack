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
 * Add an external serialization function.
 */
template<typename Archive, typename eT>
void serialize(Archive& ar, arma::SpMat<eT>& SpMat)
{
  // This is accurate from Armadillo 3.6.0 onwards.
  // We can't use CEREAL_NVP() because of the access::rw() call.
  // We need to check on this access::rw
  ar(cereal::make_nvp("n_rows", arma::access::rw(SpMat.n_rows)));
  ar(cereal::make_nvp("n_cols", arma::access::rw(SpMat.n_cols)));
  ar(cereal::make_nvp("n_elem", arma::access::rw(SpMat.n_elem)));
  ar(cereal::make_nvp("n_nonzero", arma::access::rw(SpMat.n_nonzero)));
  ar(cereal::make_nvp("vec_state", arma::access::rw(SpMat.vec_state)));

  // Now we have to serialize the values, row indices, and column pointers.
  // If we are loading, we need to initialize space for these things.
  if (Archive::is_loading::value)
  {
    const arma::uword new_n_nonzero = SpMat.n_nonzero; // Save this; we're about to nuke it.
    SpMat.init(SpMat.n_rows,SpMat.n_cols); // Allocate column pointers.
    SpMat.mem_resize(new_n_nonzero); // Allocate storage.
    // These calls will set the sentinel values at the end of the storamat
    // column pointers, if necessary, so we don't need to worry about them.
  }

  ar(cereal::make_array(arma::access::rwp(SpMat.values), SpMat.n_nonzero));
  ar(cereal::make_array(arma::access::rwp(SpMat.row_indices), SpMat.n_nonzero));
  ar(cereal::make_array(arma::access::rwp(SpMat.col_ptrs), SpMat.n_cols + 1));
}
