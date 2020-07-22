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
 * Add a serialization function.
 */
template<typename eT>
template<typename Archive>
void serialize(Archive& ar, arma::SpMat<eT>& SpaMat)
{
  // This is accurate from Armadillo 3.6.0 onwards.
  // We can't use CEREAL_NVP() because of the access::rw() call.
  // We need to check on this access::rw
  ar(cereal::make_nvp("n_rows", access::rw(SpaMat.n_rows)));
  ar(cereal::make_nvp("n_cols", access::rw(SpaMat.n_cols)));
  ar(cereal::make_nvp("n_elem", access::rw(SpaMat.n_elem)));
  ar(cereal::make_nvp("n_nonzero", access::rw(SpaMat.n_nonzero)));
  ar(cereal::make_nvp("vec_state", access::rw(SpaMat.vec_state)));

  // Now we have to serialize the values, row indices, and column pointers.
  // If we are loading, we need to initialize space for these things.
  if (Archive::is_loading::value)
  {
    const uword new_n_nonzero = SpaMat.n_nonzero; // Save this; we're about to nuke it.
    init(SpaMat.n_rows,SpaMat.n_cols); // Allocate column pointers.
    mem_resize(new_n_nonzero); // Allocate storage.
    // These calls will set the sentinel values at the end of the storage and
    // column pointers, if necessary, so we don't need to worry about them.
  }

  ar(cereal::make_array(access::rwp(SpaMat.values), SpaMat.n_nonzero));
  ar(cereal::make_array(access::rwp(SpaMat.row_indices), SpaMat.n_nonzero));
  ar(cereal::make_array(access::rwp(SpaMat.col_ptrs), SpaMat.n_cols + 1));
}
