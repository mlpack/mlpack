// Copyright (C) 2008-2016 National ICT Australia (NICTA)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
// -------------------------------------------------------------------
//
// Written by Conrad Sanderson - http://conradsanderson.id.au
// Written by Ryan Curtin

// Add an external serialization function.
template<typename Archive, typename eT>
void serialize(Archive& ar, arma::Mat<eT>& mat)
{
  const arma::uword old_n_elem = mat.n_elem;

  // This is accurate from Armadillo 3.6.0 onwards.
  // We can't use CEREAL_NVP() because of the arma::access::rw() call.
  ar(cereal::make_nvp("n_rows", arma::access::rw(mat.n_rows)));
  ar(cereal::make_nvp("n_cols", arma::access::rw(mat.n_cols)));
  ar(cereal::make_nvp("n_elem", arma::access::rw(mat.n_elem)));
  ar(cereal::make_nvp("vec_state", arma::access::rw(mat.vec_state)));

  // mem_state will always be 0 on load, so we don't need to save it.
  if (Archive::is_loading::value)
  {
    // Don't free if local memory is being used.
    if (mat.mem_state == 0 && mat.mem != NULL && 
            old_n_elem > arma::arma_config::mat_prealloc)
    {
      arma::memory::release(arma::access::rw(mat.mem));
    }

    arma::access::rw(mat.mem_state) = 0;

    // We also need to allocate the memory we're using.
    mat.init_cold();
  }

  ar & cereal::make_array(arma::access::rwp(mat.mem), mat.n_elem);
}
