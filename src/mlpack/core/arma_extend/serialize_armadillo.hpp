// Copyright (C) 2008-2016 National ICT Australia (NICTA)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
// -------------------------------------------------------------------
//
// Written by Conrad Sanderson - http://conradsanderson.id.au
// Written by Ryan Curtin

#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/json.hpp>

#include <mlpack/core/cereal/array_wrapper.hpp>

#include <armadillo>


/**
 * Add an external serialization function for SpMat.
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

// Add an external serialization function for Mat.
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

// Add a serialization function for armadillo Cube
template<typename Archive, typename eT>
void serialize(Archive& ar, arma::Cube<eT>& cube)
{
  const arma::uword old_n_elem = cube.n_elem;

  // This is accurate from Armadillo 3.6.0 onwards.
  ar(cereal::make_nvp("n_rows",arma::access::rw(cube.n_rows)));
  ar(cereal::make_nvp("n_cols",arma::access::rw(cube.n_cols)));
  ar(cereal::make_nvp("n_elem_slice",arma::access::rw(cube.n_elem_slice)));
  ar(cereal::make_nvp("n_slices",arma::access::rw(cube.n_slices)));
  ar(cereal::make_nvp("n_elem",arma::access::rw(cube.n_elem)));

  // mem_state will always be 0 on load, so we don't need to save it.
  if (Archive::is_loading::value)
  {
    // Clean any mat pointers.
    cube.delete_mat();

    // Don't free if local memory is being used.
    if (cube.mem_state == 0 && cube.mem != NULL && 
            old_n_elem > arma::arma_config::mat_prealloc)
    {
      arma::memory::release(arma::access::rw(cube.mem));
    }

    arma::access::rw(cube.mem_state) = 0;

    // We also need to allocate the memory we're using.
    cube.init_cold();
  }

  ar & cereal::make_array(arma::access::rwp(cube.mem), cube.n_elem);
}
