// Copyright (C) 2008-2016 National ICT Australia (NICTA)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
// -------------------------------------------------------------------
//
// Written by Conrad Sanderson - http://conradsanderson.id.au
// Written by Ryan Curtin

#ifndef MLPACK_CORE_ARMA_EXTEND_SERIALIZE_ARMADILLO_HPP
#define MLPACK_CORE_ARMA_EXTEND_SERIALIZE_ARMADILLO_HPP

#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/json.hpp>

#include <mlpack/core/cereal/array_wrapper.hpp>

#include <armadillo>

/**
 * Add an external serialization function for SpMat.
 */
namespace cereal {

template<typename Archive, typename eT>
void serialize(Archive& ar, arma::SpMat<eT>& mat)
{
  // This is accurate from Armadillo 3.6.0 onwards.
  // We can't use CEREAL_NVP() because of the access::rw() call.
  // We need to check on this access::rw
  arma::uword n_rows = mat.n_rows;
  arma::uword n_cols = mat.n_cols;
  arma::uword n_nonzero = mat.n_nonzero;
  arma::uword vec_state = mat.vec_state;

  ar(CEREAL_NVP(n_rows));
  ar(CEREAL_NVP(n_cols));
  ar(CEREAL_NVP(n_nonzero));
  ar(CEREAL_NVP(vec_state));

  // Now we have to serialize the values, row indices, and column pointers.
  // If we are loading, we need to initialize space for these things.
  if (Archive::is_loading::value)
  {
    mat.zeros(n_rows, n_cols);
    arma::access::rw(mat.vec_state) = vec_state;
    mat.mem_resize(n_nonzero); // Allocate storage.
    // These calls will set the sentinel values at the end of the stored
    // column pointers, if necessary, so we don't need to worry about them.
  }

  // Manually set the values in the sparse matrix by assigning what we deserialized.
  ar(cereal::make_array(arma::access::rwp(mat.values), mat.n_nonzero));
  ar(cereal::make_array(arma::access::rwp(mat.row_indices), mat.n_nonzero));
  ar(cereal::make_array(arma::access::rwp(mat.col_ptrs), mat.n_cols + 1));
}

// Add an external serialization function for Mat.
template<typename Archive, typename eT>
void serialize(Archive& ar, arma::Mat<eT>& mat)
{
  // This is accurate from Armadillo 3.6.0 onwards.
  // We can't use CEREAL_NVP() because of the arma::access::rw() call.
  arma::uword n_rows = mat.n_rows;
  arma::uword n_cols = mat.n_cols;
  arma::uword vec_state = mat.vec_state;

  ar(CEREAL_NVP(n_rows));
  ar(CEREAL_NVP(n_cols));
  ar(CEREAL_NVP(vec_state));

  if (Archive::is_loading::value)
  {
    mat.set_size(n_rows, n_cols);
    arma::access::rw(mat.vec_state) = vec_state;
  }

  ar & cereal::make_array(arma::access::rwp(mat.mem), mat.n_elem);
}

// Add a serialization function for armadillo Cube
template<typename Archive, typename eT>
void serialize(Archive& ar, arma::Cube<eT>& cube)
{
  // This is accurate from Armadillo 3.6.0 onwards.
  arma::uword n_rows = cube.n_rows;
  arma::uword n_cols = cube.n_cols;
  arma::uword n_slices = cube.n_slices;

  ar(CEREAL_NVP(n_rows));
  ar(CEREAL_NVP(n_cols));
  ar(CEREAL_NVP(n_slices));

  if (Archive::is_loading::value)
  {
    cube.set_size(n_rows, n_cols, n_slices);
  }

  ar & cereal::make_array(arma::access::rwp(cube.mem), cube.n_elem);
}
} // end namespace cereal

#endif //MLPACK_CORE_ARMA_EXTEND_SERIALIZE_ARMADILLO_HPP
