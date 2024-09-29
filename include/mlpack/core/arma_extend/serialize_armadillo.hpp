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
  if (cereal::is_loading<Archive>())
  {
    mat.zeros(n_rows, n_cols);
    arma::access::rw(mat.vec_state) = vec_state;
    mat.mem_resize(n_nonzero); // Allocate storage.
    // These calls will set the sentinel values at the end of the stored
    // column pointers, if necessary, so we don't need to worry about them.
  }

  // Serialize the values held in the sparse matrix.
  for (size_t i = 0; i < mat.n_nonzero; ++i)
    ar(cereal::make_nvp("value", arma::access::rw(mat.values[i])));
  for (size_t i = 0; i < mat.n_nonzero; ++i)
    ar(cereal::make_nvp("row_index", arma::access::rw(mat.row_indices[i])));
  for (size_t i = 0; i < mat.n_cols + 1; ++i)
    ar(cereal::make_nvp("col_ptr", arma::access::rw(mat.col_ptrs[i])));
}

// Add an external serialization function for Mat.
template<typename Archive, typename eT>
void serialize(Archive& ar, arma::Mat<eT>& mat)
{
  // This is accurate from Armadillo 3.6.0 onwards.
  arma::uword n_rows = mat.n_rows;
  arma::uword n_cols = mat.n_cols;
  arma::uword vec_state = mat.vec_state;

  ar(CEREAL_NVP(n_rows));
  ar(CEREAL_NVP(n_cols));
  ar(CEREAL_NVP(vec_state));

  if (cereal::is_loading<Archive>())
  {
    mat.set_size(n_rows, n_cols);
    arma::access::rw(mat.vec_state) = vec_state;
  }

  // Directly serialize the contents of the matrix's memory.
  for (size_t i = 0; i < mat.n_elem; ++i)
    ar(cereal::make_nvp("elem", arma::access::rw(mat.mem[i])));
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

  if (cereal::is_loading<Archive>())
    cube.set_size(n_rows, n_cols, n_slices);

  // Directly serialize the contents of the cube's memory.
  for (size_t i = 0; i < cube.n_elem; ++i)
    ar(cereal::make_nvp("elem", arma::access::rw(cube.mem[i])));
}

} // end namespace cereal

#endif // MLPACK_CORE_ARMA_EXTEND_SERIALIZE_ARMADILLO_HPP
