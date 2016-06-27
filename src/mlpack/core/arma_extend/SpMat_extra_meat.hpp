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
void SpMat<eT>::serialize(Archive& ar, const unsigned int /* version */)
{
  using boost::serialization::make_nvp;
  using boost::serialization::make_array;

  // This is accurate from Armadillo 3.6.0 onwards.
  // We can't use BOOST_SERIALIZATION_NVP() because of the access::rw() call.
  ar & make_nvp("n_rows", access::rw(n_rows));
  ar & make_nvp("n_cols", access::rw(n_cols));
  ar & make_nvp("n_elem", access::rw(n_elem));
  ar & make_nvp("n_nonzero", access::rw(n_nonzero));
  ar & make_nvp("vec_state", access::rw(vec_state));

  // Now we have to serialize the values, row indices, and column pointers.
  // If we are loading, we need to initialize space for these things.
  if (Archive::is_loading::value)
  {
    const uword new_n_nonzero = n_nonzero; // Save this; we're about to nuke it.
    init(n_rows, n_cols); // Allocate column pointers.
    mem_resize(new_n_nonzero); // Allocate storage.
    // These calls will set the sentinel values at the end of the storage and
    // column pointers, if necessary, so we don't need to worry about them.
  }

  ar & make_array(access::rwp(values), n_nonzero);
  ar & make_array(access::rwp(row_indices), n_nonzero);
  ar & make_array(access::rwp(col_ptrs), n_cols + 1);
}

#if ARMA_VERSION_MAJOR < 4 || \
    (ARMA_VERSION_MAJOR == 4 && ARMA_VERSION_MINOR < 349)
template<typename eT>
inline typename SpMat<eT>::const_row_col_iterator
SpMat<eT>::begin_row_col() const
  {
  return begin();
  }



template<typename eT>
inline typename SpMat<eT>::row_col_iterator
SpMat<eT>::begin_row_col()
  {
  return begin();
  }



template<typename eT>
inline typename SpMat<eT>::const_row_col_iterator
SpMat<eT>::end_row_col() const
  {
  return end();
  }



template<typename eT>
inline typename SpMat<eT>::row_col_iterator
SpMat<eT>::end_row_col()
  {
  return end();
  }
#endif
