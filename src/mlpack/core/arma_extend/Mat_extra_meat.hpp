// Copyright (C) 2008-2016 National ICT Australia (NICTA)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
// -------------------------------------------------------------------
//
// Written by Conrad Sanderson - http://conradsanderson.id.au
// Written by Ryan Curtin

// Add a serialization operator.
template<typename eT>
template<typename Archive>
void Mat<eT>::serialize(Archive& ar, const unsigned int /* version */)
{
  using boost::serialization::make_nvp;
  using boost::serialization::make_array;

  const uword old_n_elem = n_elem;

  // This is accurate from Armadillo 3.6.0 onwards.
  // We can't use BOOST_SERIALIZATION_NVP() because of the access::rw() call.
  ar & make_nvp("n_rows", access::rw(n_rows));
  ar & make_nvp("n_cols", access::rw(n_cols));
  ar & make_nvp("n_elem", access::rw(n_elem));
  ar & make_nvp("vec_state", access::rw(vec_state));

  // mem_state will always be 0 on load, so we don't need to save it.
  if (Archive::is_loading::value)
  {
    // Don't free if local memory is being used.
    if (mem_state == 0 && mem != NULL && old_n_elem > arma_config::mat_prealloc)
    {
      memory::release(access::rw(mem));
    }

    access::rw(mem_state) = 0;

    // We also need to allocate the memory we're using.
    init_cold();
  }

  ar & make_array(access::rwp(mem), n_elem);
}