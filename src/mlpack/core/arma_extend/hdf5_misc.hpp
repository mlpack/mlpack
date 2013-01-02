// To hack in u64/s64 support to Armadillo when it is not compiled with
// ARMA_64BIT_WORD.
#ifdef ARMA_USE_HDF5

template<>
inline
hid_t
get_hdf5_type< long long >()
  {
  return H5Tcopy(H5T_NATIVE_LLONG);
  }

template<>
inline
hid_t
get_hdf5_type< unsigned long long >()
  {
  return H5Tcopy(H5T_NATIVE_ULLONG);
  }

#endif
