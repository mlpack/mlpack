// To hack in u64/s64 support to Armadillo when it is not compiled with
// ARMA_64BIT_WORD.
namespace hdf5_misc {

#if defined(ARMA_USE_HDF5)
  #if !(defined(ARMA_64BIT_WORD) || defined(ARMA_USE_U64S64))
    #if defined(ULLONG_MAX)
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
  #endif
#endif

} // namespace hdf5_misc
