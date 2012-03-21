// Extra traits to support u64 and s64 until that patch is applied to the
// Armadillo sources.

#if ARMA_VERSION_MAJOR < 1 || \
    (ARMA_VERSION_MAJOR == 1 && ARMA_VERSION_MINOR <= 2)
// For old Armadillo versions ( <= 1.2.0 ), all we have to do is define these
// two structs which say these element types are supported.
template<> struct isnt_supported_elem_type< u64 >                  : public isnt_supported_elem_type_false {};
template<> struct isnt_supported_elem_type< s64 >                  : public isnt_supported_elem_type_false {};

#else
// For new Armadillo versions ( > 1.2.0 ) we have to get a little bit more
// tricky.  We will overload the values for the is_supported_elem_type
// structure, allowing us to redefine it to report success for u64s and s64s.

// This isn't necessary if Armadillo was compiled with 64-bit support.
#ifndef ARMA_64BIT_WORD
template<>
struct is_supported_elem_type<u64>
  {
  static const bool value = true;
  };

template<>
struct is_supported_elem_type<s64>
  {
  static const bool value = true;
  };
#endif

#endif
