// Extra traits to support u64 and s64 until that patch is applied to the
// Armadillo sources.

// This isn't necessary if Armadillo was compiled with 64-bit support.
#ifndef ARMA_64BIT_WORD
  template<typename T1>
  struct is_u64
    { static const bool value = false; };

  template<>
  struct is_u64<u64>
    { static const bool value = true; };


  template<typename T1>
  struct is_s64
    { static const bool value = false; };

  template<>
  struct is_s64<s64>
    { static const bool value = true; };

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


  template<>
  struct is_signed<u64>
    {
    static const bool value = false;
    };

#endif
