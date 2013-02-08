// Extra traits to support u64 and s64 (or, specifically, unsigned long and
// long) until that is applied to the Armadillo sources.

// This isn't necessary if Armadillo was compiled with 64-bit support, or if
// ARMA_USE_U64S64 is enabled, or if Armadillo >= 3.6.2 is used (by default
// Armadillo 3.6.2 allows long types).
#if (ARMA_VERSION_MAJOR < 3) || \
    ((ARMA_VERSION_MAJOR == 3) && (ARMA_VERSION_MINOR < 6)) || \
    ((ARMA_VERSION_MAJOR == 3) && (ARMA_VERSION_MINOR == 6) && \
        (ARMA_VERSION_PATCH < 2))
  #if !(defined(ARMA_64BIT_WORD) || defined(ARMA_USE_U64S64))
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
#endif
