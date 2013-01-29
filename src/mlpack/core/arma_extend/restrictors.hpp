// Modifications to allow u64/s64 in Armadillo when ARMA_64BIT_WORD is not
// defined.  Only required on Armadillo < 3.6.2.
#if (ARMA_VERSION_MAJOR < 3) || \
    ((ARMA_VERSION_MAJOR == 3) && (ARMA_VERSION_MINOR < 6)) || \
    ((ARMA_VERSION_MAJOR == 3) && (ARMA_VERSION_MINOR == 6) && \
        (ARMA_VERSION_PATCH < 2))
  #ifndef ARMA_64BIT_WORD

    template<> struct arma_scalar_only<u64>   { typedef u64   result; };
    template<> struct arma_scalar_only<s64>   { typedef s64   result; };

    template<> struct arma_integral_only<u64> { typedef u64   result; };
    template<> struct arma_integral_only<s64> { typedef s64   result; };

    template<> struct arma_unsigned_integral_only<u64> { typedef u64 result; };

    template<> struct arma_signed_integral_only<s64> { typedef s64 result; };

    template<> struct arma_signed_only<s64> { typedef s64 result; };

  #endif
#endif
