// Extensions to typedef u64 and s64 until that support is added into
// Armadillo.  We only need to typedef s64 on Armadillo > 1.2.0.  This is not
// necessary for Armadillo > 3.6.1.
#if (ARMA_VERSION_MAJOR < 3) || \
    ((ARMA_VERSION_MAJOR == 3) && (ARMA_VERSION_MINOR < 6)) || \
    ((ARMA_VERSION_MAJOR == 3) && (ARMA_VERSION_MINOR == 6) && \
        (ARMA_VERSION_PATCH < 2))
  #ifndef ARMA_64BIT_WORD
    // We must typedef both u64 and s64.
    #if   ULONG_MAX >= 0xffffffffffffffff
      typedef unsigned long u64;
      typedef          long s64;
    #elif ULLONG_MAX >= 0xffffffffffffffff
      typedef unsigned long long u64;
      typedef          long long s64;
    #else
      #error "don't know how to typedef 'u64' on this system"
    #endif

    namespace junk
      {
      struct arma_64_elem_size_test
        {
        arma_static_check( (sizeof(u64) != 8), ERROR___TYPE_U64_HAS_UNSUPPORTED_SIZE );
        arma_static_check( (sizeof(s64) != 8), ERROR___TYPE_S64_HAS_UNSUPPORTED_SIZE );
        };
      }

  #endif
#endif
