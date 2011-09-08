// Extensions to typedef u64 and s64 until that support is added into
// Armadillo.  We only need to typedef s64 on Armadillo >= 1.2.0
/*
#if   (ARMA_VERSION_MAJOR >= 1) && \
      (ARMA_VERSION_MINOR >= 2) && \
      (ARMA_VERSION_PATCH >= 0)
  // We only need to typedef s64.
  #if   ULONG_MAX >= 0xffffffffffffffff
    typedef          long s64;
  #elif ULLONG_MAX >= 0xffffffffffffffff
    typedef          long s64;
  #else
    #error "don't know how to typedef 's64' on this system"
  #endif
#else
*/
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
//#endif
