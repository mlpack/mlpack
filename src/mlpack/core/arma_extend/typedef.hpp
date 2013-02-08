// Extensions to typedef u64 and s64 until that support is added into
// Armadillo.  We only need to typedef s64 on Armadillo > 1.2.0.

#if   ((ARMA_VERSION_MAJOR > 1)) || \
      ((ARMA_VERSION_MAJOR == 1) && (ARMA_VERSION_MINOR > 2)) || \
      ((ARMA_VERSION_MAJOR == 1) && (ARMA_VERSION_MINOR == 2) && \
       (ARMA_VERSION_PATCH > 0))
#ifndef ARMA_64BIT_WORD
  // An unincluded header file typedefs u64 for us.
  template<const bool size_t_is_greater_or_equal_to_8_bytes>
  struct deduce_u64
    {
    };

  template<>
  struct deduce_u64<true>
    {
    typedef std::size_t u64;

    static const u64  max   = (sizeof(u64) >= 8) ? 0xFFFFFFFFFFFFFFFF : 0xFFFFFFFF;  // check required for silly compilers
    static const bool trunc = false;
    };

  template<>
  struct deduce_u64<false>
    {
    #if (ULONG_MAX >= 0xFFFFFFFFFFFFFFFF)
      typedef unsigned long u64;
      static const u64  max   = 0xFFFFFFFFFFFFFFFF;
      static const bool trunc = false;
    #elif defined(ULLONG_MAX)
      typedef unsigned long long u64;
      static const u64  max   = 0xFFFFFFFFFFFFFFFF;
      static const bool trunc = false;
    #elif (_MSC_VER >= 1200)
    //#elif (_MSC_VER >= 1310) && defined(_MSC_EXTENSIONS)
      typedef unsigned __int64 u64;
      static const u64  max   = 0xFFFFFFFFFFFFFFFF;
      static const bool trunc = false;
    #else
      #error "don't know how to typedef 'u64' on this system"
    #endif
    };

  typedef deduce_u64<(sizeof(std::size_t) >= 8)>::u64 u64;
#endif

  // We only need to typedef s64.
  #if   ULONG_MAX >= 0xffffffffffffffff
    typedef          long s64;
  #elif ULLONG_MAX >= 0xffffffffffffffff
    typedef          long s64;
  #else
    #error "don't know how to typedef 's64' on this system"
  #endif
#else

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

#endif
