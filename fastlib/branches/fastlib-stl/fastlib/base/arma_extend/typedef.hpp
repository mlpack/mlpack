// Extensions to typedef u64 and s64 until that support is added into Armadillo.

#if   ULONG_MAX >= 0xffffffffffffffff
  typedef unsigned long u64;
  typedef          long s64;
#elif ULLONG_MAX >= 0xffffffffffffffff
  typedef unsigned long long u64;
  typedef          long long s64;
#else
  #error "don't know how to typedef 'u64' on this system"
#endif
