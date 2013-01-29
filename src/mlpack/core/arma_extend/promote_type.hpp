// Extra promote_type definitions until 64-bit index support is added to
// Armadillo.  These aren't necessary on Armadillo > 3.6.1.
#if (ARMA_VERSION_MAJOR < 3) || \
    ((ARMA_VERSION_MAJOR == 3) && (ARMA_VERSION_MINOR < 6)) || \
    ((ARMA_VERSION_MAJOR == 3) && (ARMA_VERSION_MINOR == 6) && \
        (ARMA_VERSION_PATCH < 2))
  #ifndef ARMA_64BIT_WORD
    template<typename T> struct is_promotable<std::complex<T>, s64> : public is_promotable_ok { typedef std::complex<T> result; };
    template<typename T> struct is_promotable<std::complex<T>, u64> : public is_promotable_ok { typedef std::complex<T> result; };

    template<> struct is_promotable<double, s64> : public is_promotable_ok { typedef double result; };
    template<> struct is_promotable<double, u64> : public is_promotable_ok { typedef double result; };

    template<> struct is_promotable<float, s64> : public is_promotable_ok { typedef float result; };
    template<> struct is_promotable<float, u64> : public is_promotable_ok { typedef float result; };

    template<> struct is_promotable<s64, u64> : public is_promotable_ok { typedef s64 result; };
    template<> struct is_promotable<s64, s32> : public is_promotable_ok { typedef s64 result; };
    template<> struct is_promotable<s64, u32> : public is_promotable_ok { typedef s64 result; };  // float ?
    template<> struct is_promotable<s64, s16> : public is_promotable_ok { typedef s64 result; };
    template<> struct is_promotable<s64, u16> : public is_promotable_ok { typedef s64 result; };
    template<> struct is_promotable<s64, s8 > : public is_promotable_ok { typedef s64 result; };
    template<> struct is_promotable<s64, u8 > : public is_promotable_ok { typedef s64 result; };

    template<> struct is_promotable<u64, u32> : public is_promotable_ok { typedef u64 result; };
    template<> struct is_promotable<u64, u16> : public is_promotable_ok { typedef u64 result; };
    template<> struct is_promotable<u64, u8 > : public is_promotable_ok { typedef u64 result; };

    template<typename T> struct is_promotable<s64, std::complex<T> > : public is_promotable_ok { typedef std::complex<T> result; };
    template<typename T> struct is_promotable<u64, std::complex<T> > : public is_promotable_ok { typedef std::complex<T> result; };

    template<> struct is_promotable<s64, double> : public is_promotable_ok { typedef double result; };
    template<> struct is_promotable<u64, double> : public is_promotable_ok { typedef double result; };

    template<> struct is_promotable<s64, float> : public is_promotable_ok { typedef float result; };
    template<> struct is_promotable<u64, float> : public is_promotable_ok { typedef float result; };

    template<> struct is_promotable<u64, s64> : public is_promotable_ok { typedef s64 result; };  // float ?

    template<> struct is_promotable<u32, s64> : public is_promotable_ok { typedef s64 result; };  // float ?
    template<> struct is_promotable<s16, s64> : public is_promotable_ok { typedef s64 result; };
    template<> struct is_promotable<u16, s64> : public is_promotable_ok { typedef s64 result; };
    template<> struct is_promotable<s8 , s64> : public is_promotable_ok { typedef s64 result; };
    template<> struct is_promotable<u8 , s64> : public is_promotable_ok { typedef s64 result; };

    template<> struct is_promotable<u32, u64> : public is_promotable_ok { typedef u64 result; };
    template<> struct is_promotable<u16, u64> : public is_promotable_ok { typedef u64 result; };
    template<> struct is_promotable<u8 , u64> : public is_promotable_ok { typedef u64 result; };
  #endif
#endif
