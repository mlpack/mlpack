// Extra promote_type definitions until 64-bit index support is added to
// Armadillo.  The syntax was changed for 2.1.91, so we need to be careful about
// how we do that.
#if ((ARMA_VERSION_MAJOR > 2)) || \
    ((ARMA_VERSION_MAJOR == 2) && (ARMA_VERSION_MINOR > 1)) || \
    ((ARMA_VERSION_MAJOR == 2) && (ARMA_VERSION_MINOR == 1) && \
     (ARMA_VERSION_PATCH >= 91))

// The new syntax changed the name of 'promote_type' to 'is_promotable'.  We
// have to update accordingly...
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

#else
// The old syntax used the 'promote_type' struct.  We just define all of these
// for u64 and s64.
template<typename T> struct promote_type<std::complex<T>, s64> : public promote_type_ok { typedef std::complex<T> result; };
template<typename T> struct promote_type<std::complex<T>, u64> : public promote_type_ok { typedef std::complex<T> result; };

template<> struct promote_type<double, s64  > : public promote_type_ok { typedef double result; };
template<> struct promote_type<double, u64  > : public promote_type_ok { typedef double result; };

template<> struct promote_type<float, s64> : public promote_type_ok { typedef float result; };
template<> struct promote_type<float, u64> : public promote_type_ok { typedef float result; };

template<> struct promote_type<s64, u64> : public promote_type_ok { typedef s64 result; };
template<> struct promote_type<s64, s32> : public promote_type_ok { typedef s64 result; };
template<> struct promote_type<s64, u32> : public promote_type_ok { typedef s64 result; };  // float ?
template<> struct promote_type<s64, s16> : public promote_type_ok { typedef s64 result; };
template<> struct promote_type<s64, u16> : public promote_type_ok { typedef s64 result; };
template<> struct promote_type<s64, s8 > : public promote_type_ok { typedef s64 result; };
template<> struct promote_type<s64, u8 > : public promote_type_ok { typedef s64 result; };

template<> struct promote_type<u64, s32> : public promote_type_ok { typedef s64 result; };  // float ?
template<> struct promote_type<u64, u32> : public promote_type_ok { typedef u64 result; };
template<> struct promote_type<u64, s16> : public promote_type_ok { typedef s64 result; };  // float ?
template<> struct promote_type<u64, u16> : public promote_type_ok { typedef u64 result; };
template<> struct promote_type<u64, s8 > : public promote_type_ok { typedef s64 result; };  // float ?
template<> struct promote_type<u64, u8 > : public promote_type_ok { typedef u64 result; };

template<typename T> struct promote_type<s64, std::complex<T> > : public promote_type_ok { typedef std::complex<T> result; };
template<typename T> struct promote_type<u64, std::complex<T> > : public promote_type_ok { typedef std::complex<T> result; };

template<> struct promote_type<s64  , double> : public promote_type_ok { typedef double result; };
template<> struct promote_type<u64  , double> : public promote_type_ok { typedef double result; };

template<> struct promote_type<s64, float> : public promote_type_ok { typedef float result; };
template<> struct promote_type<u64, float> : public promote_type_ok { typedef float result; };

template<> struct promote_type<u64, s64> : public promote_type_ok { typedef s64 result; };  // float ?
template<> struct promote_type<s32, s64> : public promote_type_ok { typedef s64 result; };
template<> struct promote_type<u32, s64> : public promote_type_ok { typedef s64 result; };  // float ?
template<> struct promote_type<s16, s64> : public promote_type_ok { typedef s64 result; };
template<> struct promote_type<u16, s64> : public promote_type_ok { typedef s64 result; };
template<> struct promote_type<s8 , s64> : public promote_type_ok { typedef s64 result; };
template<> struct promote_type<u8 , s64> : public promote_type_ok { typedef s64 result; };

template<> struct promote_type<s32, u64> : public promote_type_ok { typedef s64 result; };  // float ?
template<> struct promote_type<u32, u64> : public promote_type_ok { typedef u64 result; };
template<> struct promote_type<s16, u64> : public promote_type_ok { typedef s64 result; };  // float ?
template<> struct promote_type<u16, u64> : public promote_type_ok { typedef u64 result; };
template<> struct promote_type<s8 , u64> : public promote_type_ok { typedef s64 result; };  // float ?
template<> struct promote_type<u8 , u64> : public promote_type_ok { typedef u64 result; };

#endif
