// Extra promote_type definitions until 64-bit index support is added to
// Armadillo.

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
