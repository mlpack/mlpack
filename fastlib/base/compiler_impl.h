// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION


// Implementation of compiler.h

#ifndef BASE_COMPILER_IMPL_H
#define BASE_COMPILER_IMPL_H

#ifndef BASE_COMPILER_H__WANT_COMPILER_IMPL
"No need to include this file directly; include compiler.h"
#endif

#ifdef __cplusplus
#define EXTERN_C_START__impl extern "C" {
#define EXTERN_C_END__impl };
#define COMPILER_CAST__impl(cast, type, var) (cast< type >((var)))
#else
#define EXTERN_C_START__impl 
#define EXTERN_C_END__impl 
#define COMPILER_CAST__impl(cast, type, var) ((type)(var))
#endif

#if defined(__GNUC__) || defined(__INTEL_COMPILER)
#define expect__impl(expr, value) (__builtin_expect((expr), (value)))
#define likely__impl(x) (__builtin_expect(!!(x), 1))
#define unlikely__impl(x) (__builtin_expect(!!(x), 0))
#define COMPILER_NORETURN__IMPL __attribute__((noreturn))
#define COMPILER_PRINTF__IMPL(format_arg, dotdotdot_arg) \
    __attribute__((format(printf, format_arg, dotdotdot_arg)))
#define COMPILER_FUNCTIONAL__IMPL __attribute__((const))
#define COMPILER_NOINLINE__IMPL __attribute__((noinline))
#define IS_CONSTANT_EXPRESSION__impl(x) (__builtin_constant_p(x))
#else
#warning unknown compiler -- not using special compiler optimizations
#define expect__impl(expr, value) (expr)
#define likely__impl(x) (x)
#define unlikely__impl(x) (x)
#define COMPILER_NORETURN__IMPL
#define COMPILER_PRINTF__IMPL(format_arg, dotdotdot_arg)
#define COMPILER_FUNCTIONAL__IMPL
#define COMPILER_NOINLINE__IMPL
#define IS_CONSTANT_EXPRESSION__impl(x) (0)
#endif

#ifdef __cplusplus
template <typename T>
struct base_c__stride {
  struct S { T t; char c; };
  // this is the stride that the compiler will give to it in a struct
  static const int NATURAL_STRIDE = 
      (sizeof(S) > sizeof(T)) ? (sizeof(S) - sizeof(T)) : sizeof(T);
  // this is the power-of-two stride that is more likely to be faster
  static const int PREFERRED_STRIDE =
      sizeof(T) >= 8 ? 8 : sizeof(T) >= 4 ? 4 : 0;
  // we'll take the larger of two
  enum { STRIDE = NATURAL_STRIDE > PREFERRED_STRIDE ?
      NATURAL_STRIDE : PREFERRED_STRIDE };
};
#undef BASE_C__FULLY_ALIGN
#define strideof__impl(T) (base_c__stride<T>::STRIDE)
#else
#define strideof__impl(T) (sizeof(struct {T x;char c;})-sizeof(T))
#endif

#endif
