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
#define IS_CONSTANT_EXPRESSION__impl(x) (__builtin_constant_p(x))
#endif

#ifdef __GNUC__
#define expect__impl(expr, value) (__builtin_expect((expr), (value)))
#define likely__impl(x) (__builtin_expect(!!(x), 1))
#define unlikely__impl(x) (__builtin_expect(!!(x), 0))
#define COMPILER_NORETURN__IMPL __attribute__((noreturn))
#define COMPILER_PRINTF__IMPL(format_arg, dotdotdot_arg) \
    __attribute__((format(printf, format_arg, dotdotdot_arg)))
#define COMPILER_FUNCTIONAL__IMPL __attribute__((const))
#else
#define expect__impl(expr, value) (expr)
#define likely__impl(x) (x)
#define unlikely__impl(x) (x)
#define IS_CONSTANT_EXPRESSION__impl(x) (0)
#define COMPILER_NORETURN__IMPL
#define COMPILER_PRINTF__IMPL(format_arg, dotdotdot_arg)
#define COMPILER_FUNCTIONAL__IMPL
#endif

#ifdef __cplusplus
template <typename T>
struct base_compiler_h__Tchar {
  T t;
  char c;
};
#define strideof__impl(T)                             \
   ((sizeof(base_compiler_h__Tchar<T>) > sizeof(T)) ?            \
   sizeof(base_compiler_h__Tchar<T>)-sizeof(T) : sizeof(T))
#else
#define strideof__impl(T) (sizeof(struct {T x;char c;})-sizeof(T))
#endif

#endif
