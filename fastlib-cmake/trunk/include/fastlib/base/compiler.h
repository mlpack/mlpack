/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @file compiler.h
 *
 * Defines compiler-specific directives and optimizations.  Especially
 * useful are branch-prediction macros "likely" and "unlikely", which
 * can dramatically affect your code's running time.
 */

#ifndef BASE_COMPILER_H
#define BASE_COMPILER_H

/**
 * Begin listing C function prototypes.
 *
 * Needed for proper linking in C++.
 */
#ifdef __cplusplus
#define EXTERN_C_BEGIN extern "C" {
#else
#define EXTERN_C_BEGIN
#endif

/**
 * Finish listing C function prototypes.
 *
 * Needed for proper linking in C++.
 */
#ifdef __cplusplus
#define EXTERN_C_END };
#else
#define EXTERN_C_END
#endif

/* Check for availability of compiler optimizations and directives. */
#if !defined(__GNUC__) && !defined(__INTEL_COMPILER)
#warning Unknown compiler; optimizations and directives disabled.
#ifndef NO_COMPILER_DEFS
#define NO_COMPILER_DEFS
#endif
#endif



/**
 * Optimize compilation for an expression having a given value.
 *
 * @param expr the expression to evaluate
 * @param value the expected result of expr
 * @returns the result of expr
 *
 * @see likely, unlikely
 */
#ifndef NO_COMPILER_DEFS
#define expect(expr, value) (__builtin_expect((expr), (value)))
#else
#define expect(expr, value) (expr)
#endif

/**
 * Optimize compilation for a condition being true.
 *
 * Example:
 * @code
 *   if (likely(i < size)) {
 *     ...
 *   }
 * @endcode
 *
 * Note that this normalizes the result of cond to 1 or 0, which is
 * always fine for use with if-statements and loops.
 *
 * @param cond the condition to test
 * @returns the boolean result of cond
 *
 * @see unlikely, expect
 */
#define likely(cond) expect(!!(cond), 1)

/**
 * Optimize compilation for a condition being false.
 *
 * Example:
 * @code
 *   if (unlikely(error != 0)) {
 *     ...
 *   }
 * @endcode
 *
 * Note that this normalizes the result of cond to 1 or 0, which is
 * always fine for use with if-statements and loops.
 *
 * @param cond the condition to test
 * @returns the boolean result of cond
 *
 * @see likely, expect
 */
#define unlikely(cond) expect(!!(cond), 0)

/** Returns 1 if the compiler can prove the expression is constant. */
#ifndef NO_COMPILER_DEFS
#define IS_CONST_EXPR(expr) (__builtin_constant_p(expr))
#else
#define IS_CONST_EXPR(expr) 0
#endif



/**
 * Indicates a function will not return, e.g. will abort the program.
 *
 * Use this to supress potential warning messages about not returning
 * a value in non-void functions.
 *
 * Example:
 * @code
 *   COMPILER_NO_RETURN
 *   void my_abort();
 * @endcode
 */
#ifndef NO_COMPILER_DEFS
#define COMPILER_NO_RETURN __attribute__((noreturn))
#else
#define COMPILER_NO_RETURN
#endif

/**
 * Indicates a function has printf-style arguments.
 *
 * Use this to enable compile-time warning messages for mismatched
 * printf format strings and argument lists.
 *
 * Example:
 * @code
 *   COMPILER_PRINTF(2, 4)
 *   void my_printf(int i, const char *format, int j, ...);
 * @endcode
 *
 * Note that C++ member funtions reserve argument 1 for "this", so the
 * first visible argument is 2.
 *
 * @param format_arg the position of the format string argument
 * @param dotdotdot_arg the position of the ... argument
 */
#ifndef NO_COMPILER_DEFS
#define COMPILER_PRINTF(format_arg, dotdotdot_arg) \
    __attribute__((format(printf, format_arg, dotdotdot_arg)))
#else
#define COMPILER_PRINTF(format_arg, dotdotdot_arg)
#endif

/**
 * Indicates a function has no (visible) side-effects.
 *
 * Use this to allow the compiler to optimize out repeated calls or
 * rearrange calls, and to supress warning messages when multiple
 * functions are called in an expression.  (By definition, C/C++ may
 * evaluate subexpressions and function arguments in any order,
 * causing problems they contain functions with side-effects).
 *
 * WARNING: Due to complier bugs, never use for functions of pointers.
 *
 * Example:
 * @code
 *   COMPILER_FUNCTIONAL
 *   double my_hypot(double x, double y);
 * @endcode
 */
#ifndef NO_COMPILER_DEFS
#define COMPILER_FUNCTIONAL __attribute__((const))
#else
#define COMPILER_FUNCTIONAL
#endif

/**
 * Disables the in-lining of a function.
 *
 * Used to reduce compiled binary size, but rarely worthwhile.
 */
#ifndef NO_COMPILER_DEFS
#define COMPILER_NO_INLINE __attribute__((noinline))
#else
#define COMPILER_NO_INLINE
#endif

/**
 * Denotes a function as deprecated.
 *
 * This will raise warnings when compiling with features that may
 * eventually be removed.
 *
 * @see COMPILER_DEPRECATED_MSG
 */
#if !defined(NO_COMPILER_DEFS) && !defined(NO_DEPRECATION_WARNINGS)
#define COMPILER_DEPRECATED __attribute__((deprecated))
#else
#define COMPILER_DEPRECATED
#endif

/**
 * Denotes a function as deprecated with a warning message.
 *
 * This will eventually raise the provided warning message when
 * compiling, but this feature is not currently available in all
 * compilers, so instead, the behavior is identical to
 * COMPILER_DEPRECATED.
 *
 * @param msg printed when compiling with a deprecated function
 *
 * @see COMPILER_DEPRECATED
 */
#if !defined(NO_COMPILER_DEFS) && !defined(NO_DEPRECATION_WARNINGS)
#define COMPILER_DEPRECATED_MSG(msg) COMPILER_DEPRECATED
#else
#define COMPILER_DEPRECATED_MSG(msg)
#endif



/**
 * Computes the stride (alignment) of a type.
 *
 * C/C++ will preferentially read and write members of this type at
 * memory positions which are multiples of its stride, for instance
 * when inside structs or function argument lists.
 *
 * Input must be a type, not a variable as is possible with sizeof.
 *
 * @param T a type to be measured
 * @returns the stride of T
 *
 * @see stride_align
 */
#ifdef __cplusplus
#define strideof(T) (compiler_strideof<T>::STRIDE)
template <typename T>
struct compiler_strideof {
  struct S {T x; char c;};
  /* The compiler gives this stride in a struct. */
  static const int NATURAL_STRIDE =
      (sizeof(S) > sizeof(T)) ? (sizeof(S) - sizeof(T)) : sizeof(T);
  /* This power-of-two stride is likely to be faster. */
  static const int PREFERRED_STRIDE =
      sizeof(T) >= 8 ? 8 : sizeof(T) >= 4 ? 4 : 0;
  /* Use the larger of the two. */
  enum { STRIDE = NATURAL_STRIDE > PREFERRED_STRIDE
      ? NATURAL_STRIDE : PREFERRED_STRIDE };
};
#else
#define strideof(T) (sizeof(struct {T x; char c;}) - sizeof(T))
#endif

/**
 * Aligns a size_t to a multiple of a type's stride, rounding up.
 *
 * @param num the size_t to be aligned
 * @param T align to the stride of this type
 * @returns the aligned value
 *
 * @see stride_of, stride_align_max
 **/
#define stride_align(num, T) \
    (((size_t)(num) + strideof(T) - 1) / strideof(T) * strideof(T))

/** The maximum stride of any object on the platform. */
#define MAX_STRIDE 16 /* TODO: confirm correct */

/**
 * Aligns a size_t to a multiple of the maximum stride.
 *
 * @param num the number to be aligned
 * @returns the maximally aligned value
 *
 * @see stride_align, stride_of
 */
#define stride_align_max(num) \
    (((size_t)(num) + MAX_STRIDE - 1) & ~(size_t)(MAX_STRIDE - 1))

/* Fill possibly lacking definition for member memory offsets. */
#ifndef offsetof
#define offsetof(S, member) ((size_t)(&((S const *)0)->member))
#endif



/** Performs C++ casts in either C or C++. */
#ifdef __cplusplus
#define COMPILER_CAST(cast, T, val) (cast< T >(val))
#else
#define COMPILER_CAST(cast, T, val) ((T)(val))
#endif

/**
 * Performs the equivalent of static_cast in either C or C++.
 *
 * This is useful if you compile with -Wold-style-cast.
 */
#define STATIC_CAST(T, val) COMPILER_CAST(static_cast, T, val)

/**
 * Performs the equivalent of reinterpret_cast in either C or C++.
 *
 * This is useful if you compile with -Wold-style-cast.
 */
#define REINTERPRET_CAST(T, val) COMPILER_CAST(reinterpret_cast, T, val)



/** Use a constant reference in C++, or constant pointer in C. */
#ifdef __cplusplus
#define CONST_REF const &
#else
#define CONST_REF const *
#endif

/** Use a reference in C++, or pointer in C. */
#ifdef __cplusplus
#define REF &
#else
#define REF *
#endif



#endif /* BASE_COMPILER_H */
