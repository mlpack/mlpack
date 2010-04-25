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
 * Optimize compilation for an expression having a given value.
 *
 * @param expr the expression to evaluate
 * @param value the expected result of expr
 * @returns the result of expr
 *
 * @see likely, unlikely
 */
#ifdef __GNUC__
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
#ifndef __GNUC__
#define IS_CONST_EXPR(expr) (__builtin_constant_p(expr))
#else
#define IS_CONST_EXPR(expr) 0
#endif

/**
 * Define __attribute__(( )) as nothing on compilers that don't support it.
 */
#ifndef __GNUC__
  #define __attribute__(x)
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
