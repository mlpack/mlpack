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
