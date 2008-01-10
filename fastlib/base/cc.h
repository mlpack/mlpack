// Copyright 2007 Georgia Institute of Technology. All rights reserved.
/**
 * @file cc.h
 *
 * The bare necessities of FASTlib programming in C++.
 */
#ifndef BASE_CC_H
#define BASE_CC_H

#include "common.h"

#include <algorithm>
#include <limits>

/** Extract the highly useful min function from the STL. */
using std::min;
/** Extract the highly useful max function from the STL. */
using std::max;

/** NaN value for doubles; check with isnan, not ==. */
extern const double DBL_NAN;
/** NaN value for floats; check with isnanf, not ==. */
extern const float FLT_NAN;
/** Infinity value for doubles; check with isinf, not ==. */
extern const double DBL_INF;
/** Infinity value for floats; check with isinff, not ==. */
extern const float FLT_INF;

/**
 * Disables copy construction and assignment.
 *
 * This will save you needless headaches by preventing accidental
 * copying of objects via forgetting to use references in function
 * arguments, etc.  Instead of odd behavior you will get compiler
 * errors.  The FASTlib style guide recommends disabling copy
 * construction and assignment for classes that do more than store
 * data, defining the Copy method in their lieu if appropriate.
 *
 * Always follow member declaration macros with the appropriate
 * visibility label (public, private, or protected).
 *
 * Example:
 * @code
 *   class MyClass {
 *     FORBID_ACCIDENTAL_COPIES(MyClass);
 *    private:
 *     ...
 *   };
 * @endcode
 */
#define FORBID_ACCIDENTAL_COPIES(C) \
   private: \
    C (const C &src); \
    const C &operator=(const C &src);

/**
 * Defines assignment in terms of copy construction for non-recursive
 * data structures.
 *
 * This is faster than the recursion-safe version for shallow data
 * structures but about the same otherwise; built-in assignment is
 * faster than either.  This is safe when the left-hand does not
 * contain and is not contained by the right-hand side, i.e. when
 * destructing the left does not inadvertantly change the right.
 *
 * Always follow member declaration macros with the appropriate
 * visibility label (public, private, or protected).
 *
 * @see ASSIGN_VIA_RECURSION_SAFE_COPY_CONSTRUCTION
 */
#define ASSIGN_VIA_COPY_CONSTRUCTION(C) \
   public: \
    const C &operator=(const C &src) { \
      if (likely(this != &src)) { \
        this->~C(); \
        new(this) C(src); \
      } \
      return *this; \
    }

/**
 * Defines assignment in terms of copy construction for recursive data
 * structures.
 *
 * This version of assignment works even when the left-hand side
 * contains or is contained by the right-hand side.  The left is
 * replaced by a snapshot of the right, permitting the right to change
 * or be destructed in the process.  Due to an additional memcpy at
 * the first layer, this method is slower than the recursion-unsafe
 * version for shallow data structures but about the same otherwise.
 *
 * Always follow member declaration macros with the appropriate
 * visibility label (public, private, or protected).
 *
 * @see ASSIGN_VIA_COPY_CONSTRUCTION
 */
#define ASSIGN_VIA_RECURSION_SAFE_COPY_CONSTRUCTION(C) \
   public: \
    const C &operator=(const C &src) { \
      if (likely(this != &src)) { \
        char buf[sizeof(C)]; \
        new(buf) C(src); \
        this->~C(); \
        memcpy(this, buf, sizeof(C)); \
      } \
      return *this; \
    }

/**
 * Defines all inequality operators given friended less-than.
 *
 * Example:
 * @code
 *   class MyClass {
 *     ...
 *     friend bool operator<(const MyClass &a, const MyClass &b) {
 *       return a.v < b.v;
 *     }
 *     EXPAND_LESS_THAN(MyClass);
 *     ...
 *   };
 * @endcode
 *
 * @see EXPAND_GREATER_THAN, EXPAND_EQUALS, EXPAND_HETERO_LESS_THAN
 */
#define EXPAND_LESS_THAN(C) \
    friend bool operator>(C const &b, C const &a) {return a < b;} \
    friend bool operator<=(C const &b, C const &a) {return !(a < b);} \
    friend bool operator>=(C const &a, C const &b) {return !(a < b);}

/**
 * Defines all inequality operators given friended greater-than.
 *
 * Example:
 * @code
 *   class MyClass {
 *     ...
 *     friend bool operator>(const MyClass &a, const MyClass &b) {
 *       return a.v > b.v;
 *     }
 *     EXPAND_GREATER_THAN(MyClass);
 *     ...
 *   };
 * @endcode
 *
 * @see EXPAND_LESS_THAN, EXPAND_EQUALS, EXPAND_HETERO_GREATER_THAN
 */
#define EXPAND_GREATER_THAN(C) \
    friend bool operator<(C const &b, C const &a) {return a > b;} \
    friend bool operator>=(C const &b, C const &a) {return !(a > b);} \
    friend bool operator<=(C const &a, C const &b) {return !(a > b);}

/**
 * Defines not-equals given friended equals.
 *
 * Example:
 * @code
 *   class MyClass {
 *     ...
 *     friend bool operator==(const MyClass &a, const MyClass &b) {
 *       return a.v == b.v;
 *     }
 *     EXPAND_EQUALS(MyClass);
 *     ...
 *   };
 * @endcode
 *
 * @see EXPAND_LESS_THAN, EXPAND_GREATER_THAN, EXPAND_HETERO_EQUALS
 */
#define EXPAND_EQUALS(C) \
    friend bool operator!=(C const &a, C const &b) {return !(a == b);}

/**
 * Defines several heterogeneous inequality operators given friended
 * less-than with the first class on the left.
 *
 * Use twice, exchanging argument order, to obtain all heterogeneous
 * inequalities or use in conjunction with EXPAND_HETERO_GREATER_THAN.
 *
 * Example:
 * @code
 *   class MyClass {
 *     ...
 *     friend bool operator<(const MyClass &a, const int &b) {
 *       return a.v < b;
 *     }
 *     EXPAND_HETERO_LESS_THAN(MyClass, int);
 *     friend bool operator<(const int &a, const MyClass &b) {
 *       return a < b.v;
 *     }
 *     EXPAND_HETERO_LESS_THAN(int, MyClass);
 *     ...
 *   };
 * @endcode
 *
 * @see EXPAND_HETERO_GREATER_THAN, EXPAND_HETERO_EQUALS
 */
#define EXPAND_HETERO_LESS_THAN(C, T) \
    friend bool operator>(T const &b, C const &a) {return a < b;} \
    friend bool operator<=(T const &b, C const &a) {return !(a < b);} \
    friend bool operator>=(C const &a, T const &b) {return !(a < b);}

/**
 * Defines several heterogeneous inequality operators given friended
 * greater-than with the first class on the left.
 *
 * Use twice, exchanging argument order, to obtain all heterogeneous
 * inequalities or use in conjunction with EXPAND_HETERO_LESS_THAN.
 *
 * Example:
 * @code
 *   class MyClass {
 *     ...
 *     friend bool operator<(const MyClass &a, const int &b) {
 *       return a.v < b;
 *     }
 *     EXPAND_HETERO_LESS_THAN(MyClass, int);
 *     friend bool operator>(const MyClass &a, const int &b) {
 *       return a.v > b;
 *     }
 *     EXPAND_HETERO_GREATER_THAN(MyClass, int);
 *     ...
 *   };
 * @endcode
 *
 * @see EXPAND_HETERO_LESS_THAN, EXPAND_HETERO_EQUALS
 */
#define EXPAND_HETERO_GREATER_THAN(C, T) \
    friend bool operator<(T const &b, C const &a) {return a > b;} \
    friend bool operator>=(T const &b, C const &a) {return !(a > b);} \
    friend bool operator<=(C const &a, T const &b) {return !(a > b);}

/**
 * Defines heterogeneus equals with arguments exchanged and both
 * not-equals given friended equals with the first class on the left.
 *
 * Example:
 * @code
 *   class MyClass {
 *     ...
 *     friend bool operator==(const MyClass &a, const int &b) {
 *       return a.v == b;
 *     }
 *     EXPAND_HETERO_EQUALS(MyClass, int);
 *     ...
 *   };
 * @endcode
 *
 * @see EXPAND_HETERO_LESS_THAN, EXPAND_HETERO_GREATER_THAN
 */
#define EXPAND_HETERO_EQUALS(C, T) \
    friend bool operator==(T const &b, C const &a) {return a == b;} \
    friend bool operator!=(C const &a, T const &b) {return !(a == b);} \
    friend bool operator!=(T const &b, C const &a) {return !(a == b);}

/** Empty object. */
class Empty {};

#endif /* BASE_CC_H */
