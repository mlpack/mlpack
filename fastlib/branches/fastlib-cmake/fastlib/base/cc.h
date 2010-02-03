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
 * @file cc.h
 *
 * The bare necessities of FASTlib programming in C++.
 */
#ifndef BASE_CC_H
#define BASE_CC_H

#include "fastlib/base/common.h"

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
 * @see ASSIGN_VIA_RECURSION_SAFE_COPY_CONSTRUCTION
 */
#define ASSIGN_VIA_COPY_CONSTRUCTION(C) \
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
 * @see ASSIGN_VIA_COPY_CONSTRUCTION
 */
#define ASSIGN_VIA_RECURSION_SAFE_COPY_CONSTRUCTION(C) \
    const C &operator=(const C &src) { \
      if (likely(this != &src)) { \
        char buf[sizeof(C)]; \
        memcpy(buf, this, sizeof(C)); \
        new(this) C(src); \
        reinterpret_cast<C *>(buf)->~C(); \
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

/**
 * Applies some type-parameterized macro to all C/C++ primitives.
 *
 * This is handy for defining templated functions with special
 * behavior for primitive types.  The input macro must accept two
 * arguments: the type and the printf format specifier for the type;
 * it does not have to make use of the latter.
 */
#define FOR_ALL_PRIMITIVES_DO(macro) \
    macro(char, "%c") \
    macro(short, "%d") \
    macro(int, "%d") \
    macro(long, "%ld") \
    macro(long long, "%lld") \
    macro(unsigned char, "%u") \
    macro(unsigned short, "%u") \
    macro(unsigned int, "%u") \
    macro(unsigned long, "%lu") \
    macro(unsigned long long, "%llu") \
    macro(float, "%g") \
    macro(double, "%g") \
    macro(long double, "%Lg")

/** Empty object. */
class Empty {};

#endif /* BASE_CC_H */
