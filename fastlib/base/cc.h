// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file cc.h
 *
 * Bare necessities for C++ in FASTlib.
 *
 * This includes common.h plus some macros to help you create classes.
 */
#ifndef BASE_CC_H
#define BASE_CC_H

#include "compiler.h"
#include "debug.h"
#include "scale.h"

#include <cstdlib>

/** NaN value for doubles.  Use isnan to check for this. */
extern const double DBL_NAN;

/** NaN value for floats.  Use isnanf to check for this. */
extern const double FLT_NAN;

/* TODO: Decide if templated min and max would prefer const refs.
 *
 * I'm purposely not making these const refs for the case of integers
 * and such, where you don't want to invalidate putting something in a
 * register.
 *
 * This might be desirable, except in the case where a and b are large
 * structs, but as far as I know, C++ will use constant references when
 * possible.
 */

/**
 * Return the minimum of two objects.
 */
template<typename T, typename Q>
inline T min(T a, Q b) {
  return a < b ? a : b;
}

/**
 * Return the maximum of two objects.
 */
template<typename T, typename Q>
inline T max(T a, Q b) {
  return a > b ? a : b;
}

/**
 * Prevents your class from being copied.
 *
 * This avoids needless headaches and errors by avoiding the copy constructor.
 *
 * From the FASTlib style guide, most classes shouldn't be copiable unless
 * it is only a data structure.
 *
 * Example:
 *
 * @code
 *  class Stuff {
 *    FORBID_COPY(Stuff);
 *   private:
 *    ...
 *  }
 * @endcode
 *
 * After calling this macro, explicitly define the visibility of whatever
 * follows (public, private, protected).
 */
#define FORBID_COPY(cl) \
        private: cl (const cl&); void operator=(const cl&)

/**
 * If you *must* define a copy constructor, this will automatically
 * make a suitable assignment operator.
 */
#define CC_ASSIGNMENT_OPERATOR(cl) \
        public: const cl&operator=(const cl&o) \
        {if(this!=&o){this->~cl();new(this)cl(o);}return *this;}

/**
 * Creates a copy constructor using the Copy method.
 *
 * This does not call the default constructor, so if you are explicitly
 * checking for debug poisons, your checks will fail.
 */
#define ALLOW_COPY(cl) \
        public: cl(const cl& other) { Copy(other); } \
        CC_ASSIGNMENT_OPERATOR(cl)

/**
 * Defines inequality comparators for this class, given the friend
 * operator less-than has already been defined.
 *
 * Example:
 *
 * @code
 *  class Stuff {
 *    friend bool operator < (const Stuff& a, const Stuff& b) {
 *       return a.x < b.x;
 *    }
 *    DEFINE_INEQUALITY_COMPARATORS(Stuff);
 *    ...
 *  }
 * @endcode
 *
 * The Stuff class will then have all inequality operators.
 *
 * @param cl the name of the class
 */
#define DEFINE_INEQUALITY_COMPARATORS(cl) \
        friend bool operator > (cl const & a, cl const & b) { return b < a; } \
        friend bool operator >= (cl const & a, cl const & b) { return !(a < b); } \
        friend bool operator <= (cl const & a, cl const & b) { return !(b < a); }

/**
 * Defines equality and inequality comparators for a class, given the
 * operators less-than and double-equals have already been defined.
 *
 * Example:
 *
 * @code
 *  class Stuff {
 *    friend bool operator < (const Stuff& a, const Stuff& b) {
 *       return a.x < b.x;
 *    }
 *    friend bool operator == (const Stuff& a, const Stuff& b) {
 *       return a.x == b.x;
 *    }
 *    DEFINE_ALL_COMPARATORS(Stuff);
 *    ...
 *  }
 * @endcode
 *
 * The Stuff class will then have all inequality operators.
 *
 * @param cl the name of the class
 */
#define DEFINE_ALL_COMPARATORS(cl) \
        DEFINE_INEQUALITY_COMPARATORS(cl) \
        friend bool operator != (cl const & a, cl const & b) { return !(a == b); }

/**
 * Defines heterogeneous inequality comparators for this class, given the
 * friend operators less-than for both classes has already been defined.
 *
 * Example:
 *
 * @code
 *  class Foo {
 *    friend bool operator < (const Foo& a, const Bar& b) {
 *       ...
 *    }
 *    friend bool operator < (const Bar& a, const Foo& b) {
 *       ...
 *    }
 *    DEFINE_INEQUALITY_COMPARATORS_HETERO(Stuff);
 *    ...
 *  }
 * @endcode
 *
 * The Stuff class will then have all inequality operators.
 *
 * @param cl the name of the class
 */
#define DEFINE_INEQUALITY_COMPARATORS_HETERO(cl1, cl2) \
        friend bool operator > (cl1 const & a, cl2 const & b) { return b < a; } \
        friend bool operator >= (cl1 const & a, cl2 const & b) { return !(a < b); } \
        friend bool operator <= (cl1 const & a, cl2 const & b) { return !(b < a); } \
        friend bool operator > (cl2 const & a, cl1 const & b) { return b < a; } \
        friend bool operator >= (cl2 const & a, cl1 const & b) { return !(a < b); } \
        friend bool operator <= (cl2 const & a, cl1 const & b) { return !(b < a); }

/**
 * Defines equality and inequality comparators for a class, given just
 * a few heterogeneous comparators.
 *
 * Example:
 *
 * @code
 *  class Foo {
 *    friend bool operator < (const Foo& a, const Bar& b) {
 *       ...
 *    }
 *    friend bool operator < (const Bar& a, const Foo& b) {
 *       ...
 *    }
 *    friend bool operator == (const Foo& a, const Bar& b) {
 *       ...
 *    }
 *    DEFINE_ALL_COMPARATORS_HETERO(Foo, Bar);
 *    ...
 *  }
 * @endcode
 *
 * The operators for both will be filled out.
 *
 * @param cl the name of the class
 */
#define DEFINE_ALL_COMPARATORS_HETERO(cl1, cl2) \
        DEFINE_INEQUALITY_COMPARATORS_HETERO(cl1, cl2) \
        friend bool operator == (cl2 const & b, cl1 const & a)\
          {return (a == b);} \
        friend bool operator != (cl1 const & a, cl2 const & b)\
          {return !(a == b);} \
        friend bool operator != (cl2 const & b, cl1 const & a)\
          {return !(a == b);}

/**
 * Empty object.
 */
struct Empty {};

/**
 * IfThenElse condition for template metaprogramming.  A google search will
 * explain how to use this.
 */
template<bool condition, class TA, class TB>
class IfThenElse;

template<class TA, class TB>
class IfThenElse<true, TA, TB> {
 private:
  typedef TA Result;
};

template<class TA, class TB>
class IfThenElse<false, TA, TB> {
 private:
  typedef TB Result;
};

#endif /* BASE_CC_H */
