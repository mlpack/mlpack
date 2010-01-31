#include "gen_range.h"

/** Initializing the empty range set of char type. */
template<>
void GenRange<char>::InitEmptySet() {
  lo = CHAR_MAX;
  hi = -CHAR_MAX;
}

/** Initializing the universal range set of char type. */
template<>
void GenRange<char>::InitUniversalSet() {
  lo = -CHAR_MAX;
  hi = CHAR_MAX;
}

/** Initializing the empty range set of int type. */
template<>
void GenRange<int>::InitEmptySet() {
  lo = INT_MAX;
  hi = -INT_MAX;
}

/** Initializing the universal range set of int type. */
template<>
void GenRange<int>::InitUniversalSet() {
  lo = -INT_MAX;
  hi = INT_MAX;
}

/** Initializing the empty range set of long long int type. */
template<>
void GenRange<long long int>::InitEmptySet() {
  lo = LLONG_MAX;
  hi = -LLONG_MAX;
}

/** Initializing the universal range set of long long int type. */
template<>
void GenRange<long long int>::InitUniversalSet() {
  lo = -LLONG_MAX;
  hi = LLONG_MAX;
}

/** Initializing the empty range set of long type. */
template<>
void GenRange<long>::InitEmptySet() {
  lo = LONG_MAX;
  hi = -LONG_MAX;
}

/** Initializing the universal range set of long type. */
template<>
void GenRange<long>::InitUniversalSet() {
  lo = -LONG_MAX;
  hi = LONG_MAX;
}

/** Initializing the empty range set of signed char type. */
template<>
void GenRange<signed char>::InitEmptySet() {
  lo = SCHAR_MAX;
  hi = -SCHAR_MAX;
}

/** Initializing the universal range set of signed char type. */
template<>
void GenRange<signed char>::InitUniversalSet() {
  lo = -SCHAR_MAX;
  hi = SCHAR_MAX;
}

/** Initializing the empty range set of short int type. */
template<>
void GenRange<short int>::InitEmptySet() {
  lo = SHRT_MAX;
  hi = -SHRT_MAX;
}

/** Initializing the universal range set of short int type. */
template<>
void GenRange<short int>::InitUniversalSet() {
  lo = -SHRT_MAX;
  hi = SHRT_MAX;
}

/** Initializing the empty range set of float type. */
template<>
void GenRange<float>::InitEmptySet() {
  lo = FLT_MAX;
  hi = -FLT_MAX;
}

/** Initializing the universal range set of float type. */
template<>
void GenRange<float>::InitUniversalSet() {
  lo = -FLT_MAX;
  hi = FLT_MAX;
}

/** Initializing the empty range set of double type. */
template<>
void GenRange<double>::InitEmptySet() {
  lo = DBL_MAX;
  hi = -DBL_MAX;
}

/** Initializing the universal range set of double type. */
template<>
void GenRange<double>::InitUniversalSet() {
  lo = -DBL_MAX;
  hi = DBL_MAX;
}
