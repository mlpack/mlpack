/**
 * @file cc.cc
 *
 * Implementations for bare-necessities FASTlib programming in C++.
 */

#include "cc.h"
#include "debug.h"

const double DBL_NAN = std::numeric_limits<double>::quiet_NaN();
const float FLT_NAN = std::numeric_limits<float>::quiet_NaN();
const double DBL_INF = std::numeric_limits<double>::infinity();
const float FLT_INF = std::numeric_limits<float>::infinity();

#if defined(DEBUG) || defined(PROFILE)

namespace cc__private {
  /** Hidden class that emits messages for debug and profile modes. */
  class InformDebug {
   public:
    InformDebug() {
      PROFILE_ONLY(NOTIFY_STAR("Profiling information available with:\n"));
      PROFILE_ONLY(NOTIFY_STAR("  gprof $THIS > prof.out && less prof.out\n"));
      DEBUG_ONLY(NOTIFY_STAR(
          ANSI_BLACK"Program compiled with debug checks."ANSI_CLEAR"\n"));
    }
    ~InformDebug() {
      PROFILE_ONLY(NOTIFY_STAR("Profiling information available with:\n"));
      PROFILE_ONLY(NOTIFY_STAR("  gprof $THIS > prof.out && less prof.out\n"));
      DEBUG_ONLY(NOTIFY_STAR(
          ANSI_BLACK"Program compiled with debug checks."ANSI_CLEAR"\n"));
    }
  };

  /** Global instance prints messages before and after computation. */
  InformDebug inform_debug;
};

#endif
