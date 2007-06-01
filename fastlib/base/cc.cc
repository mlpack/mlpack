#include "cc.h"

static const double DBL_ZERO = 0.0;
const double DBL_NAN = DBL_ZERO / DBL_ZERO;
const double FLT_NAN = DBL_NAN;

#if defined(DEBUG) || defined(PROFILE)
namespace cc__private {
/**
 * Class to emit a warning message whenever the library is compiled in
 * debug mode.
 */
class CCInformDebug {
 public:
  CCInformDebug() {
#ifdef DEBUG
    fprintf(stderr, "Program is being run with debugging checks on.\n");
#endif
  }
  ~CCInformDebug() {
#ifdef PROFILE
    fprintf(stderr, "[*] To collect profiling information:\n");
    fprintf(stderr, "[*] -> gprof $this_binary >profile.out && less profile.out\n");
#endif
#ifdef DEBUG
    fprintf(stderr, "Program was run with debugging checks on.\n");
#endif
  }
};

CCInformDebug cc_inform_debug_instance;
};
#endif
