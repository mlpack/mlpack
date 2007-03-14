#include "cc.h"

#if defined(DEBUG) || defined(PROFILE)
namespace cc__private {
/**
 * Class to emit a warning message whenever the library is compiled in
 * debug mode.
 */
class CCInformDebug {
 public:
  CCInformDebug() {
    DEBUG_MSG(0.0, "Running in debug mode; performance is sub-optimal.");
  }
  ~CCInformDebug() {
#ifdef PROFILE
    fprintf(stderr, "[*] To collect profiling information:\n");
    fprintf(stderr, "[*] -> gprof $this_binary >profile.out && less profile.out\n");
#endif
    DEBUG_MSG(0.0, "Program was run in debug mode; up to 25%% slower.");
  }
};

CCInformDebug cc_inform_debug_instance;
};
#endif
