/**
 * @file ccmem.cc
 *
 * Implementations for ccmem non-inline routines.
 */

#include "ccmem.h"

namespace mem {
  void SwapBytes__Chars(long *a_lp_in, long *b_lp_in, size_t remaining) {
    char *a_cp = reinterpret_cast<char*>(a_lp_in);
    char *b_cp = reinterpret_cast<char*>(b_lp_in);
    
    while (remaining) {
      char ta = *a_cp;
      char tb = *b_cp;
      remaining--;
      *b_cp = ta;
      b_cp++;
      *a_cp = tb;
      a_cp++;
    }
  }
};

