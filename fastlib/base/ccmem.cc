/**
 * @file ccmem.cc
 *
 * Implementations for ccmem non-inline routines.
 */

#include "ccmem.h"

void mem::SwapBytes__Chars(long *a_lp_in, long *b_lp_in, ssize_t remaining) {
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

void mem::SwapBytes__Impl(long *a_lp, long *b_lp, ssize_t remaining) {
  //DEBUG_MSG(3.0,"Swapping %d bytes, %d left", int(elems), int(remaining));
  
  // TODO: Not as good as an MMX memcpy, but still good...
  // TODO: replace 'remaining' decrement with end pointer
  for (;;) {
    remaining -= sizeof(long);
    if (unlikely(remaining < 0)) break;
    long ta = *a_lp;
    long tb = *b_lp;
    *b_lp = ta;
    b_lp++;
    *a_lp = tb;
    a_lp++;
  }
  
  remaining += sizeof(long);
  
  if (unlikely(remaining != 0)) {
    SwapBytes__Chars(a_lp, b_lp, remaining);
  }
}
