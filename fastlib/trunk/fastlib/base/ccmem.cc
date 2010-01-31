// Copyright 2007 Georgia Institute of Technology. All rights reserved.
/**
 * @file ccmem.cc
 *
 * Implementations for non-template, non-inlined low-level memory
 * management routines.
 *
 * @see namespace mem
 */

#include "fastlib/base/ccmem.h"

const int32 mem__private::BIG_BAD_BUF[] = {
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER
};

void mem__private::PoisonBytes(char *array_cp, size_t bytes) {
  while (bytes >= BIG_BAD_BUF_SIZE) {
    ::memcpy(array_cp, BIG_BAD_BUF, BIG_BAD_BUF_SIZE);

    bytes -= BIG_BAD_BUF_SIZE;
    array_cp += BIG_BAD_BUF_SIZE;
  }
  if (bytes > 0) {
    ::memcpy(array_cp, BIG_BAD_BUF, bytes);
  }
}

void mem__private::SwapBytes(char *a_cp, char *b_cp, size_t bytes) {
  char buf[SWAP_BUF_SIZE];

  while (bytes >= SWAP_BUF_SIZE) {
    ::memcpy(buf, a_cp, SWAP_BUF_SIZE);
    ::memcpy(a_cp, b_cp, SWAP_BUF_SIZE);
    ::memcpy(b_cp, buf, SWAP_BUF_SIZE);

    bytes -= SWAP_BUF_SIZE;
    a_cp += SWAP_BUF_SIZE;
    b_cp += SWAP_BUF_SIZE;
  }
  if (bytes > 0) {
    ::memcpy(buf, a_cp, bytes);
    ::memcpy(a_cp, b_cp, bytes);
    ::memcpy(b_cp, buf, bytes);
  }
}
