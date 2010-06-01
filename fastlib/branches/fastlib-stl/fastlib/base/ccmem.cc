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
 * @file ccmem.cc
 *
 * Implementations for non-template, non-inlined low-level memory
 * management routines.
 *
 * @see namespace mem
 */

#include "ccmem.h"

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
