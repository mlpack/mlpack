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
 * @file template_types.c
 *
 * Determines the size of each of the machine's basic types, automatically
 * generating basic_types.h.
 */

#include <stdio.h>

#define SHOWSIZE(def, type) \
    printf("#define %s %d\n", sizeof(type));

int main(int argc, char *argv[]) {
  const char *int8 = 0;
  const char *int16 = 0;
  const char *int32 = 0;
  const char *int64 = 0;
  const char *L8 = 0;
  const char *L16 = 0;
  const char *L32 = 0;
  const char *L64 = 0;

  printf("#ifndef FASTLIB_BASE_TYPE_H\n");
  printf("#define FASTLIB_BASE_TYPE_H\n");
  printf("\n");

  if (sizeof(char) == 1) {
    int8 = "char";
    L8 = "";
  } else {
    fprintf(stderr, "No 8-bit integral type.\n");
  }
  if (int8) {
    printf("typedef unsigned %s uint8;\n", int8);
    printf("typedef signed %s int8;\n", int8);
    printf("#define L8 \"%s\"\n", L8);
    printf("\n");
  }

  if (sizeof(short) == 2) {
    int16 = "short";
    L16 = "";
  } else {
    fprintf(stderr, "No 16-bit integral type.\n");
  }
  if (int16) {
    printf("typedef unsigned %s uint16;\n", int16);
    printf("typedef signed %s int16;\n", int16);
    printf("#define L16 \"%s\"\n", L16);
    printf("\n");
  }

  if (sizeof(int) == 4) {
    int32 = "int";
    L32 = "";
  } else if (sizeof(long) == 4) {
    int32 = "long";
    L32 = "l";
  } else {
    fprintf(stderr, "No 32-bit integral type.\n");
  }
  if (int32) {
    printf("typedef unsigned %s uint32;\n", int32);
    printf("typedef signed %s int32;\n", int32);
    printf("#define L32 \"%s\"\n", L32);
    printf("\n");
  }

  if (sizeof(long) == 8) {
    int64 = "long";
    L64 = "l";
  } else if (sizeof(long long) == 8) {
    int64 = "long long";
    L64 = "ll";
  } else {
    fprintf(stderr, "No 64-bit integral type.\n");
  }
  if (int64) {
    printf("typedef unsigned %s uint64;\n", int64);
    printf("typedef signed %s int64;\n", int64);
    printf("#define L64 \"%s\"\n", L64);
    printf("\n");
  }

  if (sizeof(float) == 4) {
    printf("typedef float float32;\n");
  } else {
    fprintf(stderr, "No 32-bit floating-point type.\n");
  }
  if (sizeof(double) == 8) {
    printf("typedef double float64;\n");
  } else {
    fprintf(stderr, "No 64-bit floating-point type.\n");
  }
  printf("\n");

  printf("#endif\n");

  return 0;
}
