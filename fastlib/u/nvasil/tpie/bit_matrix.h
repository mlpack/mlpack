// Copyright (c) 1994 Darren Vengroff
//
// File: bit_matrix.h
// Author: Darren Vengroff <darrenv@eecs.umich.edu>
// Created: 11/4/94
//
// $Id: bit_matrix.h,v 1.14 2005/01/14 18:35:00 tavi Exp $
//
#ifndef _BIT_MATRIX_H
#define _BIT_MATRIX_H

// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"

#include "u/nvasil/tpie/bit.h"
#include "u/nvasil/tpie/matrix.h"

#include <sys/types.h>


// typedef matrix<bit> bit_matrix_0;

class bit_matrix : public matrix<bit> {
public:
  using matrix<bit>::rows;
  using matrix<bit>::cols;
  
  bit_matrix(matrix<bit> &mb);
  bit_matrix(TPIE_OS_SIZE_T rows, TPIE_OS_SIZE_T cols);
  virtual ~bit_matrix(void);

  bit_matrix operator=(const bit_matrix &rhs);
    
  // We can assign from an offset, which is typically a source
  // address for a BMMC permutation.
  bit_matrix &operator=(const TPIE_OS_OFFSET &rhs);

  operator TPIE_OS_OFFSET(void);
  
  friend bit_matrix operator+(const bit_matrix &op1, const bit_matrix &op2);
  friend bit_matrix operator*(const bit_matrix &op1, const bit_matrix &op2);
};

bit_matrix operator+(const bit_matrix &op1, const bit_matrix &op2);
bit_matrix operator*(const bit_matrix &op1, const bit_matrix &op2);

ostream &operator<<(ostream &s, bit_matrix &bm);

#endif // _BIT_MATRIX_H 
