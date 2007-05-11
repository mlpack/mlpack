// Copyright (c) 1995 Darren Vengroff
//
// File: ami_bit_permute.h
// Author: Darren Vengroff <darrenv@eecs.umich.edu>
// Created: 1/9/95
//
// $Id: ami_bit_permute.h,v 1.8 2004/08/12 12:35:29 jan Exp $
//
// For the moment this is done in terms of general permutations.
// This will obviously change in the future.
//
#ifndef _AMI_BIT_PERMUTE_H
#define _AMI_BIT_PERMUTE_H

// Get definitions for working with Unix and Windows
#include <portability.h>

// Get bit_matrix.
#include <bit_matrix.h>

// Get AMI_gen_perm_object.
#include <ami_gen_perm_object.h>
// Get the AMI_general_permute().
#include <ami_gen_perm.h>

class AMI_bit_perm_object {
private:
    // The matrices that define the permutation.
    bit_matrix mA;
    bit_matrix mc;
public:
    AMI_bit_perm_object(const bit_matrix &A,
                        const bit_matrix &c);
    ~AMI_bit_perm_object(void);

    bit_matrix A(void);
    bit_matrix c(void);
};
    
template<class T>
class bmmc_as_gen_po : public AMI_gen_perm_object {
private:
    bit_matrix *src_bits;
    bit_matrix A;
    bit_matrix c;
public:
    bmmc_as_gen_po(AMI_bit_perm_object &bpo) :
        A(bpo.A()), c(bpo.c())
    {
        tp_assert(A.rows() == A.cols(), "A is not square.");
        tp_assert(c.cols() == 1, "c is not a column vector.");
        tp_assert(c.rows() == A.cols(), "A and c dimensions do not match.");
        src_bits = new bit_matrix(c.rows(),1);
    };
    
    AMI_err initialize(TPIE_OS_OFFSET /*stream_len*/) {
        return AMI_ERROR_NO_ERROR;
    }
    
    TPIE_OS_OFFSET destination(TPIE_OS_OFFSET input_offset) {
        
        *src_bits = input_offset;

        bit_matrix r1 = A * *src_bits;
        bit_matrix res = r1 + c;

        return TPIE_OS_OFFSET(res);
    }
};

#ifndef TPIE_LIBRARY

template<class T>
AMI_err AMI_BMMC_permute(AMI_STREAM<T> *instream, AMI_STREAM<T> *outstream,
                         AMI_bit_perm_object *bpo)
{
    TPIE_OS_OFFSET sz_len = instream->stream_len();

    TPIE_OS_OFFSET sz_pow2;
    unsigned int bits;
    
    // Make sure the length of the input stream is a power of two.

    for (sz_pow2 = 1, bits = 0; sz_pow2 < sz_len; sz_pow2 += sz_pow2) {
        bits++;
    }

    if (sz_pow2 != sz_len) {
        return AMI_ERROR_NOT_POWER_OF_2;
    }
    
    // Make sure the number of bits in the permutation matrix matches
    // the log of the number of items in the input stream.

    {
        bit_matrix A = bpo->A();
        bit_matrix c = bpo->c();
        
        if (A.rows() != bits) {
            return AMI_ERROR_BIT_MATRIX_BOUNDS;
        }
    
        if (A.cols() != bits) {
            return AMI_ERROR_BIT_MATRIX_BOUNDS;
        }

        if (c.rows() != bits) {
            return AMI_ERROR_BIT_MATRIX_BOUNDS;
        }

        if (c.cols() != 1) {
            return AMI_ERROR_BIT_MATRIX_BOUNDS;
        }
    }
        
    // Create the general permutation object.
    bmmc_as_gen_po<T> gpo(*bpo);
    
    // Do the permutation.
    return AMI_general_permute(instream, outstream,
                               (AMI_gen_perm_object *)&gpo);
}

#endif // ndef TPIE_LIBRARY

#endif // _AMI_BIT_PERMUTE_H 
