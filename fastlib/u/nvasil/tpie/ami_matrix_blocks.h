// Copyright (c) 1994 Darren Vengroff
//
// File: ami_matrix_blocks.h
// Author: Darren Vengroff <darrenv@eecs.umich.edu>
// Created: 12/11/94
//
// $Id: ami_matrix_blocks.h,v 1.4 2004/08/12 12:35:30 jan Exp $
//
#ifndef _AMI_MATRIX_BLOCKS_H
#define _AMI_MATRIX_BLOCKS_H

// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"
// Get AMI_gen_perm_object.
#include "u/nvasil/tpie/ami_gen_perm_object.h"

class perm_matrix_into_blocks : public AMI_gen_perm_object {
private:
    TPIE_OS_OFFSET r,c,be;
public:    
    perm_matrix_into_blocks(TPIE_OS_OFFSET rows, TPIE_OS_OFFSET cols,
                            TPIE_OS_OFFSET block_extent);
    virtual ~perm_matrix_into_blocks();
    AMI_err initialize(TPIE_OS_OFFSET len);
    TPIE_OS_OFFSET destination(TPIE_OS_OFFSET source);
};

class perm_matrix_outof_blocks : public AMI_gen_perm_object {
private:
    TPIE_OS_OFFSET r,c,be;
public:    
    perm_matrix_outof_blocks(TPIE_OS_OFFSET rows, TPIE_OS_OFFSET cols,
                             TPIE_OS_OFFSET block_extent);
    virtual ~perm_matrix_outof_blocks();
    AMI_err initialize(TPIE_OS_OFFSET len);
    TPIE_OS_OFFSET destination(TPIE_OS_OFFSET source);
};


#endif // _AMI_MATRIX_BLOCKS_H 



