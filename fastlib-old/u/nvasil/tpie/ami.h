// Copyright (c) 1994 Darren Erik Vengroff
//
// File: ami.h
// Author: Darren Erik Vengroff <dev@cs.duke.edu>
// Created: 5/19/94
//
// $Id: ami.h,v 1.19 2003/04/17 11:59:40 jan Exp $
//
#ifndef _AMI_H
#define _AMI_H

// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"

// Get a stream implementation.
#include "u/nvasil/tpie/ami_stream.h"

// Get templates for ami_scan().
#include "u/nvasil/tpie/ami_scan.h"

// Get templates for ami_merge().
#include "u/nvasil/tpie/ami_merge.h"

// Get templates for ami_sort().
#include "u/nvasil/tpie/ami_sort.h"

// Get templates for general permutation.
#include "u/nvasil/tpie/ami_gen_perm.h"

// Get templates for bit permuting.
#include "u/nvasil/tpie/ami_bit_permute.h"

// Get a collection implementation.
#include "u/nvasil/tpie/ami_coll.h"

// Get a block implementation.
#include "u/nvasil/tpie/ami_block.h"

// Get templates for AMI_btree.
#include "u/nvasil/tpie/ami_btree.h"

#endif // _AMI_H 
