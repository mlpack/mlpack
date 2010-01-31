// Copyright (C) 2001 Octavian Procopiuc
//
// File:   ami_coll_base.h
// Author: Octavian Procopiuc <tavi@cs.duke.edu>
//
// $Id: ami_coll_base.h,v 1.4 2004/08/17 16:47:45 jan Exp $
//
// Basic definitions for all AMI_COLLECTION implementations.
//
#ifndef _AMI_COLL_BASE_H
#define _AMI_COLL_BASE_H

// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"

// AMI collection types passed to constructors
enum AMI_collection_type {
    AMI_READ_COLLECTION = 1,	// Open existing collection for reading
    AMI_WRITE_COLLECTION,      	// Open for writing.  Create if non-existent
    AMI_READ_WRITE_COLLECTION	// Open to read and write.
};

// AMI collection status.
enum AMI_collection_status {
  AMI_COLLECTION_STATUS_VALID = 0,
  AMI_COLLECTION_STATUS_INVALID = 1
};

#endif // _AMI_COLL_BASE_H
