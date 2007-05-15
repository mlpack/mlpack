// Copyright (c) 1995 Darren Erik Vengroff
//
// File: persist.h
// Author: Darren Erik Vengroff <darrenv@eecs.umich.edu>
// Created: 4/7/95
//
// $Id: persist.h,v 1.3 2003/09/13 17:42:27 jan Exp $
//
// Persistence flags for TPIE streams.
//
#ifndef _PERSIST_H
#define _PERSIST_H

// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"

enum persistence {
    // Delete the stream from the disk when it is destructed.
    PERSIST_DELETE = 0,
    // Do not delete the stream from the disk when it is destructed.
    PERSIST_PERSISTENT = 1,
    // Delete each block of data from the disk as it is read.
    // If not supported by the OS (see portability.h), delete
    // the stream when it is destructed (see PERSIST_DELETE).
    PERSIST_READ_ONCE = TPIE_OS_PERSIST_READ_ONCE
};

#endif // _PERSIST_H 
