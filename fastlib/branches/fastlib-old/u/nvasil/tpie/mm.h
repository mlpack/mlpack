// Copyright (c) 1994 Darren Erik Vengroff
//
// File: mm.h (plus contents from mm_imps.h, now deprecated)
// Author: Darren Erik Vengroff <dev@cs.duke.edu>
// Created: 5/30/94
//
// $Id: mm.h,v 1.3 2003/04/17 19:38:28 jan Exp $
//
#ifndef _MM_H
#define _MM_H

// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"

// Get the base class, enums, etc...
#include "u/nvasil/tpie/mm_base.h"

// Get an implementation definition

// For now only single address space memory management is supported.
#ifdef MM_IMP_REGISTER
#include "u/nvasil/tpie/mm_register.h"
#else
#error No MM implementation selected.
#endif

#endif // _MM_H 
