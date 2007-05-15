// File: versions.h
// Created: 99/11/15

// This file defines a macro VERSION that creates a static variable
// __name whose contents contain the given string __id. This is
// intended to be used for creating RCS version identifiers as static
// data in object files and executables.

// The "compiler_fooler stuff creates a (small) self-referential
// structure that prevents the compiler from warning that __name is
// never referenced.

// $Id: versions.h,v 1.5 2003/04/17 20:12:01 jan Exp $

#ifndef _VERSIONS_H
#define _VERSIONS_H

// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"

#endif // _VERSIONS_H 
