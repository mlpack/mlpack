// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file fastlib.h
 *
 * Includes all of fastlib.
 */

#ifndef FASTLIB_FASTLIB_H
#define FASTLIB_FASTLIB_H

#include "base/common.h"
#include "base/cc.h"
#include "la/la.h"
#include "la/matrix.h"
#include "col/arraylist.h"
#include "col/heap.h"
#include "col/string.h"
#include "data/dataset.h"
#include "data/crossvalidation.h"
#include "math/math.h"
#include "file/textfile.h"
#include "fx/fx.h"
#include "par/thread.h"
#include "par/grain.h"
#include "tree/spacetree.h"
#include "tree/bounds.h"
#include "tree/statistic.h"
#include "tree/kdtree.h"

#endif

/**
 * @mainpage FASTlib Documentation
 *
 * @section intro_sec Introduction
 *
 * First make sure you have looked at the tutorial on the wiki 
 * and sample code in @c u/example.  The
 * sample code is meant as a starting point -- you should begin by hacking
 * onto the sample code, though make sure to make your own copy so that
 * you don't check it into subversion.
 *
 * The cookbook should be built as a sort of micro-howto for small tasks that
 * you are interested in doing.
 * However, this Doxygen documentation will always be the
 * most complete -- and Doxygen links to the cross-referenced code if the
 * comments aren't good enough.  If you must declare classes in header
 * files that you don't want documented because they are just helpers,
 * remove them to a file ending in "_impl.h" and it will be skipped.
 *
 * @section using_doxygen Using Doxygen in Your Code
 *
 * So, you want your code to be documented like this, too?  Doxygen will,
 * like Javadoc, scrape your source code for comments with the specific
 * slash-star-star begining.  It's best to comment header files and not
 * source files -- even without Doxygen, header files remain a natural place
 * for documentation in any case.
 *
 * To generate this HTML documentation yourself, go in to the source
 * directory and type:
 *
 * @code
 * doxygen
 * @endcode
 *
 * Then, visit @c doc/html/index.html (inside the same directory).
 *
 * @section remarks Final Remarks
 *
 * This software was written at Georgia Institute of Technology.
 * This software is not yet ready for distribution.
 *
 * The current core maintainers are garryb@gatech.edu
 * and rriegel@cc.gatech.edu.  
 *
 * The source for this main page is in @c fastlib/fastlib.h.
 */

