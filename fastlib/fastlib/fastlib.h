// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file fastlib.h
 *
 * Includes all of fastlib.
 */


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
//#include "file/serialize.h"
#include "file/textfile.h"
#include "fx/fx.h"
#include "par/thread.h"
#include "par/grain.h"
#include "tree/spacetree.h"
#include "tree/bounds.h"
#include "tree/statistic.h"
#include "tree/kdtree.h"


/** @mainpage FASTlib Documentation
 *
 * @section intro_sec Introduction
 *
 * First make sure you have looked at the tutorial and sample code.  The
 * sample code is meant as a starting point -- you should begin by hacking
 * onto the sample code.
 *
 * The cookbook should be built as a sort of micro-howto for small tasks that
 * you are interested in doing.
 * However, this Doxygen documentation will always be the
 * most complete -- and Doxygen links to the cross-reference code if the
 * comments aren't good enough.
 *
 * @section using_doxygen Using Doxygen in Your Code
 *
 * So, you want your code to be documented like this too?  Doxygen will,
 * like Javadoc, scrape your source code for comments with the specific
 * slash-star-star begining.  We recommend you
 * comment header files and not source files -- even without Doxygen, header
 * files remain a natural place for documentation.
 *
 * To generate this HTML documentation yourself, go in to the code directory
 * and type:
 *
 * @code
 * doxygen
 * @endcode
 *
 * Then, visit doc/html/index.html (within the code directory).
 *
 * @section remarks Final Remarks
 *
 * This software was written at Georgia Institute of Technology.
 * This software is not yet ready for distribution.
 *
 * The current core maintainers are garryb@cc.gatech.edu
 * and rriegel@cc.gatech.edu.  
 *
 * The source for this main page is in fastlib/fastlib.h.
 */

