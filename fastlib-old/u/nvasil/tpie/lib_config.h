//
// File: lib_config.h
// Author: Darren Vengroff <darrenv@eecs.umich.edu>
// Created: 10/31/94
//
// $Id: lib_config.h,v 1.4 2003/04/17 21:00:01 jan Exp $
//
#ifndef _LIB_CONFIG_H
#define _LIB_CONFIG_H

#include "u/nvasil/tpie/config.h"

// Use logs if requested.
#if TP_LOG_LIB
#define TPL_LOGGING 1
#endif
#include "u/nvasil/tpie/tpie_log.h"

// Enable assertions if requested.
#if TP_ASSERT_LIB
#define DEBUG_ASSERTIONS 1
#endif
#include "u/nvasil/tpie/tpie_assert.h"


#endif // _LIB_CONFIG_H 
 
