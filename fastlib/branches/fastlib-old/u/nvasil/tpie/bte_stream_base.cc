//
// File: bte_stream_base.cpp
// Author: Octavian Procopiuc <tavi@cs.duke.edu>
//         (using some code by Darren Erik Vengroff)
// Created: 01/08/02
//

#include "lib_config.h"
#include <versions.h>
VERSION(bte_stream_base_cpp,"$Id: bte_stream_base.cpp,v 1.3 2003/04/23 07:32:15 tavi Exp $");

#include <bte_stream_base.h>

static unsigned long get_remaining_streams() {
	TPIE_OS_SET_LIMITS_BODY;
}

tpie_stats_stream BTE_stream_base_generic::gstats_;

int BTE_stream_base_generic::remaining_streams = get_remaining_streams();

