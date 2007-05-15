// Copyright (C) 2001 Octavian Procopiuc
//
// File:    tpie_stats_tree.h
// Author:  Octavian Procopiuc <tavi@cs.duke.edu>
//
// $Id: tpie_stats_tree.h,v 1.2 2003/04/17 20:07:23 jan Exp $
//
//
#ifndef _TPIE_STATS_TREE_H
#define _TPIE_STATS_TREE_H

// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"

#include "u/nvasil/tpie/tpie_stats.h"

#define TPIE_STATS_TREE_COUNT 14
enum TPIE_STATS_TREE {
  LEAF_FETCH = 0,
  LEAF_RELEASE,
  LEAF_READ,
  LEAF_WRITE,
  LEAF_CREATE,
  LEAF_DELETE,
  LEAF_COUNT,
  NODE_FETCH,
  NODE_RELEASE,
  NODE_READ,
  NODE_WRITE,
  NODE_CREATE,
  NODE_DELETE,
  NODE_COUNT
};

typedef tpie_stats<TPIE_STATS_TREE_COUNT> tpie_stats_tree;

#endif // _TPIE_STATS_TREE_H
