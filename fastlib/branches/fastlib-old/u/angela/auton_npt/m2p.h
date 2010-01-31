/*
   File:        m2p.h
   Author:      Andrew W. Moore
   Created:     July 12, 2000
   Description: Multi-radius 2-pt

   Copyright 2000, the Auton Lab
*/

#ifndef M2P_H
#define M2P_H

#include "npt.h"
#include "lingraph.h"

typedef struct bucket
{
  double thresh_ntuples;
  int lo_tindex;
  int hi_tindex; /* Represents the set of tindexes
                    { lo , lo+1 , lo+2 , .... hi-1 } */
} bucket;

typedef struct bres
{
  bucket *b;
  double log_lo_count;
  double log_hi_count;
} bres;

#define MAX_BRESSES 3000
  
typedef struct bresses
{
  int size;
  bres *bs[MAX_BRESSES];
  double max_log_count; /* -ve means compute me lazily, baby */
} bresses;

#endif
