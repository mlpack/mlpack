//
// File:    tpie_stats_stream.h
// Authors: Octavian Procopiuc <tavi@cs.duke.edu>
//
// $Id: tpie_stats_stream.h,v 1.3 2004/08/17 16:48:25 jan Exp $
//
// Statistics for streams.

#ifndef _TPIE_STATS_STREAM_H
#define _TPIE_STATS_STREAM_H

// Get definitions for working with Unix and Windows
#include <portability.h>

#include <tpie_stats.h>

#define TPIE_STATS_STREAM_COUNT 11
enum TPIE_STATS_STREAM {
  BLOCK_READ = 0,
  BLOCK_WRITE,
  ITEM_READ,
  ITEM_WRITE,
  ITEM_SEEK,
  STREAM_OPEN,
  STREAM_CLOSE,
  STREAM_CREATE,
  STREAM_DELETE,
  SUBSTREAM_CREATE,
  SUBSTREAM_DELETE
};

typedef tpie_stats<TPIE_STATS_STREAM_COUNT> tpie_stats_stream;

#endif //_TPIE_STATS_STREAM_H
