//
// File: bte_err.h
// Author: Octavian Procopiuc <tavi@cs.duke.edu>
//         (from Darren's bte_base_stream.h)
// Created: 12/29/01
// $Id: bte_err.h,v 1.2 2003/04/17 14:56:26 jan Exp $
//
// BTE error codes, moved here from bte_base_stream.h
//

#ifndef _BTE_ERR_H
#define _BTE_ERR_H

// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"

//
// BTE error codes are returned using the BTE_err type.
//
enum BTE_err {
  BTE_ERROR_NO_ERROR = 0,
  BTE_ERROR_IO_ERROR,
  BTE_ERROR_END_OF_STREAM,
  BTE_ERROR_READ_ONLY,
  BTE_ERROR_OS_ERROR,
  BTE_ERROR_BASE_METHOD,
  BTE_ERROR_MEMORY_ERROR,
  BTE_ERROR_PERMISSION_DENIED,
  BTE_ERROR_OFFSET_OUT_OF_RANGE,
  BTE_ERROR_OUT_OF_SPACE,
  BTE_ERROR_STREAM_IS_SUBSTREAM,
  BTE_ERROR_WRITE_ONLY,
  BTE_ERROR_BAD_HEADER,
  BTE_ERROR_INVALID_PLACEHOLDER
};

#endif // _BTE_ERR_H
