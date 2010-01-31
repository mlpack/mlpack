//
// File: ami_err.h
// Author: Octavian Procopiuc <tavi@cs.duke.edu>
//         (from Darren's ami_base.h)
// Created: 12/29/01
// $Id: ami_err.h,v 1.4 2005/07/07 20:38:39 adanner Exp $
//
// AMI error codes, moved here from ami_base.h
//

#ifndef _AMI_ERR_H
#define _AMI_ERR_H

// AMI error codes are returned using the AMI_err type.
enum AMI_err {
  AMI_ERROR_NO_ERROR = 0,
  AMI_ERROR_IO_ERROR,
  AMI_ERROR_END_OF_STREAM,
  AMI_ERROR_READ_ONLY,
  AMI_ERROR_OS_ERROR,
  AMI_ERROR_BASE_METHOD,
  AMI_ERROR_BTE_ERROR,
  AMI_ERROR_MM_ERROR,
  AMI_ERROR_OBJECT_INITIALIZATION,
  AMI_ERROR_OBJECT_INVALID,
  AMI_ERROR_PERMISSION_DENIED,
  AMI_ERROR_INSUFFICIENT_MAIN_MEMORY,
  AMI_ERROR_INSUFFICIENT_AVAILABLE_STREAMS,
  AMI_ERROR_ENV_UNDEFINED,
  AMI_ERROR_NO_MAIN_MEMORY_OPERATION,
  AMI_ERROR_BIT_MATRIX_BOUNDS,
  AMI_ERROR_NOT_POWER_OF_2,
  AMI_ERROR_NULL_POINTER,

  AMI_ERROR_GENERIC_ERROR = 0xfff,

  // Values returned by scan objects.
  AMI_SCAN_DONE = 0x1000,
  AMI_SCAN_CONTINUE,

  // Values returned by merge objects.
  AMI_MERGE_DONE = 0x2000,
  AMI_MERGE_CONTINUE,
  AMI_MERGE_OUTPUT,
  AMI_MERGE_READ_MULTIPLE,

  // Matrix related errors
  AMI_MATRIX_BOUNDS = 0x3000,

  // Values returned by sort routines.
  AMI_SORT_ALREADY_SORTED = 0x4000
  
};

#endif // _AMI_ERR_H
