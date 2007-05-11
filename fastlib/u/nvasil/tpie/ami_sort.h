//
// File: ami_sort.h
// Author: Darren Erik Vengroff <dev@cs.duke.edu>
// Created: 6/10/94
//
// $Id: ami_sort.h,v 1.11 2003/09/27 07:00:48 tavi Exp $
//
#ifndef _AMI_SORT_H
#define _AMI_SORT_H

// Get definitions for working with Unix and Windows
#include <portability.h>

#define CONST const

#include <ami_stream.h>

#ifdef AMI_STREAM_IMP_SINGLE
#include <ami_sort_single.h>
#include <ami_optimized_sort.h>
#include <ami_sort_single_dh.h>
#endif

#endif // _AMI_SORT_H 
