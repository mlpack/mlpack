//
// File: app_config.h
// Authors: Darren Erik Vengroff
//          Octavian Procopiuc <tavi@cs.duke.edu>
//
// Created: 10/6/94
//
// $Id: app_config.h,v 1.36 2004/08/17 16:49:16 jan Exp $
//
#ifndef _APP_CONFIG_H
#define _APP_CONFIG_H

// Get the configuration as set up by the TPIE configure script.
#include <config.h>

// <><><><><><><><><><><><><><><><><><><><><><> //
// <><><><><><><> Developer use  <><><><><><><> //
// <><><><><><><><><><><><><><><><><><><><><><> //

// Set up some defaults for the test applications

#include <portability.h>
#include <sys/types.h> // for size_t
#include <stdlib.h> // for random()

#define DEFAULT_TEST_SIZE (20000000)
#define DEFAULT_RANDOM_SEED 17
#define DEFAULT_TEST_MM_SIZE (1024 * 1024 * 32)

extern bool verbose;
extern TPIE_OS_SIZE_T test_mm_size;
extern TPIE_OS_OFFSET test_size;
extern int random_seed;


// <><><><><><><><><><><><><><><><><><><><><><> //
// <><><> Choose default BTE COLLECTION  <><><> //
// <><><><><><><><><><><><><><><><><><><><><><> //

#if (!defined(BTE_COLLECTION_IMP_MMAP) && !defined(BTE_COLLECTION_IMP_UFS) && !defined(BTE_COLLECTION_IMP_USER_DEFINED))
// Define only one (default is BTE_COLLECTION_IMP_MMAP)
#define BTE_COLLECTION_IMP_MMAP
//#define BTE_COLLECTION_IMP_UFS
//#define BTE_COLLECTION_IMP_USER_DEFINED
#endif

// <><><><><><><><><><><><><><><><><><><><><><> //
// <><><><><><> Choose BTE STREAM  <><><><><><> //
// <><><><><><><><><><><><><><><><><><><><><><> //

// Define only one (default is BTE_STREAM_IMP_UFS)
#define BTE_STREAM_IMP_UFS
//#define BTE_STREAM_IMP_MMAP
//#define BTE_STREAM_IMP_STDIO
//#define BTE_STREAM_IMP_USER_DEFINED


// <><><><><><><><><><><><><><><><><><><><><><><><> //
// <> BTE_COLLECTION_MMAP configuration options  <> //
// <><><><><><><><><><><><><><><><><><><><><><><><> //

// Define write behavior.
// Allowed values:
//  0    (synchronous writes)
//  1    (asynchronous writes using MS_ASYNC - see msync(2))
//  2    (asynchronous bulk writes) [default]
#ifndef BTE_COLLECTION_MMAP_LAZY_WRITE 
#define BTE_COLLECTION_MMAP_LAZY_WRITE 2
#endif

// <><><><><><><><><><><><><><><><><><><><><><><><> //
// <> BTE_COLLECTION_UFS configuration options   <> //
// <><><><><><><><><><><><><><><><><><><><><><><><> //



// <><><><><><><><><><><><><><><><><><><><><><><><> //
// <><> BTE_STREAM_MMAP configuration options  <><> //
// <><><><><><><><><><><><><><><><><><><><><><><><> //

#ifdef BTE_STREAM_IMP_MMAP
// Define logical blocksize factor (default is 32)
#ifndef BTE_STREAM_MMAP_BLOCK_FACTOR
#ifdef _WIN32
#define BTE_STREAM_MMAP_BLOCK_FACTOR 4
#else
#define BTE_STREAM_MMAP_BLOCK_FACTOR 32
#endif
#endif

 // Enable/disable TPIE read ahead; default is enabled (set to 1)
//#define BTE_STREAM_MMAP_READ_AHEAD 1

// read ahead method, ignored unless BTE_STREAM_MMAP_READ_AHEAD is set
// to 1; if USE_LIBAIO is enabled, use asynchronous IO read ahead;
// otherwise use use mmap-based read ahead; default is mmap-based read
// ahead (USE_LIBAIO not defined)

//#define USE_LIBAIO
#endif


// <><><><><><><><><><><><><><><><><><><><><><><><> //
// <><> BTE_STREAM_UFS configuration options <><><> //
// <><><><><><><><><><><><><><><><><><><><><><><><> //

#ifdef BTE_STREAM_IMP_UFS
 // Define logical blocksize factor (default is 32)
#ifndef BTE_STREAM_UFS_BLOCK_FACTOR
#ifdef _WIN32
#define BTE_STREAM_UFS_BLOCK_FACTOR 4
#else
#define BTE_STREAM_UFS_BLOCK_FACTOR 32
#endif
#endif

 // Enable/disable TPIE read ahead; default is disabled (set to 0)
#define BTE_STREAM_UFS_READ_AHEAD 0
// read ahead method, ignored unless BTE_STREAM_UFS_READ_AHEAD is set
// to 1; if USE_LIBAIO is set to 1, use asynchronous IO read ahead;
// otherwise no TPIE read ahead is done; default is disabled (set to 0)
#define USE_LIBAIO 0
#endif


// <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> //
//                   logging and assertions;                    //
//            this should NOT be modified by user!!!            //
//   in order to enable/disable library/application logging,    //
//     run tpie configure script with appropriate options       //
// <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> //

// Use logs if requested.
#if TP_LOG_APPS
#define TPL_LOGGING 1
#endif

#include <tpie_log.h>

// Enable assertions if requested.
#if TP_ASSERT_APPS
#define DEBUG_ASSERTIONS 1
#endif

#include <tpie_assert.h>

#endif
