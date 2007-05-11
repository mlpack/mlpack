// Copyright (c) 1995 Darren Erik Vengroff
//
// File: ami_key.h
// Author: Darren Erik Vengroff <dev@cs.duke.edu>
// Created: 3/12/95
//
// $Id: ami_key.h,v 1.2 2003/04/17 12:34:44 jan Exp $
//
#ifndef _AMI_KEY_H
#define _AMI_KEY_H

// Get definitions for working with Unix and Windows
#include <portability.h>

// Temporary until the configuration script is edited to determine word
// size.
#define UINT32 unsigned long


// Radix keys are unsigned 32 bit integers.
typedef UINT32 kb_key;

#define KEY_MAX 0x80000000
#define KEY_MIN 0

// A range of keys.  A stream having this range of keys is guarantted
// to have no keys < min and no keys >= max.
class key_range {
public:
    kb_key min;
    kb_key max;
    key_range(void);
    key_range(kb_key min_key, kb_key max_key);
};

#endif // _AMI_KEY_H 
