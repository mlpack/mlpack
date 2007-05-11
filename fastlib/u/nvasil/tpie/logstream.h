// Copyright (c) 1994 Darren Erik Vengroff
//
// File: logstream.h
// Author: Darren Erik Vengroff <dev@cs.duke.edu>
// Created: 5/12/94
//
// $Id: logstream.h,v 1.20 2004/08/17 16:48:14 jan Exp $
//

#ifndef _LOGSTREAM_H
#define _LOGSTREAM_H

// Get definitions for working with Unix and Windows
#include <portability.h>

// For size_t
#include <sys/types.h>

// A macro for declaring output operators for log streams.
#define _DECLARE_LOGSTREAM_OUTPUT_OPERATOR(T) logstream& operator<<(T)

// A log is like a regular output stream, but it also supports messages
// at different priorities.  If a message's priority is at least as high
// as the current priority threshold, then it appears in the log.  
// Otherwise, it does not.  Lower numbers have higher priority; 0 is
// the highest.  1 is the default if not 

class logstream : public ofstream {

  public:
    static bool log_initialized;
    unsigned int priority;
    unsigned int threshold;

    logstream(const char *fname, unsigned int p = 0, unsigned int tp = 0);
    ~logstream();

    // Output operators

    _DECLARE_LOGSTREAM_OUTPUT_OPERATOR(const char *);
    _DECLARE_LOGSTREAM_OUTPUT_OPERATOR(const char);
    _DECLARE_LOGSTREAM_OUTPUT_OPERATOR(const int);
    _DECLARE_LOGSTREAM_OUTPUT_OPERATOR(const unsigned int);
    _DECLARE_LOGSTREAM_OUTPUT_OPERATOR(const long int);
    _DECLARE_LOGSTREAM_OUTPUT_OPERATOR(const long unsigned int);
    _DECLARE_LOGSTREAM_OUTPUT_OPERATOR(const float);
    _DECLARE_LOGSTREAM_OUTPUT_OPERATOR(const double);
    
    //  Unix "long long", Win32 "LONGLONG".
    TPIE_OS_DECLARE_LOGSTREAM_LONGLONG
};


// The logmanip template is based on the omanip template from iomanip.h 
// in the libg++ sources.

template <class TP> class logmanip {
    logstream& (*_f)(logstream&, TP);
    TP _a;
public:
    logmanip(logstream& (*f)(logstream&, TP), TP a) : _f(f), _a(a) {}

    friend logstream& operator<< (logstream& o, const logmanip<TP>& m) {
	(*m._f)(o, m._a); 
	return o;
    }
};


logmanip<unsigned long> setpriority(unsigned long p);
logmanip<unsigned long> setthreshold(unsigned long p);

#endif // _LOGSTREAM_H 
