////////////////////////////////////////////////////////////////////////////////
// The Loki Library
// Copyright (c) 2006 by Peter Kümmel
// Permission to use, copy, modify, distribute and sell this software for any 
//     purpose is hereby granted without fee, provided that the above copyright 
//     notice appear in all copies and that both that copyright notice and this 
//     permission notice appear in supporting documentation.
// The author makes no representations about the 
//     suitability of this software for any purpose. It is provided "as is" 
//     without express or implied warranty.
////////////////////////////////////////////////////////////////////////////////
#ifndef LOKI_LOKIEXPORT_INC_
#define LOKI_LOKIEXPORT_INC_

// $Id: LokiExport.h 748 2006-10-17 19:49:08Z syntheticpp $


#ifdef __GNUC__

#ifdef _HAVE_GCC_VISIBILITY
#define LOKI_EXPORT_SPEC __attribute__ ((visibility("default")))
#define LOKI_IMPORT_SPEC 
#else
#define LOKI_EXPORT_SPEC
#define LOKI_IMPORT_SPEC 
#endif

#else

#ifdef _WIN32
#define LOKI_EXPORT_SPEC __declspec(dllexport)
#define LOKI_IMPORT_SPEC __declspec(dllimport)
#else
#define LOKI_EXPORT_SPEC 
#define LOKI_IMPORT_SPEC 
#endif

#endif


#if (defined(LOKI_MAKE_DLL) && defined(LOKI_DLL)) || \
    (defined(LOKI_MAKE_DLL) && defined(LOKI_STATIC)) || \
    (defined(LOKI_DLL) && defined(LOKI_STATIC))
#error export macro error: you could not build AND use the library
#endif

#ifdef LOKI_MAKE_DLL
#define LOKI_EXPORT LOKI_EXPORT_SPEC
#endif

#ifdef LOKI_DLL
#define LOKI_EXPORT LOKI_IMPORT_SPEC
#endif

#ifdef LOKI_STATIC
#define LOKI_EXPORT
#endif

#if !defined(LOKI_EXPORT) && !defined(EXPLICIT_EXPORT)
#define LOKI_EXPORT
#endif

#ifndef LOKI_EXPORT
#error export macro error: LOKI_EXPORT was not defined, disable EXPLICIT_EXPORT or define a export specification
#endif


#endif // end file guardian

