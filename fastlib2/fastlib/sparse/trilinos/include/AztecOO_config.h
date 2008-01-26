/* src/AztecOO_config.h.  Generated from AztecOO_config.h.in by configure.  */
/* src/AztecOO_config.h.in.  Generated from configure.ac by autoheader.  */

/* Define to enable capture-matrix feature */
/* #undef AZ_ENABLE_CAPTURE_MATRIX */

/* Define to enable Teuchos TimeMonitors within Aztec solvers */
/* #undef AZ_ENABLE_TIMEMONITOR */

/* Define to dummy `main' function (if any) required to link to the Fortran
   libraries. */
/* #undef F77_DUMMY_MAIN */

/* Define to a macro mangling the given C identifier (in lower and upper
   case), which must not contain underscores, for linking with Fortran. */
#define F77_FUNC(name,NAME) name ## _

/* As F77_FUNC, but for C identifiers containing underscores. */
#define F77_FUNC_(name,NAME) name ## __

/* Define if F77 and FC dummy `main' functions are identical. */
/* #undef FC_DUMMY_MAIN_EQ_F77 */

/* Define to 1 if you have the <assert.h> header file. */
/* #undef HAVE_ASSERT_H */

/* Define if you want to build AZ_lu */
/* #undef HAVE_AZLU */

/* Define if want to build with aztecoo enabled */
/* #undef HAVE_AZTECOO_EPETRAEXT */

/* Define if want to build aztecoo-examples */
#define HAVE_AZTECOO_EXAMPLES 

/* Define if want to build with aztecoo enabled */
#define HAVE_AZTECOO_IFPACK 1

/* Define if want to build aztecoo-tests */
#define HAVE_AZTECOO_TESTS 

/* Define if want to build with aztecoo enabled */
#define HAVE_AZTECOO_TEUCHOS 1

/* Define if you have a BLAS library. */
#define HAVE_BLAS 1

/* define if bool is a built-in type */
#define HAVE_BOOL 

/* Define to 1 if you have the <cassert> header file. */
#define HAVE_CASSERT 1

/* Define to 1 if you have the <cfloat> header file. */
#define HAVE_CFLOAT 1

/* Define to 1 if you have the <cmath> header file. */
#define HAVE_CMATH 1

/* Define to 1 if you have the <cstdio> header file. */
#define HAVE_CSTDIO 1

/* Define to 1 if you have the <cstdlib> header file. */
#define HAVE_CSTDLIB 1

/* Define to 1 if you have the <ctime> header file. */
#define HAVE_CTIME 1

/* Define if want to build examples */
#define HAVE_EXAMPLES 

/* Define if you want to build export makefiles. */
#define HAVE_EXPORT_MAKEFILES 

/* Define to 1 if you have the <float.h> header file. */
/* #undef HAVE_FLOAT_H */

/* Define to 1 if you have the `floor' function. */
#define HAVE_FLOOR 1

/* Define to 1 if you have the <fstream> header file. */
#define HAVE_FSTREAM 1

/* Define if you are using gnumake - this will shorten your link lines. */
/* #undef HAVE_GNUMAKE */

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the <iomanip> header file. */
#define HAVE_IOMANIP 1

/* Define to 1 if you have the <iostream> header file. */
#define HAVE_IOSTREAM 1

/* Define if you have LAPACK library. */
#define HAVE_LAPACK 1

/* Define if want to build libcheck */
#define HAVE_LIBCHECK 

/* Define to 1 if your system has a GNU libc compatible `malloc' function, and
   to 0 otherwise. */
/* #undef HAVE_MALLOC */

/* Define to 1 if you have the <malloc.h> header file. */
#define HAVE_MALLOC_H 1

/* Define to 1 if you have the <math.h> header file. */
/* #undef HAVE_MATH_H */

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* define if we want to use MPI */
/* #undef HAVE_MPI */

/* define if the compiler supports the mutable keyword */
#define HAVE_MUTABLE 

/* define if the compiler implements namespaces */
#define HAVE_NAMESPACES 

/* define if the compiler accepts the new for scoping rules */
#define HAVE_NEW_FOR_SCOPING 

/* Define to 1 if you have the `pow' function. */
#define HAVE_POW 1

/* Define to 1 if you have the `sqrt' function. */
#define HAVE_SQRT 1

/* Define to 1 if you have the <sstream> header file. */
#define HAVE_SSTREAM 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdio.h> header file. */
/* #undef HAVE_STDIO_H */

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* define if std::sprintf is supported */
#define HAVE_STD_SPRINTF 

/* define if the compiler supports Standard Template Library */
#define HAVE_STL 

/* Define to 1 if you have the `strchr' function. */
#define HAVE_STRCHR 1

/* Define to 1 if you have the <string> header file. */
#define HAVE_STRING 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/time.h> header file. */
/* #undef HAVE_SYS_TIME_H */

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define if want to build tests */
#define HAVE_TESTS 

/* Define to 1 if you have the `uname' function. */
#define HAVE_UNAME 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "maherou@sandia.gov"

/* Define to the full name of this package. */
#define PACKAGE_NAME "aztecoo"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "aztecoo 3.6"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "aztecoo"

/* Define to the version of this package. */
#define PACKAGE_VERSION "3.6"

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Define to `__inline__' or `__inline' if that's what the C compiler
   calls it, or to nothing if 'inline' is not supported under any name.  */
#ifndef __cplusplus
/* #undef inline */
#endif

/* Define to rpl_malloc if the replacement function should be used. */
/* #undef malloc */
