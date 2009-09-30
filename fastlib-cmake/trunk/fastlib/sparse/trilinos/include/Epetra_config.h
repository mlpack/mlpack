/* src/Epetra_config.h.  Generated from Epetra_config.h.in by configure.  */
/* src/Epetra_config.h.in.  Generated from configure.ac by autoheader.  */

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

/* Define if you have a BLAS library. */
#define HAVE_BLAS 1

/* Define to 1 if you have the <cassert> header file. */
#define HAVE_CASSERT 1

/* Define to 1 if you have the <cmath> header file. */
#define HAVE_CMATH 1

/* Define to 1 if you have the <cstdio> header file. */
#define HAVE_CSTDIO 1

/* Define to 1 if you have the <cstdlib> header file. */
#define HAVE_CSTDLIB 1

/* Define if want to build epetra-abc */
/* #undef HAVE_EPETRA_ARRAY_BOUNDS_CHECK */

/* Define if want to build epetra-examples */
#define HAVE_EPETRA_EXAMPLES 

/* Define if want to build epetra-tests */
#define HAVE_EPETRA_TESTS 

/* Define if want to build with epetra enabled */
#define HAVE_EPETRA_TEUCHOS 1

/* Define if want to build examples */
#define HAVE_EXAMPLES 

/* Define if you want to build export makefiles. */
#define HAVE_EXPORT_MAKEFILES 

/* Define if want to build with fatal_messages enabled */
#define HAVE_FATAL_MESSAGES 1

/* Define if want to build with format_io enabled */
#define HAVE_FORMAT_IO 1

/* Define if you are using gnumake - this will shorten your link lines. */
/* #undef HAVE_GNUMAKE */

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the <iomanip> header file. */
#define HAVE_IOMANIP 1

/* Define to 1 if you have the <iomanip.h> header file. */
/* #undef HAVE_IOMANIP_H */

/* Define to 1 if you have the <iostream> header file. */
#define HAVE_IOSTREAM 1

/* Define to 1 if you have the <iostream.h> header file. */
/* #undef HAVE_IOSTREAM_H */

/* Define if you have LAPACK library. */
#define HAVE_LAPACK 1

/* Define if want to build libcheck */
#define HAVE_LIBCHECK 

/* Define to 1 if you have the <math.h> header file. */
#define HAVE_MATH_H 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* define if we want to use MPI */
/* #undef HAVE_MPI */

/* Define to 1 if you have the <sstream> header file. */
#define HAVE_SSTREAM 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdio.h> header file. */
/* #undef HAVE_STDIO_H */

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <string> header file. */
#define HAVE_STRING 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define if want to build tests */
#define HAVE_TESTS 

/* Define if want to build with threads enabled */
/* #undef HAVE_THREADS */

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define if want to build with warning_messages enabled */
/* #undef HAVE_WARNING_MESSAGES */

/* Define if want to build with zoltan enabled */
/* #undef HAVE_ZOLTAN */

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "maherou@sandia.gov"

/* Define to the full name of this package. */
#define PACKAGE_NAME "epetra"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "epetra 3.6"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "epetra"

/* Define to the version of this package. */
#define PACKAGE_VERSION "3.6"

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1
