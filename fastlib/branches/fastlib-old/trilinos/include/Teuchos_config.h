/* src/Teuchos_config.h.  Generated from Teuchos_config.h.in by configure.  */
/* src/Teuchos_config.h.in.  Generated from configure.ac by autoheader.  */

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

/* Define to 1 if you have the <algorithm> header file. */
#define HAVE_ALGORITHM 1

/* Define if you have a BLAS library. */
#define HAVE_BLAS 1

/* define if bool is a built-in type */
#define HAVE_BOOL 

/* Define to 1 if you have the <cassert> header file. */
#define HAVE_CASSERT 1

/* Define to 1 if you have the <cerrno> header file. */
#define HAVE_CERRNO 1

/* Define to 1 if you have the <climits> header file. */
#define HAVE_CLIMITS 1

/* Define to 1 if you have the <cmath> header file. */
#define HAVE_CMATH 1

/* Define to 1 if you have the <complex> header file. */
#define HAVE_COMPLEX 1

/* Define to 1 if you have the <cstdarg> header file. */
#define HAVE_CSTDARG 1

/* Define to 1 if you have the <cstdlib> header file. */
#define HAVE_CSTDLIB 1

/* Define to 1 if you have the <cstring> header file. */
#define HAVE_CSTRING 1

/* Define if want to build examples */
#define HAVE_EXAMPLES 

/* Define if you want to build export makefiles. */
#define HAVE_EXPORT_MAKEFILES 

/* Define to 1 if you have the <fpu_control.h> header file. */
#define HAVE_FPU_CONTROL_H 1

/* define if the compiler supports abi::__cxa_demangle(...) */
#define HAVE_GCC_ABI_DEMANGLE 

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

/* Define to 1 if you have the <list> header file. */
#define HAVE_LIST 1

/* Define to 1 if you have the <map> header file. */
#define HAVE_MAP 1

/* Define to 1 if you have the <memory> header file. */
#define HAVE_MEMORY 1

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

/* define if the compiler supports numeric_limits<> */
#define HAVE_NUMERIC_LIMITS 

/* Define to 1 if you have the `pow' function. */
#define HAVE_POW 1

/* define if the compiler supports access of protected templated nested
   classes in derived classes */
#define HAVE_PROTECTED_NESTED_TEMPLATE_CLASS_ACCESS 

/* Define to 1 if you have the <set> header file. */
#define HAVE_SET 1

/* Define if want to build shared */
/* #undef HAVE_SHARED */

/* Define to 1 if you have the `sqrt' function. */
#define HAVE_SQRT 1

/* Define to 1 if you have the <sstream> header file. */
#define HAVE_SSTREAM 1

/* Define to 1 if you have the <stdexcept> header file. */
#define HAVE_STDEXCEPT 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* define if std::ios_base::fmtflags is supported type */
#define HAVE_STD_IOS_BASE_FMTFLAGS 

/* define if std::sprintf is supported */
#define HAVE_STD_SPRINTF 

/* define if the compiler supports Standard Template Library */
#define HAVE_STL 

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

/* Define if want to build teuchos-arprec */
/* #undef HAVE_TEUCHOS_ARPREC */

/* Define if want to build teuchos-abc */
/* #undef HAVE_TEUCHOS_ARRAY_BOUNDSCHECK */

/* Define if want to build teuchos-blasfloat */
#define HAVE_TEUCHOS_BLASFLOAT 

/* Define if want to build teuchos-boost */
/* #undef HAVE_TEUCHOS_BOOST */

/* Define if want to build teuchos-comm_timers */
#define HAVE_TEUCHOS_COMM_TIMERS 

/* Define if want to build teuchos-complex */
#define HAVE_TEUCHOS_COMPLEX 

/* Define if want to build teuchos-debug */
/* #undef HAVE_TEUCHOS_DEBUG */

/* Define if want to build teuchos-demangle */
#define HAVE_TEUCHOS_DEMANGLE 

/* Define if want to build teuchos-examples */
#define HAVE_TEUCHOS_EXAMPLES 

/* Define if want to build teuchos-expat */
/* #undef HAVE_TEUCHOS_EXPAT */

/* Define if want to build teuchos-extended */
#define HAVE_TEUCHOS_EXTENDED 

/* Define if want to build teuchos-gmp */
/* #undef HAVE_TEUCHOS_GNU_MP */

/* Define if want to build teuchos-libxml2 */
/* #undef HAVE_TEUCHOS_LIBXML2 */

/* Define if want to build teuchos-tests */
#define HAVE_TEUCHOS_TESTS 

/* Define to 1 if you have the <typeinfo> header file. */
#define HAVE_TYPEINFO 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to 1 if you have the <vector> header file. */
#define HAVE_VECTOR 1

/* template qualifier required for calling template methods from non-template
   code */
#define INVALID_TEMPLATE_QUALIFIER 

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "hkthorn@sandia.gov"

/* Define to the full name of this package. */
#define PACKAGE_NAME "teuchos"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "teuchos 1.4"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "teuchos"

/* Define to the version of this package. */
#define PACKAGE_VERSION "1.4"

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Define to `__inline__' or `__inline' if that's what the C compiler
   calls it, or to nothing if 'inline' is not supported under any name.  */
#ifndef __cplusplus
/* #undef inline */
#endif
