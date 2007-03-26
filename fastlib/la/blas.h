#ifndef LA_BLAS_H
#define LA_BLAS_H

#include "base/fortran.h"
#include "base/compiler.h"

EXTERN_C_START

f77_ret_void F77_FUNC(srot)(f77_integer CONST_REF, f77_real *, f77_integer CONST_REF, f77_real *, f77_integer CONST_REF, const f77_real *, const f77_real *);
f77_ret_void F77_FUNC(srotg)(f77_real *, f77_real *, f77_real *, f77_real *);
f77_ret_void F77_FUNC(srotm)(f77_integer CONST_REF, f77_real *, f77_integer CONST_REF, f77_real *, f77_integer CONST_REF, const f77_real *);
f77_ret_void F77_FUNC(srotmg)(f77_real *, f77_real *, f77_real *, const f77_real *, f77_real *);
f77_ret_void F77_FUNC(sswap)(f77_integer CONST_REF, f77_real *, f77_integer CONST_REF, f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(scopy)(f77_integer CONST_REF, const f77_real *, f77_integer CONST_REF, f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(saxpy)(f77_integer CONST_REF, f77_real CONST_REF, const f77_real *, f77_integer CONST_REF, f77_real *, f77_integer CONST_REF);
f77_ret_real F77_FUNC(sdot)(f77_integer CONST_REF, const f77_real *, f77_integer CONST_REF, const f77_real *, f77_integer CONST_REF);
f77_ret_real F77_FUNC(sdsdot)(f77_integer CONST_REF, const f77_real *, const f77_real *, f77_integer CONST_REF, const f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(sscal)(f77_integer CONST_REF, f77_real CONST_REF, f77_real *, f77_integer CONST_REF);
f77_ret_real F77_FUNC(snrm2)(f77_integer CONST_REF, const f77_real *, f77_integer CONST_REF);
f77_ret_real F77_FUNC(sasum)(f77_integer CONST_REF, const f77_real *, f77_integer CONST_REF);
f77_ret_integer F77_FUNC(isamax)(f77_integer CONST_REF, const f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(drot)(f77_integer CONST_REF, f77_double *, f77_integer CONST_REF, f77_double *, f77_integer CONST_REF, const f77_double *, const f77_double *);
f77_ret_void F77_FUNC(drotg)(f77_double *, f77_double *, f77_double *, f77_double *);
f77_ret_void F77_FUNC(drotm)(f77_integer CONST_REF, f77_double *, f77_integer CONST_REF, f77_double *, f77_integer CONST_REF, const f77_double *);
f77_ret_void F77_FUNC(drotmg)(f77_double *, f77_double *, f77_double *, const f77_double *, f77_double *);
f77_ret_void F77_FUNC(dswap)(f77_integer CONST_REF, f77_double *, f77_integer CONST_REF, f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(dcopy)(f77_integer CONST_REF, const f77_double *, f77_integer CONST_REF, f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(daxpy)(f77_integer CONST_REF, f77_double CONST_REF, const f77_double *, f77_integer CONST_REF, f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(dswap)(f77_integer CONST_REF, f77_double *, f77_integer CONST_REF, f77_double *, f77_integer CONST_REF);
f77_ret_double F77_FUNC(dsdot)(f77_integer CONST_REF, const f77_real *, f77_integer CONST_REF, const f77_real *, f77_integer CONST_REF);
f77_ret_double F77_FUNC(ddot)(f77_integer CONST_REF, const f77_double *, f77_integer CONST_REF, const f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(dscal)(f77_integer CONST_REF, f77_double CONST_REF, f77_double *, f77_integer CONST_REF);
f77_ret_double F77_FUNC(dnrm2)(f77_integer CONST_REF, const f77_double *, f77_integer CONST_REF);
f77_ret_double F77_FUNC(dasum)(f77_integer CONST_REF, const f77_double *, f77_integer CONST_REF);
f77_ret_integer F77_FUNC(idamax)(f77_integer CONST_REF, const f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(cswap)(f77_integer CONST_REF, f77_complex *, f77_integer CONST_REF, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(ccopy)(f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(caxpy)(f77_integer CONST_REF, f77_complex CONST_REF, const f77_complex *, f77_integer CONST_REF, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(cswap)(f77_integer CONST_REF, f77_complex *, f77_integer CONST_REF, f77_complex *, f77_integer CONST_REF);
/*f77_ret_complex F77_FUNC(cdotc)(f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF);*/
/*f77_ret_complex F77_FUNC(cdotu)(f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF);*/
f77_ret_void F77_FUNC(cscal)(f77_integer CONST_REF, f77_complex CONST_REF, f77_complex *, f77_integer CONST_REF);
f77_ret_integer F77_FUNC(icamax)(f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(csscal)(f77_integer CONST_REF, f77_real CONST_REF, f77_complex *, f77_integer CONST_REF);
f77_ret_real F77_FUNC(scnrm2)(f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF);
f77_ret_real F77_FUNC(scasum)(f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(zswap)(f77_integer CONST_REF, f77_doublecomplex *, f77_integer CONST_REF, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(zcopy)(f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(zaxpy)(f77_integer CONST_REF, f77_doublecomplex CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(zswap)(f77_integer CONST_REF, f77_doublecomplex *, f77_integer CONST_REF, f77_doublecomplex *, f77_integer CONST_REF);
/*f77_ret_doublecomplex F77_FUNC(zdotc)(f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF);*/
/*f77_ret_doublecomplex F77_FUNC(zdotu)(f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF);*/
f77_ret_void F77_FUNC(zdscal)(f77_integer CONST_REF, f77_double CONST_REF, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(zscal)(f77_integer CONST_REF, f77_doublecomplex CONST_REF, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_double F77_FUNC(dznrm2)(f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_double F77_FUNC(dzasum)(f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_integer F77_FUNC(izamax)(f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(sgemv)(const char *, f77_integer CONST_REF, f77_integer CONST_REF, f77_real CONST_REF, const f77_real *, f77_integer CONST_REF, const f77_real *, f77_integer CONST_REF, f77_real CONST_REF, f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(sgbmv)(const char *, f77_integer CONST_REF, f77_integer CONST_REF, f77_integer CONST_REF, f77_integer CONST_REF, const f77_real *, const f77_real *, f77_integer CONST_REF, const f77_real *, f77_integer CONST_REF, const f77_real *, f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(ssymv)(const char *, f77_integer CONST_REF, const f77_real *, const f77_real *, f77_integer CONST_REF, const f77_real *, f77_integer CONST_REF, const f77_real *, f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(ssbmv)(const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_real *, const f77_real *, f77_integer CONST_REF, const f77_real *, f77_integer CONST_REF, const f77_real *, f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(sspmv)(const char *, f77_integer CONST_REF, const f77_real *, const f77_real *, const f77_real *, f77_integer CONST_REF, const f77_real *, f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(strmv)(const char *, const char *, const char *, f77_integer CONST_REF, const f77_real *, f77_integer CONST_REF, f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(stbmv)(const char *, const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_real *, f77_integer CONST_REF, f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(strsv)(const char *, const char *, const char *, f77_integer CONST_REF, const f77_real *, f77_integer CONST_REF, f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(stbsv)(const char *, const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_real *, f77_integer CONST_REF, f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(stpmv)(const char *, const char *, const char *, f77_integer CONST_REF, const f77_real *, f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(stpsv)(const char *, const char *, const char *, f77_integer CONST_REF, const f77_real *, f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(sger)(f77_integer CONST_REF, f77_integer CONST_REF, const f77_real *, const f77_real *, f77_integer CONST_REF, const f77_real *, f77_integer CONST_REF, f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(ssyr)(const char *, f77_integer CONST_REF, const f77_real *, const f77_real *, f77_integer CONST_REF, f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(sspr)(const char *, f77_integer CONST_REF, const f77_real *, const f77_real *, f77_integer CONST_REF, f77_real *);
f77_ret_void F77_FUNC(sspr2)(const char *, f77_integer CONST_REF, const f77_real *, const f77_real *, f77_integer CONST_REF, const f77_real *, f77_integer CONST_REF, f77_real *);
f77_ret_void F77_FUNC(ssyr2)(const char *, f77_integer CONST_REF, const f77_real *, const f77_real *, f77_integer CONST_REF, const f77_real *, f77_integer CONST_REF, f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(dgemv)(const char *, f77_integer CONST_REF, f77_integer CONST_REF, f77_double CONST_REF, const f77_double *, f77_integer CONST_REF, const f77_double *, f77_integer CONST_REF, f77_double CONST_REF, f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(dgbmv)(const char *, f77_integer CONST_REF, f77_integer CONST_REF, f77_integer CONST_REF, f77_integer CONST_REF, const f77_double *, const f77_double *, f77_integer CONST_REF, const f77_double *, f77_integer CONST_REF, const f77_double *, f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(dsymv)(const char *, f77_integer CONST_REF, const f77_double *, const f77_double *, f77_integer CONST_REF, const f77_double *, f77_integer CONST_REF, const f77_double *, f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(dsbmv)(const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_double *, const f77_double *, f77_integer CONST_REF, const f77_double *, f77_integer CONST_REF, const f77_double *, f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(dspmv)(const char *, f77_integer CONST_REF, const f77_double *, const f77_double *, const f77_double *, f77_integer CONST_REF, const f77_double *, f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(dtrmv)(const char *, const char *, const char *, f77_integer CONST_REF, const f77_double *, f77_integer CONST_REF, f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(dtbmv)(const char *, const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_double *, f77_integer CONST_REF, f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(dtrsv)(const char *, const char *, const char *, f77_integer CONST_REF, const f77_double *, f77_integer CONST_REF, f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(dtbsv)(const char *, const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_double *, f77_integer CONST_REF, f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(dtpmv)(const char *, const char *, const char *, f77_integer CONST_REF, const f77_double *, f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(dtpsv)(const char *, const char *, const char *, f77_integer CONST_REF, const f77_double *, f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(dger)(f77_integer CONST_REF, f77_integer CONST_REF, const f77_double *, const f77_double *, f77_integer CONST_REF, const f77_double *, f77_integer CONST_REF, f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(dsyr)(const char *, f77_integer CONST_REF, const f77_double *, const f77_double *, f77_integer CONST_REF, f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(dspr)(const char *, f77_integer CONST_REF, const f77_double *, const f77_double *, f77_integer CONST_REF, f77_double *);
f77_ret_void F77_FUNC(dspr2)(const char *, f77_integer CONST_REF, const f77_double *, const f77_double *, f77_integer CONST_REF, const f77_double *, f77_integer CONST_REF, f77_double *);
f77_ret_void F77_FUNC(dsyr2)(const char *, f77_integer CONST_REF, const f77_double *, const f77_double *, f77_integer CONST_REF, const f77_double *, f77_integer CONST_REF, f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(cgemv)(const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_complex *, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(cgbmv)(const char *, f77_integer CONST_REF, f77_integer CONST_REF, f77_integer CONST_REF, f77_integer CONST_REF, const f77_complex *, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(chemv)(const char *, f77_integer CONST_REF, const f77_complex *, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(chbmv)(const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_complex *, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(chpmv)(const char *, f77_integer CONST_REF, const f77_complex *, const f77_complex *, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(ctrmv)(const char *, const char *, const char *, f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(ctbmv)(const char *, const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(ctpmv)(const char *, const char *, const char *, f77_integer CONST_REF, const f77_complex *, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(ctrsv)(const char *, const char *, const char *, f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(ctbsv)(const char *, const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(ctpsv)(const char *, const char *, const char *, f77_integer CONST_REF, const f77_complex *, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(cgerc)(f77_integer CONST_REF, f77_integer CONST_REF, const f77_complex *, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(cgeru)(f77_integer CONST_REF, f77_integer CONST_REF, const f77_complex *, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(cher)(const char *, f77_integer CONST_REF, const f77_real *, const f77_complex *, f77_integer CONST_REF, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(cher2)(const char *, f77_integer CONST_REF, const f77_complex *, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(chpr)(const char *, f77_integer CONST_REF, const f77_real *, const f77_complex *, f77_integer CONST_REF, f77_complex *);
f77_ret_void F77_FUNC(chpr2)(const char *, f77_integer CONST_REF, const f77_real *, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF, f77_complex *);
f77_ret_void F77_FUNC(zgemv)(const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_doublecomplex *, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(zgbmv)(const char *, f77_integer CONST_REF, f77_integer CONST_REF, f77_integer CONST_REF, f77_integer CONST_REF, const f77_doublecomplex *, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(zhemv)(const char *, f77_integer CONST_REF, const f77_doublecomplex *, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(zhbmv)(const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_doublecomplex *, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(zhpmv)(const char *, f77_integer CONST_REF, const f77_doublecomplex *, const f77_doublecomplex *, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(ztrmv)(const char *, const char *, const char *, f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(ztbmv)(const char *, const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(ztpmv)(const char *, const char *, const char *, f77_integer CONST_REF, const f77_doublecomplex *, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(ztrsv)(const char *, const char *, const char *, f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(ztbsv)(const char *, const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(ztpsv)(const char *, const char *, const char *, f77_integer CONST_REF, const f77_doublecomplex *, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(zgerc)(f77_integer CONST_REF, f77_integer CONST_REF, const f77_doublecomplex *, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(zgeru)(f77_integer CONST_REF, f77_integer CONST_REF, const f77_doublecomplex *, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(zher)(const char *, f77_integer CONST_REF, const f77_double *, const f77_doublecomplex *, f77_integer CONST_REF, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(zher2)(const char *, f77_integer CONST_REF, const f77_doublecomplex *, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(zhpr)(const char *, f77_integer CONST_REF, const f77_double *, const f77_doublecomplex *, f77_integer CONST_REF, f77_doublecomplex *);
f77_ret_void F77_FUNC(zhpr2)(const char *, f77_integer CONST_REF, const f77_double *, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF, f77_doublecomplex *);
f77_ret_void F77_FUNC(sgemm)(const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, f77_integer CONST_REF, f77_real CONST_REF, const f77_real *, f77_integer CONST_REF, const f77_real *, f77_integer CONST_REF, f77_real CONST_REF, f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(ssymm)(const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_real *, const f77_real *, f77_integer CONST_REF, const f77_real *, f77_integer CONST_REF, const f77_real *, f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(ssyrk)(const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_real *, const f77_real *, f77_integer CONST_REF, const f77_real *, f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(ssyr2k)(const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_real *, const f77_real *, f77_integer CONST_REF, const f77_real *, f77_integer CONST_REF, const f77_real *, f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(strmm)(const char *, const char *, const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_real *, const f77_real *, f77_integer CONST_REF, f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(strsm)(const char *, const char *, const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_real *, const f77_real *, f77_integer CONST_REF, f77_real *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(dgemm)(const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, f77_integer CONST_REF, f77_double CONST_REF, const f77_double *, f77_integer CONST_REF, const f77_double *, f77_integer CONST_REF, f77_double CONST_REF, f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(dsymm)(const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_double *, const f77_double *, f77_integer CONST_REF, const f77_double *, f77_integer CONST_REF, const f77_double *, f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(dsyrk)(const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_double *, const f77_double *, f77_integer CONST_REF, const f77_double *, f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(dsyr2k)(const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_double *, const f77_double *, f77_integer CONST_REF, const f77_double *, f77_integer CONST_REF, const f77_double *, f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(dtrmm)(const char *, const char *, const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_double *, const f77_double *, f77_integer CONST_REF, f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(dtrsm)(const char *, const char *, const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_double *, const f77_double *, f77_integer CONST_REF, f77_double *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(cgemm)(const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, f77_integer CONST_REF, f77_complex CONST_REF, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF, f77_complex CONST_REF, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(csymm)(const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_complex *, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(chemm)(const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_complex *, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(csyrk)(const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_complex *, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(cherk)(const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_complex *, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(csyr2k)(const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_complex *, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(cher2k)(const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_complex *, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_integer CONST_REF, const f77_complex *, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(ctrmm)(const char *, const char *, const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_complex *, const f77_complex *, f77_integer CONST_REF, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(ctrsm)(const char *, const char *, const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_complex *, const f77_complex *, f77_integer CONST_REF, f77_complex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(zgemm)(const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, f77_integer CONST_REF, f77_doublecomplex CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF, f77_doublecomplex CONST_REF, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(zsymm)(const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_doublecomplex *, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(zhemm)(const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_doublecomplex *, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(zsyrk)(const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_doublecomplex *, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(zherk)(const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_doublecomplex *, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(zsyr2k)(const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_doublecomplex *, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(zher2k)(const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_doublecomplex *, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_integer CONST_REF, const f77_doublecomplex *, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(ztrmm)(const char *, const char *, const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_doublecomplex *, const f77_doublecomplex *, f77_integer CONST_REF, f77_doublecomplex *, f77_integer CONST_REF);
f77_ret_void F77_FUNC(ztrsm)(const char *, const char *, const char *, const char *, f77_integer CONST_REF, f77_integer CONST_REF, const f77_doublecomplex *, const f77_doublecomplex *, f77_integer CONST_REF, f77_doublecomplex *, f77_integer CONST_REF);

EXTERN_C_END

#endif
