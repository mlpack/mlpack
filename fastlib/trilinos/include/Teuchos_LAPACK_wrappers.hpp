// @HEADER
// ***********************************************************************
// 
//                    Teuchos: Common Tools Package
//                 Copyright (2004) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//  
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//  
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ***********************************************************************
// @HEADER

#ifndef _TEUCHOS_LAPACK_WRAPPERS_HPP_
#define _TEUCHOS_LAPACK_WRAPPERS_HPP_

#include "Teuchos_ConfigDefs.hpp"

/*! \file Teuchos_LAPACK_wrappers.hpp
    \brief The Templated LAPACK wrappers
*/
/* Define fcd (Fortran Teuchos_fcd descriptor) for non-standard situations */

#if defined(CRAY_T3X) || defined(INTEL_CXML) || defined(INTEL_MKL)

#if defined(CRAY_T3X)

#include <fortran.h>
#define PREFIX
#define Teuchos_fcd fcd 

#define DGEQRF_F77  F77_FUNC(sgeqrf,SGEQRF)
#define DGETRF_F77  F77_FUNC(sgetrf,SGETRF)
#define DGETRS_F77  F77_FUNC(sgetrs,SGETRS)
#define DGTTRF_F77  F77_FUNC(sgttrf,SGTTRF)
#define DGTTRS_F77  F77_FUNC(sgttrs,SGTTRS)
#define DPTTRF_F77  F77_FUNC(spttrf,SPTTRF)
#define DPTTRS_F77  F77_FUNC(spttrs,SPTTRS)
#define DGETRI_F77  F77_FUNC(sgetri,SGETRI)
#define DGERFS_F77  F77_FUNC(sgerfs,SGERFS)
#define DGECON_F77  F77_FUNC(sgecon,SGECON)
#define DGESVX_F77  F77_FUNC(sgesvx,SGESVX)
#define DGESV_F77   F77_FUNC(sgesv,SGESV)
#define DGEEQU_F77  F77_FUNC(sgeequ,SGEEQU)
#define DSYTRD_F77  F77_FUNC(ssytrd,SSYTRD)
#define DPOTRF_F77  F77_FUNC(spotrf,SPOTRF)
#define DPOTRS_F77  F77_FUNC(spotrs,SPOTRS)
#define DPOTRI_F77  F77_FUNC(spotri,SPOTRI)
#define DPOCON_F77  F77_FUNC(spocon,SPOCON)
#define DPOSV_F77   F77_FUNC(sposv,SPOSV)
#define DPOEQU_F77  F77_FUNC(spoequ,SPOEQU)
#define DPORFS_F77  F77_FUNC(sporfs,SPORFS)
#define DPOSVX_F77  F77_FUNC(sposvx,SPOSVX)
#define DLAMCH_F77  F77_FUNC(slamch,SLAMCH)
#define DTRTRS_F77  F77_FUNC(strtrs,STRTRS)
#define DGELS_F77   F77_FUNC(sgels,SGELS)
#define DGEEV_F77   F77_FUNC(sgeev,SGEEV)
#define DGGEVX_F77  F77_FUNC(sggevx,SGGEVX)
#define DGEHRD_F77  F77_FUNC(sgehrd,SGEHRD)
#define DHSEQR_F77  F77_FUNC(shseqr,SHSEQR)
#define DORMQR_F77  F77_FUNC(sormqr,SORMQR)
#define DORGQR_F77  F77_FUNC(sorgqr,SORGQR)
#define DORGHR_F77  F77_FUNC(sorghr,SORGHR)
#define DORMHR_F77  F77_FUNC(sormhr,SORMHR)
#define DTREVC_F77  F77_FUNC(strevc,STREVC)
#define DTREXC_F77  F77_FUNC(strexc,STREXC)
#define DGEES_F77   F77_FUNC(sgees,SGEES)
#define DSPEV_F77   F77_FUNC(sspev,SSPEV)
#define DSYEV_F77   F77_FUNC(ssyev,SSYEV)
#define DSYGV_F77   F77_FUNC(ssygv,SSYGV)
#define DSTEQR_F77  F77_FUNC(ssteqr,SSTEQR)
#define DLAPY2_F77  F77_FUNC(slapy2,SLAPY2)
#define DLARTG_F77  F77_FUNC(slartg,SLARTG)
#define DLARFG_F77  F77_FUNC(slarfg,SLARFG)
#define DLARND_F77  F77_FUNC(slarnd,SLARND)
#define DLARNV_F77  F77_FUNC(slarnv,SLARNV)
#define ILAENV_F77  F77_FUNC(ilaenv,ILAENV)

#ifdef HAVE_TEUCHOS_COMPLEX

#define ZGEQRF_F77  F77_FUNC(cgeqrf,CGEQRF)
#define ZUNGQR_F77  F77_FUNC(cungqr,CUNGQR)
#define ZGETRF_F77  F77_FUNC(cgetrf,CGETRF)
#define ZGETRS_F77  F77_FUNC(cgetrs,CGETRS)
#define ZGTTRF_F77  F77_FUNC(cgttrf,CGTTRF)
#define ZGTTRS_F77  F77_FUNC(cgttrs,CGTTRS)
#define ZPTTRF_F77  F77_FUNC(cpttrf,CPTTRF)
#define ZPTTRS_F77  F77_FUNC(cpttrs,CPTTRS)
#define ZGETRI_F77  F77_FUNC(cgetri,CGETRI)
#define ZGERFS_F77  F77_FUNC(cgerfs,CGERFS)
#define ZGECON_F77  F77_FUNC(cgecon,CGECON)
#define ZGESVX_F77  F77_FUNC(cgesvx,CGESVX)
#define ZGESV_F77   F77_FUNC(cgesv,CGESV)
#define ZGEEQU_F77  F77_FUNC(cgeequ,CGEEQU)
#define ZPOTRF_F77  F77_FUNC(cpotrf,CPOTRF)
#define ZPOTRS_F77  F77_FUNC(cpotrs,CPOTRS)
#define ZPOTRI_F77  F77_FUNC(cpotri,CPOTRI)
#define ZPOCON_F77  F77_FUNC(cpocon,CPOCON)
#define ZPOSV_F77   F77_FUNC(cposv,CPOSV)
#define ZPOEQU_F77  F77_FUNC(cpoequ,CPOEQU)
#define ZPORFS_F77  F77_FUNC(cporfs,CPORFS)
#define ZPOSVX_F77  F77_FUNC(cposvx,CPOSVX)
#define ZTRTRS_F77  F77_FUNC(ctrtrs,CTRTRS)
#define ZGELS_F77   F77_FUNC(cgels,CGELS)
#define ZGEEV_F77   F77_FUNC(cgeev,CGEEV)
//#define ZGGEVX_F77  F77_FUNC(cggevx,CGGEVX)
#define ZGEHRD_F77  F77_FUNC(cgehrd,CGEHRD)
#define ZHSEQR_F77  F77_FUNC(chseqr,CHSEQR)
#define ZTREVC_F77  F77_FUNC(ctrevc,CTREVC)
#define ZTREXC_F77  F77_FUNC(ctrexc,CTREXC)
#define ZGEES_F77   F77_FUNC(cgees,CGEES)
#define ZSTEQR_F77  F77_FUNC(csteqr,CSTEQR)
#define ZHEEV_F77   F77_FUNC(cheev,CHEEV)
#define ZHEGV_F77   F77_FUNC(chegv,CHEGV)
#define ZLARTG_F77  F77_FUNC(clartg,cLARTG)
#define ZLARFG_F77  F77_FUNC(clarfg,cLARFG)
#define ZLARND_F77  F77_FUNC(clarnd,CLARND)
#define ZLARNV_F77  F77_FUNC(clarnv,CLARNV)

#endif /* HAVE_TEUCHOS_COMPLEX */

#elif defined(INTEL_CXML)

#define PREFIX __stdcall 
#define Teuchos_fcd const char *, unsigned int 

#define DGEQRF_F77  F77_FUNC(dgeqrf,DGEQRF)
#define DGETRF_F77  F77_FUNC(dgetrf,DGETRF)
#define DGETRS_F77  F77_FUNC(dgetrs,DGETRS)
#define DGTTRF_F77  F77_FUNC(dgttrf,DGTTRF)
#define DGTTRS_F77  F77_FUNC(dgttrs,DGTTRS)
#define DPTTRF_F77  F77_FUNC(dpttrf,DPTTRF)
#define DPTTRS_F77  F77_FUNC(dpttrs,DPTTRS)
#define DGETRI_F77  F77_FUNC(dgetri,DGETRI)
#define DGERFS_F77  F77_FUNC(dgerfs,DGERFS)
#define DGECON_F77  F77_FUNC(dgecon,DGECON)
#define DGESVX_F77  F77_FUNC(dgesvx,DGESVX)
#define DGESV_F77   F77_FUNC(dgesv,DGESV)
#define DGEEQU_F77  F77_FUNC(dgeequ,DGEEQU)
#define DSYTRD_F77  F77_FUNC(dsytrd,DSYTRD)
#define DPOTRF_F77  F77_FUNC(dpotrf,DPOTRF)
#define DPOTRS_F77  F77_FUNC(dpotrs,DPOTRS)
#define DPOTRI_F77  F77_FUNC(dpotri,DPOTRI)
#define DPOCON_F77  F77_FUNC(dpocon,DPOCON)
#define DPOSV_F77   F77_FUNC(dposv,DPOSV)
#define DPOEQU_F77  F77_FUNC(dpoequ,DPOEQU)
#define DPORFS_F77  F77_FUNC(dporfs,DPORFS)
#define DPOSVX_F77  F77_FUNC(dposvx,DPOSVX)
#define DTRTRS_F77  F77_FUNC(dtrtrs,DTRTRS)
#define DLAMCH_F77  F77_FUNC(dlamch,DLAMCH)
#define DGELS_F77   F77_FUNC(dgels,DGELS)
#define DGEEV_F77   F77_FUNC(dgeev,DGEEV)
#define DGGEVX_F77  F77_FUNC(dggevx,DGGEVX)
#define DGEHRD_F77  F77_FUNC(dgehrd,DGEHRD)
#define DHSEQR_F77  F77_FUNC(dhseqr,DHSEQR)
#define DORGHR_F77  F77_FUNC(dorghr,DORGHR)
#define DORGQR_F77  F77_FUNC(dorgqr,DORGQR)
#define DORMHR_F77  F77_FUNC(dormhr,DORMHR)
#define DORMQR_F77  F77_FUNC(dormqr,DORMQR)
#define DTREVC_F77  F77_FUNC(dtrevc,DTREVC)
#define DTREXC_F77  F77_FUNC(dtrexc,DTREXC)
#define DGEES_F77   F77_FUNC(dgees,DGEES)
#define DSPEV_F77   F77_FUNC(dspev,DSPEV)
#define DSYEV_F77   F77_FUNC(dsyev,DSYEV)
#define DSYGV_F77   F77_FUNC(dsygv,DSYGV)
#define DSTEQR_F77  F77_FUNC(dsteqr,DSTEQR)
#define DLAPY2_F77  F77_FUNC(dlapy2,DLAPY2)
#define DLARTG_F77  F77_FUNC(dlartg,DLARTG)
#define DLARFG_F77  F77_FUNC(dlarfg,DLARFG)
#define DLARND_F77  F77_FUNC(dlarnd,DLARND)
#define DLARNV_F77  F77_FUNC(dlarnv,DLARNV)
#define ILAENV_F77  F77_FUNC(ilaenv,ILAENV)

#ifdef HAVE_TEUCHOS_COMPLEX

#define ZGEQRF_F77  F77_FUNC(zgeqrf,ZGEQRF)
#define ZUNGQR_F77  F77_FUNC(zungqr,ZUNGQR)
#define ZGETRF_F77  F77_FUNC(zgetrf,ZGETRF)
#define ZGETRS_F77  F77_FUNC(zgetrs,ZGETRS)
#define ZGTTRF_F77  F77_FUNC(zgttrf,ZGTTRF)
#define ZGTTRS_F77  F77_FUNC(zgttrs,ZGTTRS)
#define ZPTTRF_F77  F77_FUNC(zpttrf,ZPTTRF)
#define ZPTTRS_F77  F77_FUNC(zpttrs,ZPTTRS)
#define ZGETRI_F77  F77_FUNC(zgetri,ZGETRI)
#define ZGERFS_F77  F77_FUNC(zgerfs,ZGERFS)
#define ZGECON_F77  F77_FUNC(zgecon,ZGECON)
#define ZGESVX_F77  F77_FUNC(zgesvx,ZGESVX)
#define ZGESV_F77   F77_FUNC(zgesv,ZGESV)
#define ZGEEQU_F77  F77_FUNC(zgeequ,ZGEEQU)
#define ZPOTRF_F77  F77_FUNC(zpotrf,ZPOTRF)
#define ZPOTRS_F77  F77_FUNC(zpotrs,ZPOTRS)
#define ZPOTRI_F77  F77_FUNC(zpotri,ZPOTRI)
#define ZPOCON_F77  F77_FUNC(zpocon,ZPOCON)
#define ZPOSV_F77   F77_FUNC(zposv,ZPOSV)
#define ZPOEQU_F77  F77_FUNC(zpoequ,ZPOEQU)
#define ZPORFS_F77  F77_FUNC(zporfs,ZPORFS)
#define ZPOSVX_F77  F77_FUNC(zposvx,ZPOSVX)
#define ZTRTRS_F77  F77_FUNC(ztrtrs,ZTRTRS)
#define ZGELS_F77   F77_FUNC(zgels,ZGELS)
#define ZGEEV_F77   F77_FUNC(zgeev,ZGEEV)
//#define ZGGEVX_F77  F77_FUNC(zggevx,ZGGEVX)
#define ZGEHRD_F77  F77_FUNC(zgehrd,ZGEHRD)
#define ZHSEQR_F77  F77_FUNC(zhseqr,ZHSEQR)
#define ZTREVC_F77  F77_FUNC(ztrevc,ZTREVC)
#define ZTREXC_F77  F77_FUNC(ztrexc,ZTREXC)
#define ZGEES_F77   F77_FUNC(zgees,ZGEES)
#define ZSTEQR_F77  F77_FUNC(zsteqr,ZSTEQR)
#define ZHEEV_F77   F77_FUNC(zheev,ZHEEV)
#define ZHEGV_F77   F77_FUNC(zhegv,ZHEGV)
#define ZLARTG_F77  F77_FUNC(zlartg,ZLARTG)
#define ZLARFG_F77  F77_FUNC(zlarfg,ZLARFG)
#define ZLARND_F77  F77_FUNC(zlarnd,ZLARND)
#define ZLARNV_F77  F77_FUNC(zlarnv,ZLARNV)

#endif /* HAVE_TEUCHOS_COMPLEX */

#elif defined(INTEL_MKL)

#define PREFIX
#define Teuchos_fcd const char *

#define DGEQRF_F77  F77_FUNC(dgeqrf,DGEQRF)
#define DGETRF_F77  F77_FUNC(dgetrf,DGETRF)
#define DGETRS_F77  F77_FUNC(dgetrs,DGETRS)
#define DGTTRF_F77  F77_FUNC(dgttrf,DGTTRF)
#define DGTTRS_F77  F77_FUNC(dgttrs,DGTTRS)
#define DPTTRF_F77  F77_FUNC(dpttrf,DPTTRF)
#define DPTTRS_F77  F77_FUNC(dpttrs,DPTTRS)
#define DGETRI_F77  F77_FUNC(dgetri,DGETRI)
#define DGERFS_F77  F77_FUNC(dgerfs,DGERFS)
#define DGECON_F77  F77_FUNC(dgecon,DGECON)
#define DGESVX_F77  F77_FUNC(dgesvx,DGESVX)
#define DGESV_F77   F77_FUNC(dgesv,DGESV)
#define DGEEQU_F77  F77_FUNC(dgeequ,DGEEQU)
#define DSYTRD_F77  F77_FUNC(dsytrd,DSYTRD)
#define DPOTRF_F77  F77_FUNC(dpotrf,DPOTRF)
#define DPOTRS_F77  F77_FUNC(dpotrs,DPOTRS)
#define DPOTRI_F77  F77_FUNC(dpotri,DPOTRI)
#define DPOCON_F77  F77_FUNC(dpocon,DPOCON)
#define DPOSV_F77   F77_FUNC(dposv,DPOSV)
#define DPOEQU_F77  F77_FUNC(dpoequ,DPOEQU)
#define DPORFS_F77  F77_FUNC(dporfs,DPORFS)
#define DPOSVX_F77  F77_FUNC(dposvx,DPOSVX)
#define DTRTRS_F77  F77_FUNC(dtrtrs,DTRTRS)
#define DLAMCH_F77  F77_FUNC(dlamch,DLAMCH)
#define DGELS_F77   F77_FUNC(dgels,DGELS)
#define DGEEV_F77   F77_FUNC(dgeev,DGEEV)
#define DGGEVX_F77  F77_FUNC(dggevx,DGGEVX)
#define DGEHRD_F77  F77_FUNC(dgehrd,DGEHRD)
#define DHSEQR_F77  F77_FUNC(dhseqr,DHSEQR)
#define DORGHR_F77  F77_FUNC(dorghr,DORGHR)
#define DORGQR_F77  F77_FUNC(dorgqr,DORGQR)
#define DORMHR_F77  F77_FUNC(dormhr,DORMHR)
#define DORMQR_F77  F77_FUNC(dormqr,DORMQR)
#define DTREVC_F77  F77_FUNC(dtrevc,DTREVC)
#define DTREXC_F77  F77_FUNC(dtrexc,DTREXC)
#define DGEES_F77   F77_FUNC(dgees,DGEES)
#define DSPEV_F77   F77_FUNC(dspev,DSPEV)
#define DSYEV_F77   F77_FUNC(dsyev,DSYEV)
#define DSYGV_F77   F77_FUNC(dsygv,DSYGV)
#define DSTEQR_F77  F77_FUNC(dsteqr,DSTEQR)
#define DLAPY2_F77  F77_FUNC(dlapy2,DLAPY2)
#define DLARTG_F77  F77_FUNC(dlartg,DLARTG)
#define DLARFG_F77  F77_FUNC(dlarfg,DLARFG)
#define DLARND_F77  F77_FUNC(dlarnd,DLARND)
#define DLARNV_F77  F77_FUNC(dlarnv,DLARNV)
#define ILAENV_F77  F77_FUNC(ilaenv,ILAENV)

#ifdef HAVE_TEUCHOS_COMPLEX

#define ZGEQRF_F77  F77_FUNC(zgeqrf,ZGEQRF)
#define ZUNGQR_F77  F77_FUNC(zungqr,ZUNGQR)
#define ZGTTRF_F77  F77_FUNC(zgttrf,ZGTTRF)
#define ZGTTRS_F77  F77_FUNC(zgttrs,ZGTTRS)
#define ZPTTRF_F77  F77_FUNC(zpttrf,ZPTTRF)
#define ZPTTRS_F77  F77_FUNC(zpttrs,ZPTTRS)
#define ZGETRF_F77  F77_FUNC(zgetrf,ZGETRF)
#define ZGETRS_F77  F77_FUNC(zgetrs,ZGETRS)
#define ZGETRI_F77  F77_FUNC(zgetri,ZGETRI)
#define ZGERFS_F77  F77_FUNC(zgerfs,ZGERFS)
#define ZGECON_F77  F77_FUNC(zgecon,ZGECON)
#define ZGESVX_F77  F77_FUNC(zgesvx,ZGESVX)
#define ZGESV_F77   F77_FUNC(zgesv,ZGESV)
#define ZGEEQU_F77  F77_FUNC(zgeequ,ZGEEQU)
#define ZPOTRF_F77  F77_FUNC(zpotrf,ZPOTRF)
#define ZPOTRS_F77  F77_FUNC(zpotrs,ZPOTRS)
#define ZPOTRI_F77  F77_FUNC(zpotri,ZPOTRI)
#define ZPOCON_F77  F77_FUNC(zpocon,ZPOCON)
#define ZPOSV_F77   F77_FUNC(zposv,ZPOSV)
#define ZPOEQU_F77  F77_FUNC(zpoequ,ZPOEQU)
#define ZPORFS_F77  F77_FUNC(zporfs,ZPORFS)
#define ZPOSVX_F77  F77_FUNC(zposvx,ZPOSVX)
#define ZTRTRS_F77  F77_FUNC(ztrtrs,ZTRTRS)
#define ZGELS_F77   F77_FUNC(zgels,ZGELS)
#define ZGEEV_F77   F77_FUNC(zgeev,ZGEEV)
//#define ZGGEVX_F77  F77_FUNC(zggevx,ZGGEVX)
#define ZGEHRD_F77  F77_FUNC(zgehrd,ZGEHRD)
#define ZHSEQR_F77  F77_FUNC(zhseqr,ZHSEQR)
#define ZTREVC_F77  F77_FUNC(ztrevc,ZTREVC)
#define ZTREXC_F77  F77_FUNC(ztrexc,ZTREXC)
#define ZGEES_F77   F77_FUNC(zgees,ZGEES)
#define ZSTEQR_F77  F77_FUNC(zsteqr,ZSTEQR)
#define ZHEEV_F77   F77_FUNC(zheev,ZHEEV)
#define ZHEGV_F77   F77_FUNC(zhegv,ZHEGV)
#define ZLARTG_F77  F77_FUNC(zlartg,ZLARTG)
#define ZLARFG_F77  F77_FUNC(zlarfg,ZLARFG)
#define ZLARND_F77  F77_FUNC(zlarnd,ZLARND)
#define ZLARNV_F77  F77_FUNC(zlarnv,ZLARNV)

#endif /* HAVE_TEUCHOS_COMPLEX */

#endif /* defined(CRAY_T3X) || defined(INTEL_CXML) || defined(INTEL_MKL) */

/* All three of these machines use a simple uppercase mangling of Fortran names */

/* if F77_FUNC is defined undefine it because we want to redefine */

#ifdef F77_FUNC
#undef F77_FUNC
#endif

#define F77_FUNC(lcase,UCASE) PREFIX UCASE

#else /* Define Teuchos_fcd for all other machines */

#define PREFIX
#define Teuchos_fcd const char * 

#ifndef HAVE_CONFIG_H

#ifdef F77_FUNC
#undef F77_FUNC
#endif

#ifdef TRILINOS_HAVE_NO_FORTRAN_UNDERSCORE
#define F77_FUNC(lcase,UCASE) lcase
#else /* TRILINOS_HAVE_NO_FORTRAN_UNDERSCORE not defined*/
#define F77_FUNC(lcase,UCASE) lcase ## _
#endif /* TRILINOS_HAVE_NO_FORTRAN_UNDERSCORE */

#endif /* HAVE_CONFIG_H */

#define DGEQRF_F77  F77_FUNC(dgeqrf,DGEQRF)
#define DGETRF_F77  F77_FUNC(dgetrf,DGETRF)
#define DGETRS_F77  F77_FUNC(dgetrs,DGETRS)
#define DGTTRF_F77  F77_FUNC(dgttrf,DGTTRF)
#define DGTTRS_F77  F77_FUNC(dgttrs,DGTTRS)
#define DPTTRF_F77  F77_FUNC(dpttrf,DPTTRF)
#define DPTTRS_F77  F77_FUNC(dpttrs,DPTTRS)
#define DGETRI_F77  F77_FUNC(dgetri,DGETRI)
#define DGERFS_F77  F77_FUNC(dgerfs,DGERFS)
#define DGECON_F77  F77_FUNC(dgecon,DGECON)
#define DGESVX_F77  F77_FUNC(dgesvx,DGESVX)
#define DGESV_F77   F77_FUNC(dgesv,DGESV)
#define DGEEQU_F77  F77_FUNC(dgeequ,DGEEQU)
#define DSYTRD_F77  F77_FUNC(dsytrd,DSYTRD)
#define DPOTRF_F77  F77_FUNC(dpotrf,DPOTRF)
#define DPOTRS_F77  F77_FUNC(dpotrs,DPOTRS)
#define DPOTRI_F77  F77_FUNC(dpotri,DPOTRI)
#define DPOCON_F77  F77_FUNC(dpocon,DPOCON)
#define DPOSV_F77   F77_FUNC(dposv,DPOSV)
#define DPOEQU_F77  F77_FUNC(dpoequ,DPOEQU)
#define DPORFS_F77  F77_FUNC(dporfs,DPORFS)
#define DPOSVX_F77  F77_FUNC(dposvx,DPOSVX)
#define DTRTRS_F77  F77_FUNC(dtrtrs,DTRTRS)
#define DLAMCH_F77  F77_FUNC(dlamch,DLAMCH)
#define DGELS_F77   F77_FUNC(dgels,DGELS)
#define DGEEV_F77   F77_FUNC(dgeev,DGEEV)
#define DGGEVX_F77  F77_FUNC(dggevx,DGGEVX)
#define DGEHRD_F77  F77_FUNC(dgehrd,DGEHRD)
#define DHSEQR_F77  F77_FUNC(dhseqr,DHSEQR)
#define DORGHR_F77  F77_FUNC(dorghr,DORGHR)
#define DORGQR_F77  F77_FUNC(dorgqr,DORGQR)
#define DORMHR_F77  F77_FUNC(dormhr,DORMHR)
#define DORMQR_F77  F77_FUNC(dormqr,DORMQR)
#define DTREVC_F77  F77_FUNC(dtrevc,DTREVC)
#define DTREXC_F77  F77_FUNC(dtrexc,DTREXC)
#define DGEES_F77   F77_FUNC(dgees,DGEES)
#define DSPEV_F77   F77_FUNC(dspev,DSPEV)
#define DSYEV_F77   F77_FUNC(dsyev,DSYEV)
#define DSYGV_F77   F77_FUNC(dsygv,DSYGV)
#define DSTEQR_F77  F77_FUNC(dsteqr,DSTEQR)
#define DLAPY2_F77  F77_FUNC(dlapy2,DLAPY2)
#define DLARND_F77  F77_FUNC(dlarnd,DLARND)
#define DLARNV_F77  F77_FUNC(dlarnv,DLARNV)
#define DLARTG_F77  F77_FUNC(dlartg,DLARTG)
#define DLARFG_F77  F77_FUNC(dlarfg,DLARFG)
#define ILAENV_F77  F77_FUNC(ilaenv,ILAENV)

#ifdef HAVE_TEUCHOS_COMPLEX

#define ZGEQRF_F77  F77_FUNC(zgeqrf,ZGEQRF)
#define ZUNGQR_F77  F77_FUNC(zungqr,ZUNGQR)
#define ZGETRF_F77  F77_FUNC(zgetrf,ZGETRF)
#define ZGETRS_F77  F77_FUNC(zgetrs,ZGETRS)
#define ZGTTRF_F77  F77_FUNC(zgttrf,ZGTTRF)
#define ZGTTRS_F77  F77_FUNC(zgttrs,ZGTTRS)
#define ZPTTRF_F77  F77_FUNC(zpttrf,ZPTTRF)
#define ZPTTRS_F77  F77_FUNC(zpttrs,ZPTTRS)
#define ZGETRI_F77  F77_FUNC(zgetri,ZGETRI)
#define ZGERFS_F77  F77_FUNC(zgerfs,ZGERFS)
#define ZGECON_F77  F77_FUNC(zgecon,ZGECON)
#define ZGESVX_F77  F77_FUNC(zgesvx,ZGESVX)
#define ZGESV_F77   F77_FUNC(zgesv,ZGESV)
#define ZGEEQU_F77  F77_FUNC(zgeequ,ZGEEQU)
#define ZPOTRF_F77  F77_FUNC(zpotrf,ZPOTRF)
#define ZPOTRS_F77  F77_FUNC(zpotrs,ZPOTRS)
#define ZPOTRI_F77  F77_FUNC(zpotri,ZPOTRI)
#define ZPOCON_F77  F77_FUNC(zpocon,ZPOCON)
#define ZPOSV_F77   F77_FUNC(zposv,ZPOSV)
#define ZPOEQU_F77  F77_FUNC(zpoequ,ZPOEQU)
#define ZPORFS_F77  F77_FUNC(zporfs,ZPORFS)
#define ZPOSVX_F77  F77_FUNC(zposvx,ZPOSVX)
#define ZTRTRS_F77  F77_FUNC(ztrtrs,ZTRTRS)
#define ZGELS_F77   F77_FUNC(zgels,ZGELS)
#define ZGEEV_F77   F77_FUNC(zgeev,ZGEEV)
//#define ZGGEVX_F77  F77_FUNC(zggevx,ZGGEVX)
#define ZGEHRD_F77  F77_FUNC(zgehrd,ZGEHRD)
#define ZHSEQR_F77  F77_FUNC(zhseqr,ZHSEQR)
#define ZTREVC_F77  F77_FUNC(ztrevc,ZTREVC)
#define ZTREXC_F77  F77_FUNC(ztrexc,ZTREXC)
#define ZGEES_F77   F77_FUNC(zgees,ZGEES)
#define ZSTEQR_F77  F77_FUNC(zsteqr,ZSTEQR)
#define ZHEEV_F77   F77_FUNC(zheev,ZHEEV)
#define ZHEGV_F77   F77_FUNC(zhegv,ZHEGV)
#define ZLARTG_F77  F77_FUNC(zlartg,ZLARTG)
#define ZLARFG_F77  F77_FUNC(zlarfg,ZLARFG)
#define ZLARND_F77  F77_FUNC(zlarnd,ZLARND)
#define ZLARNV_F77  F77_FUNC(zlarnv,ZLARNV)

#endif /* HAVE_TEUCHOS_COMPLEX */

#endif

#define SGEQRF_F77  F77_FUNC(sgeqrf,SGEQRF)
#define SGETRF_F77  F77_FUNC(sgetrf,SGETRF)
#define SGETRS_F77  F77_FUNC(sgetrs,SGETRS)
#define SGTTRF_F77  F77_FUNC(sgttrf,SGTTRF)
#define SGTTRS_F77  F77_FUNC(sgttrs,SGTTRS)
#define SPTTRF_F77  F77_FUNC(spttrf,SPTTRF)
#define SPTTRS_F77  F77_FUNC(spttrs,SPTTRS)
#define SGETRI_F77  F77_FUNC(sgetri,SGETRI)
#define SGERFS_F77  F77_FUNC(sgerfs,SGERFS)
#define SGECON_F77  F77_FUNC(sgecon,SGECON)
#define SGESVX_F77  F77_FUNC(sgesvx,SGESVX)
#define SGESV_F77   F77_FUNC(sgesv,SGESV)
#define SGEEQU_F77  F77_FUNC(sgeequ,SGEEQU)
#define SSYTRD_F77  F77_FUNC(ssytrd,SSYTRD)
#define SPOTRF_F77  F77_FUNC(spotrf,SPOTRF)
#define SPOTRS_F77  F77_FUNC(spotrs,SPOTRS)
#define SPOTRI_F77  F77_FUNC(spotri,SPOTRI)
#define SPOCON_F77  F77_FUNC(spocon,SPOCON)
#define SPOSV_F77   F77_FUNC(sposv,SPOSV)
#define SPOEQU_F77  F77_FUNC(spoequ,SPOEQU)
#define SPORFS_F77  F77_FUNC(sporfs,SPORFS)
#define SPOSVX_F77  F77_FUNC(sposvx,SPOSVX)
#define STRTRS_F77  F77_FUNC(strtrs,STRTRS)
#define SGELS_F77   F77_FUNC(sgels,SGELS)
#define SGEEV_F77   F77_FUNC(sgeev,SGEEV)
#define SGGEVX_F77  F77_FUNC(sggevx,SGGEVX)
#define SGEHRD_F77  F77_FUNC(sgehrd,SGEHRD)
#define SHSEQR_F77  F77_FUNC(shseqr,SHSEQR)
#define SORGHR_F77  F77_FUNC(sorghr,SORGHR)
#define SORGQR_F77  F77_FUNC(sorgqr,SORGQR)
#define SORMHR_F77  F77_FUNC(sormhr,SORMHR)
#define SORMQR_F77  F77_FUNC(sormqr,SORMQR)
#define STREVC_F77  F77_FUNC(strevc,STREVC)
#define STREXC_F77  F77_FUNC(strexc,STREXC)
#define SLAMCH_F77  F77_FUNC(slamch,SLAMCH)
#define SGEES_F77   F77_FUNC(sgees,SGEES)
#define SSPEV_F77   F77_FUNC(sspev,SSPEV)
#define SSYEV_F77   F77_FUNC(ssyev,SSYEV)
#define SSYGV_F77   F77_FUNC(ssygv,SSYGV)
#define SSTEQR_F77  F77_FUNC(ssteqr,SSTEQR)
#define SLAPY2_F77  F77_FUNC(slapy2,SLAPY2)
#define SLARTG_F77  F77_FUNC(slartg,SLARTG)
#define SLARFG_F77  F77_FUNC(slarfg,SLARFG)
#define SLARND_F77  F77_FUNC(slarnd,SLARND)
#define SLARNV_F77  F77_FUNC(slarnv,SLARNV)

#ifdef HAVE_TEUCHOS_COMPLEX

#define CGEQRF_F77  F77_FUNC(cgeqrf,CGEQRF)
#define CUNGQR_F77  F77_FUNC(cungqr,CUNGQR)
#define CGETRF_F77  F77_FUNC(cgetrf,CGETRF)
#define CGETRS_F77  F77_FUNC(cgetrs,CGETRS)
#define CGTTRF_F77  F77_FUNC(cgttrf,CGTTRF)
#define CGTTRS_F77  F77_FUNC(cgttrs,CGTTRS)
#define CPTTRF_F77  F77_FUNC(cpttrf,CPTTRF)
#define CPTTRS_F77  F77_FUNC(cpttrs,CPTTRS)
#define CGETRI_F77  F77_FUNC(cgetri,CGETRI)
#define CGERFS_F77  F77_FUNC(cgerfs,CGERFS)
#define CGECON_F77  F77_FUNC(cgecon,CGECON)
#define CGESVX_F77  F77_FUNC(cgesvx,CGESVX)
#define CGESV_F77   F77_FUNC(cgesv,CGESV)
#define CGEEQU_F77  F77_FUNC(cgeequ,CGEEQU)
#define CPOTRF_F77  F77_FUNC(cpotrf,CPOTRF)
#define CPOTRS_F77  F77_FUNC(cpotrs,CPOTRS)
#define CPOTRI_F77  F77_FUNC(cpotri,CPOTRI)
#define CPOCON_F77  F77_FUNC(cpocon,CPOCON)
#define CPOSV_F77   F77_FUNC(cposv,CPOSV)
#define CPOEQU_F77  F77_FUNC(cpoequ,CPOEQU)
#define CPORFS_F77  F77_FUNC(cporfs,CPORFS)
#define CPOSVX_F77  F77_FUNC(cposvx,CPOSVX)
#define CTRTRS_F77  F77_FUNC(ctrtrs,CTRTRS)
#define CGELS_F77   F77_FUNC(cgels,CGELS)
#define CGEEV_F77   F77_FUNC(cgeev,CGEEV)
//#define CGGEVX_F77  F77_FUNC(cggevx,CGGEVX)
#define CGEHRD_F77  F77_FUNC(cgehrd,CGEHRD)
#define CHSEQR_F77  F77_FUNC(chseqr,CHSEQR)
#define CTREVC_F77  F77_FUNC(ctrevc,CTREVC)
#define CTREXC_F77  F77_FUNC(ctrexc,CTREXC)
#define CGEES_F77   F77_FUNC(cgees,CGEES)
#define CSTEQR_F77  F77_FUNC(csteqr,CSTEQR)
#define CHEEV_F77   F77_FUNC(cheev,CHEEV)
#define CHEGV_F77   F77_FUNC(chegv,CHEGV)
#define CLARTG_F77  F77_FUNC(clartg,CLARTG)
#define CLARFG_F77  F77_FUNC(clarfg,CLARFG)
#define CLARND_F77  F77_FUNC(clarnd,CLARND)
#define CLARNV_F77  F77_FUNC(clarnv,CLARNV)

#endif /* HAVE_TEUCHOS_COMPLEX */

#ifdef __cplusplus
extern "C" {
#endif

// Double precision LAPACK linear solvers
void PREFIX DGELS_F77(Teuchos_fcd ch, const int* m, const int* n, const int* nrhs, double* a, const int* lda, double* b, const int* ldb, double* work, const int* lwork, int* info);
void PREFIX DGEQRF_F77(const int* m, const int* n, double* a, const int* lda, double* tau, double* work, const int* lwork, int* info);
void PREFIX DGETRF_F77(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info); 
void PREFIX DGETRS_F77(Teuchos_fcd, const int* n, const int* nrhs, const double* a, const int* lda,const int* ipiv, double* x , const int* ldx, int* info);
void PREFIX DGTTRF_F77(const int* n, double* dl, double* d, double* du, double* du2, int* ipiv, int* info); 
void PREFIX DGTTRS_F77(Teuchos_fcd, const int* n, const int* nrhs, const double* dl, const double* d, const double* du, const double* du2, const int* ipiv, double* x , const int* ldx, int* info);
void PREFIX DPTTRF_F77(const int* n, double* d, double* e, int* info); 
void PREFIX DPTTRS_F77(const int* n, const int* nrhs, const double* d, const double* e, double* x , const int* ldx, int* info);
void PREFIX DGETRI_F77(const int* n, double* a, const int* lda, const int* ipiv, double* work , const int* lwork, int* info);
void PREFIX DGECON_F77(Teuchos_fcd norm, const int* n, const double* a, const int* lda, const double* anorm, double* rcond, double* work, int* iwork, int* info); 
void PREFIX DGESV_F77(const int* n, const int* nrhs, double* a, const int* lda, int* ipiv, double* x , const int* ldx, int* info);
void PREFIX DGEEQU_F77(const int* m, const int* n, const double* a, const int* lda, double* r, double* c, double* rowcnd, double* colcnd, double* amax, int* info); 
void PREFIX DGERFS_F77(Teuchos_fcd, const int* n, const int* nrhs, const double* a, const int* lda, const double* af, const int* ldaf, const int* ipiv, const double* b, const int* ldb, double* x, const int* ldx, double* ferr, double* berr, double* work, int* iwork, int* info);
void PREFIX DGESVX_F77(Teuchos_fcd, Teuchos_fcd, const int* n, const int* nrhs, double* a, const int* lda, double* af, const int* ldaf, int* ipiv, Teuchos_fcd, double* r,
double* c, double* b, const int* ldb, double* x, const int* ldx, double* rcond, double* ferr, double* berr, double* work, int* iwork, int* info);
void PREFIX DSYTRD_F77(Teuchos_fcd, const int* n, double* a, const int* lda, double* D, double* E, double* tau, double* work, const int* lwork, int* info);
void PREFIX DPOTRF_F77(Teuchos_fcd, const int* n, double* a, const int* lda, int* info); 
void PREFIX DPOTRS_F77(Teuchos_fcd, const int* n, const int* nrhs, const double* a, const int* lda, double*x , const int* ldx, int* info);
void PREFIX DPOTRI_F77(Teuchos_fcd, const int* n, double* a, const int* lda, int* info); 
void PREFIX DPOCON_F77(Teuchos_fcd, const int* n, const double* a, const int* lda, const double* anorm, double* rcond, double* work, int* iwork, int* info); 
void PREFIX DPOSV_F77(Teuchos_fcd, const int* n, const int* nrhs, double* a, const int* lda, double*x , const int* ldx, int* info);
void PREFIX DPOEQU_F77(const int* n, const double* a, const int* lda, double* s, double* scond, double* amax, int* info); 
void PREFIX DPORFS_F77(Teuchos_fcd, const int* n, const int* nrhs, double* a, const int* lda, const double* af, const int* ldaf, const double* b, const int* ldb, double* x, const int* ldx, double* ferr, double* berr, double* work, int* iwork, int* info);
void PREFIX DPOSVX_F77(Teuchos_fcd, Teuchos_fcd, const int* n, const int* nrhs, double* a, const int* lda, double* af, const int* ldaf, Teuchos_fcd, double* s, double* b, const int* ldb, double* x, const int* ldx, double* rcond, double* ferr, double* berr, double* work, int* iwork, int* info);
void PREFIX DTRTRS_F77(Teuchos_fcd, Teuchos_fcd, Teuchos_fcd, const int* n, const int* nrhs, double* a, const int* lda, double* b, const int* ldb, int* info);

// Single precision LAPACK linear solvers

#ifdef HAVE_TEUCHOS_BLASFLOAT

void PREFIX SGELS_F77(Teuchos_fcd ch, const int* m, const int* n, const int* nrhs, float* a, const int* lda, float* b, const int* ldb, float* work, const int* lwork, int* info);
void PREFIX SGEQRF_F77(const int* m, const int* n, float* a, const int* lda, float* tau, float* work, const int* lwork, int* info);
void PREFIX SGETRF_F77(const int* m, const int* n, float* a, const int* lda, int* ipiv, int* info);
void PREFIX SGETRS_F77(Teuchos_fcd, const int* n, const int* nrhs, const float* a, const int* lda,const int* ipiv, float* x , const int* ldx, int* info);
void PREFIX SGTTRF_F77(const int* n, float* dl, float* d, float* du, float* du2, int* ipiv, int* info); 
void PREFIX SGTTRS_F77(Teuchos_fcd, const int* n, const int* nrhs, const float* dl, const float* d, const float* du, const float* du2, const int* ipiv, float* x , const int* ldx, int* info);
void PREFIX SPTTRF_F77(const int* n, float* d, float* e, int* info); 
void PREFIX SPTTRS_F77(const int* n, const int* nrhs, const float* d, const float* e, float* x , const int* ldx, int* info);
void PREFIX SGETRI_F77(const int* n, float* a, const int* lda, const int* ipiv, float* work , const int* lwork, int* info);
void PREFIX SGECON_F77(Teuchos_fcd norm, const int* n, const float* a, const int* lda, const float* anorm, float* rcond, float* work, int* iwork, int* info); 
void PREFIX SGESV_F77(const int* n, const int* nrhs, float* a, const int* lda, int* ipiv, float* x , const int* ldx, int* info);
void PREFIX SGEEQU_F77(const int* m, const int* n, const float* a, const int* lda, float* r, float* c, float* rowcnd, float* colcnd, float* amax, int* info); 
void PREFIX SGERFS_F77(Teuchos_fcd, const int* n, const int* nrhs, const float* a, const int* lda, const float* af, const int* ldaf, const int* ipiv, const float* b, const int* ldb, float* x, const int* ldx, float* ferr, float* berr, float* work, int* iwork, int* info);
void PREFIX SGESVX_F77(Teuchos_fcd, Teuchos_fcd, const int* n, const int* nrhs, float* a, const int* lda, float* af, const int* ldaf, int* ipiv, Teuchos_fcd, float* r,
float* c, float* b, const int* ldb, float* x, const int* ldx, float* rcond, float* ferr, float* berr, float* work, int* iwork, int* info);
void PREFIX SSYTRD_F77(Teuchos_fcd, const int* n, float* a, const int* lda, float* D, float* E, float* tau, float* work, const int* lwork, int* info);
void PREFIX SPOTRF_F77(Teuchos_fcd, const int* n, float* a, const int* lda, int* info); 
void PREFIX SPOTRS_F77(Teuchos_fcd, const int* n, const int* nrhs, const float* a, const int* lda, float*x , const int* ldx, int* info);
void PREFIX SPOTRI_F77(Teuchos_fcd, const int* n, float* a, const int* lda, int* info); 
void PREFIX SPOCON_F77(Teuchos_fcd, const int* n, const float* a, const int* lda, const float* anorm, float* rcond, float* work, int* iwork, int* info); 
void PREFIX SPOSV_F77(Teuchos_fcd, const int* n, const int* nrhs, float* a, const int* lda, float*x , const int* ldx, int* info);
void PREFIX SPOEQU_F77(const int* n, const float* a, const int* lda, float* s, float* scond, float* amax, int* info); 
void PREFIX SPORFS_F77(Teuchos_fcd, const int* n, const int* nrhs, float* a, const int* lda, const float* af, const int* ldaf, const float* b, const int* ldb, float* x, const int* ldx, float* ferr, float* berr, float* work, int* iwork, int* info);
void PREFIX SPOSVX_F77(Teuchos_fcd, Teuchos_fcd, const int* n, const int* nrhs, float* a, const int* lda, float* af, const int* ldaf, Teuchos_fcd, float* s, float* b, const int* ldb, float* x, const int* ldx, float* rcond, float* ferr, float* berr, float* work, int* iwork, int* info);
void PREFIX STRTRS_F77(Teuchos_fcd, Teuchos_fcd, Teuchos_fcd, const int* n, const int* nrhs, float* a, const int* lda, float* b, const int* ldb, int* info);

#endif // HAVE_TEUCHOS_BLASFLOAT

// Double precision LAPACK eigen solvers
void PREFIX DSPEV_F77(Teuchos_fcd, Teuchos_fcd, const int* n, double* ap, double* w, double* z, const int* ldz, double* work, int* info);
void PREFIX DSYEV_F77(Teuchos_fcd, Teuchos_fcd, const int* n, double* a, const int* lda, double* w, double* work, const int* lwork, int* info);
void PREFIX DSYGV_F77(const int* itype, Teuchos_fcd, Teuchos_fcd, const int* n, double* a, const int* lda, double* B, const int* ldb, double* w, double* work, const int* lwork, int* info);
void PREFIX DSTEQR_F77(Teuchos_fcd, const int* n, double* D, double* E, double* Z, const int* ldz, double* work, int* info);
void PREFIX DGEEV_F77(Teuchos_fcd, Teuchos_fcd, const int* n, double* a, const int* lda, double* wr, double* wi, double* vl, const int* ldvl, double* vr, const int* ldvr, double* work, const int* lwork, int* info);
void PREFIX DGGEVX_F77(Teuchos_fcd, Teuchos_fcd, Teuchos_fcd, Teuchos_fcd, const int* n, double* a, const int* lda, double* b, const int* ldb, double* alphar, double* alphai, double* beta, double* vl, const int* ldvl, double* vr, const int* ldvr, int* ilo, int* ihi, double* lscale, double* rscale, double* abnrm, double* bbnrm, double* rconde, double* rcondv, double* work, const int* lwork, int* iwork, int* bwork, int* info);
void PREFIX DGEHRD_F77(const int* n, const int* ilo, const int* ihi, double* A, const int* lda, double* tau, double* work, const int* lwork, int* info);
void PREFIX DHSEQR_F77(Teuchos_fcd job, Teuchos_fcd, const int* n, const int* ilo, const int* ihi, double* h, const int* ldh, double* wr, double* wi, double* z, const int* ldz, double* work, const int* lwork, int* info);
void PREFIX DGEES_F77(Teuchos_fcd, Teuchos_fcd, int (*ptr2func)(double*, double*), const int* n, double* a, const int* lda, int*sdim, double* wr, double* wi, double* vs, const int* ldvs, double* work, const int* lwork, int* bwork, int* info);
void PREFIX DORGHR_F77(const int* n, const int* ilo, const int* ihi, double* a, const int* lda, double* tau, double* work, int* lwork, int* info);
void PREFIX DORMHR_F77(Teuchos_fcd, Teuchos_fcd, const int* m, const int* n, const int* ilo, const int* ihi, const double* a, const int* lda, const double* tau, double* c, const int* ldc, double* work, int* lwork, int* info);
void PREFIX DORGQR_F77(const int* m, const int* n, const int* k, double* a, const int* lda, const double* tau, double* work, const int* lwork, int* info);
void PREFIX DORMQR_F77(Teuchos_fcd, Teuchos_fcd, const int* m, const int* n, const int* k, double* a, const int* lda, const double* tau, double* C, const int* ldc, double* work, const int* lwork, int* info);
void PREFIX DTREVC_F77(Teuchos_fcd, Teuchos_fcd, int (*ptr2func)(double*,double*), const int* n, const double* t, const int* ldt, double* vl, const int* ldvl, double* vr, const int* ldvr, const int* mm, int* m, double* work, int* info); 
void PREFIX DTREXC_F77(Teuchos_fcd, const int* n, double* t, const int* ldt, double* q, const int* ldq, int* ifst, int* ilst, double* work, int* info);

// Single precision LAPACK eigen solvers

#ifdef HAVE_TEUCHOS_BLASFLOAT

void PREFIX SSPEV_F77(Teuchos_fcd, Teuchos_fcd, const int* n, float* ap, float* w, float* z, const int* ldz, float* work, int* info);
void PREFIX SSYEV_F77(Teuchos_fcd, Teuchos_fcd, const int* n, float* a, const int* lda, float* w, float* work, const int* lwork, int* info);
void PREFIX SSYGV_F77(const int* itype, Teuchos_fcd, Teuchos_fcd, const int* n, float* a, const int* lda, float* B, const int* ldb, float* w, float* work, const int* lwork, int* info);
void PREFIX SSTEQR_F77(Teuchos_fcd, const int* n, float* D, float* E, float* Z, const int* ldz, float* work, int* info);
void PREFIX SGEEV_F77(Teuchos_fcd, Teuchos_fcd, const int* n, float* a, const int* lda, float* wr, float* wi, float* vl, const int* ldvl, float* vr, const int* ldvr, float* work, const int* lwork, int* info);
void PREFIX SGGEVX_F77(Teuchos_fcd, Teuchos_fcd, Teuchos_fcd, Teuchos_fcd, const int* n, float* a, const int* lda, float* b, const int* ldb, float* alphar, float* alphai, float* beta, float* vl, const int* ldvl, float* vr, const int* ldvr, int* ilo, int* ihi, float* lscale, float* rscale, float* abnrm, float* bbnrm, float* rconde, float* rcondv, float* work, const int* lwork, int* iwork, int* bwork, int* info);
void PREFIX SGEHRD_F77(const int* n, const int* ilo, const int* ihi, float* A, const int* lda, float* tau, float* work, const int* lwork, int* info);
void PREFIX SHSEQR_F77(Teuchos_fcd job, Teuchos_fcd, const int* n, const int* ilo, const int* ihi, float* h, const int* ldh, float* wr, float* wi, float* z, const int* ldz, float* work, const int* lwork, int* info);
void PREFIX SGEES_F77(Teuchos_fcd, Teuchos_fcd, int (*ptr2func)(float*, float*), const int* n, float* a, const int* lda, int* sdim, float* wr, float* wi, float* vs, const int* ldvs, float* work, const int* lwork, int* bwork, int* info);
void PREFIX SORGHR_F77(const int* n, const int* ilo, const int* ihi, float* a, const int* lda, float* tau, float* work, int* lwork, int* info);
void PREFIX SORMHR_F77(Teuchos_fcd, Teuchos_fcd, const int* m, const int* n, const int* ilo, const int* ihi, const float* a, const int* lda, const float* tau, float* c, const int* ldc, float* work, int* lwork, int* info);
void PREFIX SORGQR_F77(const int* m, const int* n, const int* k, float* a, const int* lda, const float* tau, float* work, const int* lwork, int* info);
void PREFIX SORMQR_F77(Teuchos_fcd, Teuchos_fcd, const int* m, const int* n, const int* k, float* a, const int* lda, const float* tau, float* C, const int* ldc, float* work, const int* lwork, int* info);
void PREFIX STREVC_F77(Teuchos_fcd, Teuchos_fcd, int (*ptr2func)(float*,float*), const int* n, const float* t, const int* ldt, float* vl, const int* ldvl, float* vr, const int* ldvr, const int* mm, int* m, float* work, int* info); 
void PREFIX STREXC_F77(Teuchos_fcd, const int* n, float* t, const int* ldt, float* q, const int* ldq, int* ifst, int* ilst, float* work, int* info);

#endif // HAVE_TEUCHOS_BLASFLOAT

void PREFIX SLARTG_F77(const float* f, const float* g, float* c, float* s, float* r);
void PREFIX DLARTG_F77(const double* f, const double* g, double* c, double* s, double* r);

void PREFIX SLARFG_F77(const int* n, float* alpha, float* x, const int* incx, float* tau);
void PREFIX DLARFG_F77(const int* n, double* alpha, double* x, const int* incx, double* tau);

float PREFIX SLARND_F77(const int* idist, int* seed);
double PREFIX DLARND_F77(const int* idist, int* seed);

void PREFIX SLARNV_F77(const int* idist, int* seed, const int* n, float* v);
void PREFIX DLARNV_F77(const int* idist, int* seed, const int* n, double* v);

float PREFIX SLAMCH_F77(Teuchos_fcd);
double PREFIX DLAMCH_F77(Teuchos_fcd);

#if defined(INTEL_CXML)
int PREFIX ILAENV_F77( const int* ispec, const char* name, unsigned int name_length, const char* opts, unsigned int opts_length, const int* N1, const int* N2, const int* N3, const int* N4 );
#else
int PREFIX ILAENV_F77( const int* ispec, const char* name, const char* opts, const int* N1, const int* N2, const int* N3, const int* N4, unsigned int name_length, unsigned int opts_length );
#endif

float PREFIX SLAPY2_F77(const float* x, const float* y);
double PREFIX DLAPY2_F77(const double* x, const double* y);

#ifdef HAVE_TEUCHOS_COMPLEX

// Double precision std::complex LAPACK linear solvers
void PREFIX ZGELS_F77(Teuchos_fcd ch, const int* m, const int* n, const int* nrhs, std::complex<double>* a, const int* lda, std::complex<double>* b, const int* ldb, std::complex<double>* work, const int* lwork, int* info);
void PREFIX ZGEQRF_F77(const int* m, const int* n, std::complex<double>* a, const int* lda, std::complex<double>* tau, std::complex<double>* work, const int* lwork, int* info);
void PREFIX ZUNGQR_F77(const int* m, const int* n, const int* k, std::complex<double>* a, const int* lda, const std::complex<double>* tau, std::complex<double>* work, const int* lwork, int* info);
void PREFIX ZGETRF_F77(const int* m, const int* n, std::complex<double>* a, const int* lda, int* ipiv, int* info); 
void PREFIX ZGETRS_F77(Teuchos_fcd, const int* n, const int* nrhs, const std::complex<double>* a, const int* lda,const int* ipiv, std::complex<double>* x , const int* ldx, int* info);
void PREFIX ZGTTRF_F77(const int* n, std::complex<double>* dl, std::complex<double>* d, std::complex<double>* du, std::complex<double>* du2, int* ipiv, int* info); 
void PREFIX ZGTTRS_F77(Teuchos_fcd, const int* n, const int* nrhs, const std::complex<double>* dl, const std::complex<double>* d, const std::complex<double>* du, const std::complex<double>* du2, const int* ipiv, std::complex<double>* x , const int* ldx, int* info);
void PREFIX ZPTTRF_F77(const int* n, std::complex<double>* d, std::complex<double>* e, int* info); 
void PREFIX ZPTTRS_F77(const int* n, const int* nrhs, const std::complex<double>* d, const std::complex<double>* e, std::complex<double>* x , const int* ldx, int* info);
void PREFIX ZGETRI_F77(const int* n, std::complex<double>* a, const int* lda, const int* ipiv, std::complex<double>* work , const int* lwork, int* info);
void PREFIX ZGECON_F77(Teuchos_fcd norm, const int* n, const std::complex<double>* a, const int* lda, const double* anorm, double* rcond, std::complex<double>* work, double* rwork, int* info); 
void PREFIX ZGESV_F77(const int* n, const int* nrhs, std::complex<double>* a, const int* lda, int* ipiv, std::complex<double>* x , const int* ldx, int* info);
void PREFIX ZGEEQU_F77(const int* m, const int* n, const std::complex<double>* a, const int* lda, double* r, double* c, double* rowcnd, double* colcnd, double* amax, int* info); 
void PREFIX ZGERFS_F77(Teuchos_fcd, const int* n, const int* nrhs, const std::complex<double>* a, const int* lda, const std::complex<double>* af, const int* ldaf, const int* ipiv, const std::complex<double>* b, const int* ldb, std::complex<double>* x, const int* ldx, double* ferr, double* berr, std::complex<double>* work, double* iwork, int* info);
//void PREFIX ZGESVX_F77(Teuchos_fcd, Teuchos_fcd, const int* n, const int* nrhs, std::complex<double>* a, const int* lda, std::complex<double>* af, const int* ldaf, int* ipiv, Teuchos_fcd, double* r, double* c, std::complex<double>* b, const int* ldb, std::complex<double>* x, const int* ldx, double* rcond, double* ferr, double* berr, std::complex<double>* work, double* iwork, int* info);
void PREFIX ZPOTRF_F77(Teuchos_fcd, const int* n, std::complex<double>* a, const int* lda, int* info); 
void PREFIX ZPOTRS_F77(Teuchos_fcd, const int* n, const int* nrhs, const std::complex<double>* a, const int* lda, std::complex<double>*x , const int* ldx, int* info);
void PREFIX ZPOTRI_F77(Teuchos_fcd, const int* n, std::complex<double>* a, const int* lda, int* info); 
void PREFIX ZPOCON_F77(Teuchos_fcd, const int* n, const std::complex<double>* a, const int* lda, const double* anorm, double* rcond, std::complex<double>* work, double* rwork, int* info); 
void PREFIX ZPOSV_F77(Teuchos_fcd, const int* n, const int* nrhs, std::complex<double>* a, const int* lda, std::complex<double>*x , const int* ldx, int* info);
void PREFIX ZPOEQU_F77(const int* n, const std::complex<double>* a, const int* lda, double* s, double* scond, double* amax, int* info); 
void PREFIX ZPORFS_F77(Teuchos_fcd, const int* n, const int* nrhs, std::complex<double>* a, const int* lda, const std::complex<double>* af, const int* ldaf, const std::complex<double>* b, const int* ldb, std::complex<double>* x, const int* ldx, double* ferr, double* berr, std::complex<double>* work, double* rwork, int* info);
void PREFIX ZPOSVX_F77(Teuchos_fcd, Teuchos_fcd, const int* n, const int* nrhs, std::complex<double>* a, const int* lda, std::complex<double>* af, const int* ldaf, Teuchos_fcd, double* s, std::complex<double>* b, const int* ldb, std::complex<double>* x, const int* ldx, double* rcond, double* ferr, double* berr, std::complex<double>* work, double* rwork, int* info);
void PREFIX ZTRTRS_F77(Teuchos_fcd, Teuchos_fcd, Teuchos_fcd, const int* n, const int* nrhs, std::complex<double>* a, const int* lda, std::complex<double>* b, const int* ldb, int* info);

// Single precision std::complex LAPACK linear solvers

#ifdef HAVE_TEUCHOS_BLASFLOAT

void PREFIX CGELS_F77(Teuchos_fcd ch, const int* m, const int* n, const int* nrhs, std::complex<float>* a, const int* lda, std::complex<float>* b, const int* ldb, std::complex<float>* work, const int* lwork, int* info);
void PREFIX CGEQRF_F77(const int* m, const int* n, std::complex<float>* a, const int* lda, std::complex<float>* tau, std::complex<float>* work, const int* lwork, int* info);
void PREFIX CUNGQR_F77(const int* m, const int* n, const int* k, std::complex<float>* a, const int* lda, const std::complex<float>* tau, std::complex<float>* work, const int* lwork, int* info);
void PREFIX CGETRF_F77(const int* m, const int* n, std::complex<float>* a, const int* lda, int* ipiv, int* info);
void PREFIX CGETRS_F77(Teuchos_fcd, const int* n, const int* nrhs, const std::complex<float>* a, const int* lda,const int* ipiv, std::complex<float>* x , const int* ldx, int* info);
void PREFIX CGTTRF_F77(const int* n, std::complex<float>* dl, std::complex<float>* d, std::complex<float>* du, std::complex<float>* du2, int* ipiv, int* info); 
void PREFIX CGTTRS_F77(Teuchos_fcd, const int* n, const int* nrhs, const std::complex<float>* dl, const std::complex<float>* d, const std::complex<float>* du, const std::complex<float>* du2, const int* ipiv, std::complex<float>* x , const int* ldx, int* info);
void PREFIX CPTTRF_F77(const int* n, std::complex<float>* d, std::complex<float>* e, int* info); 
void PREFIX CPTTRS_F77(const int* n, const int* nrhs, const std::complex<float>* d, const std::complex<float>* e, std::complex<float>* x , const int* ldx, int* info);
void PREFIX CGETRI_F77(const int* n, std::complex<float>* a, const int* lda, const int* ipiv, std::complex<float>* work , const int* lwork, int* info);
void PREFIX CGECON_F77(Teuchos_fcd norm, const int* n, const std::complex<float>* a, const int* lda, const float* anorm, float* rcond, std::complex<float>* work, float* rwork, int* info); 
void PREFIX CGESV_F77(const int* n, const int* nrhs, std::complex<float>* a, const int* lda, int* ipiv, std::complex<float>* x, const int* ldx, int* info);
void PREFIX CGEEQU_F77(const int* m, const int* n, const std::complex<float>* a, const int* lda, float* r, float* c, float* rowcnd, float* colcnd, float* amax, int* info); 
void PREFIX CGERFS_F77(Teuchos_fcd, const int* n, const int* nrhs, const std::complex<float>* a, const int* lda, const std::complex<float>* af, const int* ldaf, const int* ipiv, const std::complex<float>* b, const int* ldb, std::complex<float>* x, const int* ldx, float* ferr, float* berr, std::complex<float>* work, float* rwork, int* info);
void PREFIX CGESVX_F77(Teuchos_fcd, Teuchos_fcd, const int* n, const int* nrhs, std::complex<float>* a, const int* lda, std::complex<float>* af, const int* ldaf, int* ipiv, Teuchos_fcd, float* r, float* c, std::complex<float>* b, const int* ldb, std::complex<float>* x, const int* ldx, float* rcond, float* ferr, float* berr, std::complex<float>* work, float* rwork, int* info);
void PREFIX CPOTRF_F77(Teuchos_fcd, const int* n, std::complex<float>* a, const int* lda, int* info); 
void PREFIX CPOTRS_F77(Teuchos_fcd, const int* n, const int* nrhs, const std::complex<float>* a, const int* lda, std::complex<float>*x , const int* ldx, int* info);
void PREFIX CPOTRI_F77(Teuchos_fcd, const int* n, std::complex<float>* a, const int* lda, int* info); 
void PREFIX CPOCON_F77(Teuchos_fcd, const int* n, const std::complex<float>* a, const int* lda, const float* anorm, float* rcond, std::complex<float>* work, float* rwork, int* info); 
void PREFIX CPOSV_F77(Teuchos_fcd, const int* n, const int* nrhs, std::complex<float>* a, const int* lda, std::complex<float>*x , const int* ldx, int* info);
void PREFIX CPOEQU_F77(const int* n, const std::complex<float>* a, const int* lda, float* s, float* scond, float* amax, int* info); 
void PREFIX CPORFS_F77(Teuchos_fcd, const int* n, const int* nrhs, std::complex<float>* a, const int* lda, const std::complex<float>* af, const int* ldaf, const std::complex<float>* b, const int* ldb, std::complex<float>* x, const int* ldx, float* ferr, float* berr, std::complex<float>* work, float* rwork, int* info);
void PREFIX CPOSVX_F77(Teuchos_fcd, Teuchos_fcd, const int* n, const int* nrhs, std::complex<float>* a, const int* lda, std::complex<float>* af, const int* ldaf, Teuchos_fcd, float* s, std::complex<float>* b, const int* ldb, std::complex<float>* x, const int* ldx, float* rcond, float* ferr, float* berr, std::complex<float>* work, float* rwork, int* info);
void PREFIX CTRTRS_F77(Teuchos_fcd, Teuchos_fcd, Teuchos_fcd, const int* n, const int* nrhs, std::complex<float>* a, const int* lda, std::complex<float>* b, const int* ldb, int* info);

#endif // HAVE_TEUCHOS_BLASFLOAT

// Double precision std::complex LAPACK eigen solvers
void PREFIX ZSTEQR_F77(Teuchos_fcd, const int* n, double* D, double* E, std::complex<double>* Z, const int* ldz, std::complex<double>* work, int* info);
void PREFIX ZHEEV_F77(Teuchos_fcd, Teuchos_fcd, const int* n, std::complex<double>* a, const int* lda, double* w, std::complex<double>* work, const int* lwork, double* rwork, int* info);
void PREFIX ZHEGV_F77(const int* itype, Teuchos_fcd, Teuchos_fcd, const int* n, std::complex<double>* a, const int* lda, std::complex<double>* B, const int* ldb, double* w, std::complex<double>* work, const int* lwork, double *rwork, int* info);
void PREFIX ZGEEV_F77(Teuchos_fcd, Teuchos_fcd, const int* n, std::complex<double>* a, const int* lda, std::complex<double>* w, std::complex<double>* vl, const int* ldvl, std::complex<double>* vr, const int* ldvr, std::complex<double>* work, const int* lwork, double* rwork, int* info);
//void PREFIX ZGGEVX_F77(Teuchos_fcd, Teuchos_fcd, Teuchos_fcd, Teuchos_fcd, const int* n, std::complex<double>* a, const int* lda, std::complex<double>* b, const int* ldb, std::complex<double>* alpha, std::complex<double>* beta, std::complex<double>* vl, const int* ldvl, std::complex<double>* vr, const int* ldvr, int* ilo, int* ihi, double* lscale, double* rscale, double* abnrm, double* bbnrm, double* rconde, double* rcondv, std::complex<double>* work, const int* lwork, int* iwork, int* bwork, int* info);
void PREFIX ZGEHRD_F77(const int* n, const int* ilo, const int* ihi, std::complex<double>* A, const int* lda, std::complex<double>* tau, std::complex<double>* work, const int* lwork, int* info);
void PREFIX ZHSEQR_F77(Teuchos_fcd job, Teuchos_fcd, const int* n, const int* ilo, const int* ihi, std::complex<double>* h, const int* ldh, std::complex<double>* w, std::complex<double>* z, const int* ldz, std::complex<double>* work, const int* lwork, int* info);
void PREFIX ZGEES_F77(Teuchos_fcd, Teuchos_fcd, int (*ptr2func)(std::complex<double>*), const int* n, std::complex<double>* a, const int* lda, int* sdim, std::complex<double>* w, std::complex<double>* vs, const int* ldvs, std::complex<double>* work, const int* lwork, double* rwork, int* bwork, int* info);
void PREFIX ZTREVC_F77(Teuchos_fcd, Teuchos_fcd, int (*ptr2func)(std::complex<double>*), const int* n, const std::complex<double>* t, const int* ldt, std::complex<double>* vl, const int* ldvl, std::complex<double>* vr, const int* ldvr, const int* mm, int* m, std::complex<double>* work, double* rwork, int* info); 
void PREFIX ZTREXC_F77(Teuchos_fcd, const int* n, std::complex<double>* t, const int* ldt, std::complex<double>* q, const int* ldq, int* ifst, int* ilst, int* info);

// Single precision std::complex LAPACK eigen solvers

#ifdef HAVE_TEUCHOS_BLASFLOAT

void PREFIX CSTEQR_F77(Teuchos_fcd, const int* n, std::complex<float>* D, std::complex<float>* E, std::complex<float>* Z, const int* ldz, std::complex<float>* work, int* info);
void PREFIX CHEEV_F77(Teuchos_fcd, Teuchos_fcd, const int* n, std::complex<float>* a, const int* lda, float* w, std::complex<float>* work, const int* lwork, float* rwork, int* info);
void PREFIX CHEGV_F77(const int* itype, Teuchos_fcd, Teuchos_fcd, const int* n, std::complex<float>* a, const int* lda, std::complex<float>* B, const int* ldb, float* w, std::complex<float>* work, const int* lwork, float *rwork, int* info);
void PREFIX CGEEV_F77(Teuchos_fcd, Teuchos_fcd, const int* n, std::complex<float>* a, const int* lda, std::complex<float>* wr, std::complex<float>* vl, const int* ldvl, std::complex<float>* vr, const int* ldvr, std::complex<float>* work, const int* lwork, float* rwork, int* info);
void PREFIX CGGEVX_F77(Teuchos_fcd, Teuchos_fcd, Teuchos_fcd, Teuchos_fcd, const int* n, std::complex<float>* a, const int* lda, std::complex<float>* b, const int* ldb, std::complex<float>* alpha, std::complex<float>* beta, std::complex<float>* vl, const int* ldvl, std::complex<float>* vr, const int* ldvr, int* ilo, int* ihi, float* lscale, float* rscale, float* abnrm, float* bbnrm, float* rconde, float* rcondv, std::complex<float>* work, const int* lwork, int* iwork, int* bwork, int* info);
void PREFIX CGEHRD_F77(const int* n, const int* ilo, const int* ihi, std::complex<float>* A, const int* lda, std::complex<float>* tau, std::complex<float>* work, const int* lwork, int* info);
void PREFIX CHSEQR_F77(Teuchos_fcd job, Teuchos_fcd, const int* n, const int* ilo, const int* ihi, std::complex<float>* h, const int* ldh, std::complex<float>* w, std::complex<float>* z, const int* ldz, std::complex<float>* work, const int* lwork, int* info);
void PREFIX CGEES_F77(Teuchos_fcd, Teuchos_fcd, int (*ptr2func)(std::complex<float>*), const int* n, std::complex<float>* a, const int* lda, int* sdim, std::complex<float>* w, std::complex<float>* vs, const int* ldvs, std::complex<float>* work, const int* lwork, float* rwork, int* bwork, int* info);
void PREFIX CTREVC_F77(Teuchos_fcd, Teuchos_fcd, int (*ptr2func)(std::complex<float>*), const int* n, const std::complex<float>* t, const int* ldt, std::complex<float>* vl, const int* ldvl, std::complex<float>* vr, const int* ldvr, const int* mm, int* m, std::complex<float>* work, float* rwork, int* info); 
void PREFIX CTREXC_F77(Teuchos_fcd, const int* n, std::complex<float>* t, const int* ldt, std::complex<float>* q, const int* ldq, int* ifst, int* ilst, int* info);

#endif // HAVE_TEUCHOS_BLASFLOAT

void PREFIX CLARTG_F77(const std::complex<float>* f, const std::complex<float>* g, float* c, std::complex<float>* s, std::complex<float>* r);
void PREFIX ZLARTG_F77(const std::complex<double>* f, const std::complex<double>* g, double* c, std::complex<double>* s, std::complex<double>* r);

void PREFIX CLARFG_F77(const int* n, std::complex<float>* alpha, std::complex<float>* x, const int* incx, std::complex<float>* tau);
void PREFIX ZLARFG_F77(const int* n, std::complex<double>* alpha, std::complex<double>* x, const int* incx, std::complex<double>* tau);

std::complex<float> PREFIX CLARND_F77(const int* idist, int* seed);
std::complex<double> PREFIX ZLARND_F77(const int* idist, int* seed);

void PREFIX CLARNV_F77(const int* idist, int* seed, const int* n, std::complex<float>* v);
void PREFIX ZLARNV_F77(const int* idist, int* seed, const int* n, std::complex<double>* v);

#endif /* HAVE_TEUCHOS_COMPLEX */

#ifdef __cplusplus
}
#endif

#endif // end of TEUCHOS_LAPACK_WRAPPERS_HPP_
