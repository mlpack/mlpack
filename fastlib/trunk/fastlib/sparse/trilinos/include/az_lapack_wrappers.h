/*@HEADER
// ***********************************************************************
// 
//        AztecOO: An Object-Oriented Aztec Linear Solver Package 
//                 Copyright (2002) Sandia Corporation
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
//@HEADER
*/

#ifndef _AZ_LAPACK_WRAPPERS_H_
#define _AZ_LAPACK_WRAPPERS_H_

#include "az_f77func.h"

#if defined(CRAY_T3X)

#define DGETRF_F77  F77_FUNC(sgetrf,SGETRF)
#define DGETRS_F77  F77_FUNC(sgetrs,SGETRS)
#define DPOTRF_F77  F77_FUNC(spotrf,SPOTRF)
#define DGETRI_F77  F77_FUNC(sgetri,SGETRI)
#define DSTEBZ_F77  F77_FUNC(sstebz,SSTEBZ)
#define DGEEV_F77   F77_FUNC(sgeev,SGEEV)

#else

#define DGETRF_F77  F77_FUNC(dgetrf,DGETRF)
#define DGETRS_F77  F77_FUNC(dgetrs,DGETRS)
#define DPOTRF_F77  F77_FUNC(dpotrf,DPOTRF)
#define DGETRI_F77  F77_FUNC(dgetri,DGETRI)
#define DSTEBZ_F77  F77_FUNC(dstebz,DSTEBZ)
#define DGEEV_F77   F77_FUNC(dgeev,DGEEV)

#endif

#ifdef __cplusplus
extern "C" {
#include <stdio.h>
#endif


  /* Double precision LAPACK linear solvers */
void PREFIX DGETRF_F77(int* m, int* n, double* a, int* lda, int* ipiv, int* info); 
void PREFIX DGETRS_F77(az_fcd, int* n, int* nrhs, double* a,
                       int* lda, int*ipiv, double*x , int* ldx, int* info);
void PREFIX DGETRI_F77(int* n, double* a, int* lda, int*ipiv, double * work , int* lwork, int* info);
void PREFIX DPOTRF_F77(az_fcd, int* n, double* a, int* lda, int* info);
void PREFIX DSTEBZ_F77(az_fcd, az_fcd, int *, double *, double *, int *, int *,
		       double *, double *, double *, int *, int *, double *, int *,
		       int *, double *, int *, int *);
void PREFIX DGEEV_F77(az_fcd, az_fcd, int *, double *, int *, double *,
		      double *, double *, int *, double *, int *,
		      double *, int *, int *);

  /* Single precision LAPACK linear solvers*/
void PREFIX SGETRF_F77(int* m, int* n, float* a, int* lda, int* ipiv, int* info); 
void PREFIX SGETRS_F77(az_fcd, int* m, int* n, float* a,
                       int* lda, int*ipiv, float*x , int* ldx, int* info);
void PREFIX SGETRI_F77(int* n, float* a, int* lda, int*ipiv, float * work , int* lwork, int* info);
void PREFIX SPOTRF_F77(az_fcd, int* n, float* a, int* lda, int* info); 
void PREFIX SSTEBZ_F77(az_fcd, az_fcd, int *, float *, float *, int *, int *,
		       float *, float *, float *, int *, int *, double*,  int *,
		       int *, float *, int *, int *);
void PREFIX SGEEV_F77(az_fcd, az_fcd, int *, float *, int *, float *,
		      float *, float *, int *, float *, int *,
		      float *, int *, int *);


#ifdef __cplusplus
}
#endif

#endif /* _AZ_LAPACK_WRAPPERS_H_ */

void PREFIX DGETRS_F77(az_fcd, int* n, int* nrhs, double* a,
                       int* lda, int*ipiv, double*x , int* ldx, int* info);
void PREFIX SGETRS_F77(az_fcd, int* m, int* n, float* a,
                       int* lda, int*ipiv, float*x , int* ldx, int* info);
