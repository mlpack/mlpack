/*@HEADER
// ***********************************************************************
//
//       Ifpack: Object-Oriented Algebraic Preconditioner Package
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

#ifndef IFPACK_IC_UTILS_H
#define IFPACK_IC_UTILS_H

typedef struct {
    double *val;  /* also known as A  */
    int    *col;  /* also known as JA; first column is column 0 */
    int    *ptr;  /* also known as IA; with ptr[0] = 0 */
} Ifpack_AIJMatrix;

extern "C" {
void quicksort (int *const pbase, double *const daux, int total_elems);
}

void Ifpack_AIJMatrix_dealloc(Ifpack_AIJMatrix *a);

void crout_ict(
    int n,
#ifdef IFPACK
    void * A,
    int maxentries,
    int (*getcol)( void * A, int col, int ** nentries, double * val, int * ind),
    int (*getdiag)( void *A, double * diag),
#else
    const Ifpack_AIJMatrix *AL,
    const double *Adiag,
#endif
    double droptol,
    int lfil,
    Ifpack_AIJMatrix *L,
    double **pdiag);


#endif
