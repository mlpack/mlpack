//@HEADER
// ***********************************************************************
// 
//                 TriUtils: Trilinos Utilities Package
//                 Copyright (2001) Sandia Corporation
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

#ifndef IOHB_H
#define IOHB_H

#ifdef HAVE_CONFIG_H

/*
 * The macros PACKAGE, PACKAGE_NAME, etc, get defined for each package and need
 * to
 * be undef'd here to avoid warnings when this file is included from another
 * package.
 * KL 11/25/02
 */
#ifdef PACKAGE
#undef PACKAGE
#endif

#ifdef PACKAGE_NAME
#undef PACKAGE_NAME
#endif

#ifdef PACKAGE_BUGREPORT
#undef PACKAGE_BUGREPORT
#endif

#ifdef PACKAGE_STRING
#undef PACKAGE_STRING
#endif

#ifdef PACKAGE_TARNAME
#undef PACKAGE_TARNAME
#endif

#ifdef PACKAGE_VERSION
#undef PACKAGE_VERSION
#endif

#ifdef VERSION
#undef VERSION
#endif

#include "Triutils_config.h"
#endif

#include<cstdio>
#include<cstdlib>

int readHB_info(const char* filename, int* M, int* N, int* nz, char** Type, 
                                                      int* Nrhs);

int readHB_header(std::FILE* in_file, char* Title, char* Key, char* Type, 
                    int* Nrow, int* Ncol, int* Nnzero, int* Nrhs,
                    char* Ptrfmt, char* Indfmt, char* Valfmt, char* Rhsfmt, 
                    int* Ptrcrd, int* Indcrd, int* Valcrd, int* Rhscrd, 
                    char *Rhstype);

int readHB_mat_double(const char* filename, int colptr[], int rowind[], 
                                                                 double val[]);

int readHB_newmat_double(const char* filename, int* M, int* N, int* nonzeros, 
                         int** colptr, int** rowind, double** val);

int readHB_aux_double(const char* filename, const char AuxType, double b[]);

int readHB_newaux_double(const char* filename, const char AuxType, double** b);

int writeHB_mat_double(const char* filename, int M, int N, 
                        int nz, const int colptr[], const int rowind[], 
                        const double val[], int Nrhs, const double rhs[], 
                        const double guess[], const double exact[],
                        const char* Title, const char* Key, const char* Type, 
                        char* Ptrfmt, char* Indfmt, char* Valfmt, char* Rhsfmt,
                        const char* Rhstype);

int readHB_mat_char(const char* filename, int colptr[], int rowind[], 
                                           char val[], char* Valfmt);

int readHB_newmat_char(const char* filename, int* M, int* N, int* nonzeros, int** colptr, 
                          int** rowind, char** val, char** Valfmt);

int readHB_aux_char(const char* filename, const char AuxType, char b[]);

int readHB_newaux_char(const char* filename, const char AuxType, char** b, char** Rhsfmt);

int writeHB_mat_char(const char* filename, int M, int N, 
                        int nz, const int colptr[], const int rowind[], 
                        const char val[], int Nrhs, const char rhs[], 
                        const char guess[], const char exact[], 
                        const char* Title, const char* Key, const char* Type, 
                        char* Ptrfmt, char* Indfmt, char* Valfmt, char* Rhsfmt,
                        const char* Rhstype);

int ParseIfmt(char* fmt, int* perline, int* width);

int ParseRfmt(char* fmt, int* perline, int* width, int* prec, int* flag);

void IOHBTerminate(char* message);

#endif
