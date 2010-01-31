/* @HEADER
* ************************************************************************
*
*            Trilinos: An Object-Oriented Solver Framework
*                 Copyright (2001) Sandia Corporation
*

* Copyright (2001) Sandia Corportation. Under the terms of Contract
* DE-AC04-94AL85000, there is a non-exclusive license for use of this
* work by or on behalf of the U.S. Government.  Export of this program
* may require a license from the United States Government.
*
* NOTICE:  The United States Government is granted for itself and others
* acting on its behalf a paid-up, nonexclusive, irrevocable worldwide
* license in ths data to reproduce, prepare derivative works, and
* perform publicly and display publicly.  Beginning five (5) years from
* July 25, 2001, the United States Government is granted for itself and
* others acting on its behalf a paid-up, nonexclusive, irrevocable
* worldwide license in this data to reproduce, prepare derivative works,
* distribute copies to the public, perform publicly and display
* publicly, and to permit others to do so.
*
* NEITHER THE UNITED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT
* OF ENERGY, NOR SANDIA CORPORATION, NOR ANY OF THEIR EMPLOYEES, MAKES
* ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR
* RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY
* INFORMATION, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS
* THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.
*
* ************************************************************************
* @HEADER */

#ifndef TRILINOS_VERSION_H
#define TRILINOS_VERSION_H

/* The major version number xx (allows up 99 major Trilinos releases) */
#define TRILINOS_MAJOR_VERSION 8

/* The major and minor version numbers (i.e. xx.yy.zz) */
#define TRILINOS_MAJOR_MINOR_VERSION 80004

/* NOTE: These macros are given long int values to allow comparisons in
 * preprocessor #if statements.  For example, you can do comparisons
 * with ==, <, <=, >, and >=.
 */

/* NOTE: The C++ standard for the C preprocessor requires that the arguments
 * for #if must be convertible into a long int.  Expressions that convert to 1
 * are true and expressions that convert to 0 are false.
 */
 
/* NOTE: The major and major+minor version number for the development branch
 * of Trilinos will always be 00 and 000000 respectively.
 */
 
/* NOTE: All other version numbers can not start with a 0 followed by a
 * non-zero or they are interpreted as octal constants and that will cause a
 * compilation error.
 */

#endif /* TRILINOS_VERSION_H */
