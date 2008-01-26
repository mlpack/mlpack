
//@HEADER
/*
************************************************************************

              Epetra: Linear Algebra Services Package 
                Copyright (2001) Sandia Corporation

Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
license for use of this work by or on behalf of the U.S. Government.

This library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation; either version 2.1 of the
License, or (at your option) any later version.
 
This library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
 
You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
USA
Questions? Contact Michael A. Heroux (maherou@sandia.gov) 

************************************************************************
*/
//@HEADER

#ifndef EPETRA_COMBINEMODE_H
#define EPETRA_COMBINEMODE_H
/*! \file Epetra_CombineMode.h 
    \brief Epetra_Combine Mode enumerable type
 */

/*! \enum Epetra_CombineMode 
    If set to Add, components on the receiving processor will be added
    together.    If set to Zero, off-processor components will be ignored.
    If set to Insert, off-processor components will replace existing
    components on the receiving processor.  If set to InsertAdd, off-processor components
    will replace existing components, but multiple off-processor contributions will be added.
    If set to Average, off-processor components will be averaged with
    existing components on the receiving processor. (Recursive Binary Average)
    If set to AbsMax, magnitudes of off-processor components will be maxed
    with magnitudes of existing components of the receiving processor.
    { V = Supported by Epetra_Vector and Epetra_MultiVector,
      M = Supported by Epetra_CrsMatrix and Epetra_VbrMatrix }
*/

enum Epetra_CombineMode {Add,    /*!< Components on the receiving processor
                                     will be added together. (V,M) */
                        Zero,   /*!< Off-processor components will be
                                     ignored. (V,M) */
                        Insert, /*!< Off-processor components will
                                     be inserted into locations on
                                     receiving processor replacing existing values. (V,M) */
                        InsertAdd, /*!< Off-processor components will
                                     be inserted into locations on
                                     receiving processor replacing existing values. (V,M) */
                        Average,/*!< Off-processor components will be
                                     averaged with existing components 
                                     on the receiving processor. (V) */
                        AbsMax  /*!< Magnitudes of Off-processor components will be
                                     maxed with magnitudes of existing components 
                                     on the receiving processor. (V) */
                        };

#endif // EPETRA_COMBINEMODE_H
