
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

#ifndef AZTECOO_STATUSTYPE_H
#define AZTECOO_STATUSTYPE_H
/*! \file AztecOO_StatusType.h 
    \brief AztecOO StatusType: Used to return convergence status information for AztecOO_StatusTest objects.
 */

/*! \enum AztecOO_StatusType 
    When the CheckStatus and GetStatus methods of AztecOO_StatusTest objects are called a variable
    of type AztecOO_StatusType is returned.
*/



enum AztecOO_StatusType { Unchecked = 2,        /*!< Initial state of status */
			  Unconverged = 1,      /*!< Convergence is not reached. */
			  Converged = 0,        /*!< Convergence is reached. */
			  Failed = -1,          /*!< Some failure occured.  Should stop */
			  NaN = -2,              /*!< Result from test contains a NaN value.  Should stop */
			  PartialFailed = -3   /*!< Some success, e.g. implicit residual converged but explicit did not. */
			  
};
#endif /* AZTECOO_STATUSTYPE_H */
