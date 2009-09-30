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

#ifndef IFPACK_CONDEST_H
#define IFPACK_CONDEST_H

#include "Ifpack_ConfigDefs.h"
#include "Ifpack_CondestType.h"
class Ifpack_Preconditioner;
class Epetra_RowMatrix;

double Ifpack_Condest(const Ifpack_Preconditioner& IFP,
		      const Ifpack_CondestType CT,
		      const int MaxIters = 1550,
		      const double Tol = 1e-9,
		      Epetra_RowMatrix* Matrix = 0);

#endif // IFPACK_CONDEST_H
