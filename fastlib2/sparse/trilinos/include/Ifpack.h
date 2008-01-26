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

#ifndef IFPACK_H
#define IFPACK_H

#include "Ifpack_ConfigDefs.h"
#include "Ifpack_Preconditioner.h"

//! Ifpack: a function class to define Ifpack preconditioners.
/*!
Class Ifpack is a function class, that contains just one method:
Create(). Using Create(), users can easily define a variety of 
IFPACK preconditioners. 

Create requires 3 arguments:
- a string, indicating the preconditioner to be built;
- a pointer to an Epetra_RowMatrix, representing the matrix
  to be used to define the preconditioner;
- an interger (defaulted to 0), that specifies the amount of
  overlap among the processes.

The first argument can assume the following values:
- \c "point relaxation" : returns an instance of Ifpack_AdditiveSchwarz<Ifpack_PointRelaxation>
- \c "point relaxation stand-alone" : returns an instance of Ifpack_PointRelaxation (value of overlap is ignored).
- \c "block relaxation" : returns an instance of Ifpack_AdditiveSchwarz<Ifpack_BlockRelaxation>
- \c "block relaxation stand-alone)" : returns an instance of Ifpack_BlockRelaxation.
- \c "Amesos" : returns an instance of Ifpack_AdditiveSchwarz<Ifpack_Amesos>.
- \c "Amesos" : returns an instance of Ifpack_Amesos.
- \c "IC" : returns an instance of Ifpack_AdditiveSchwarz<Ifpack_IC>.
- \c "IC stand-alone" : returns an instance of Ifpack_AdditiveSchwarz<Ifpack_IC>.
- \c "ICT" : returns an instance of Ifpack_AdditiveSchwarz<Ifpack_ICT>.
- \c "ICT stand-alone" : returns an instance of Ifpack_ICT.
- \c "ILU" : returns an instance of Ifpack_AdditiveSchwarz<Ifpack_ILU>.
- \c "ILU stand-alone" : returns an instance of Ifpack_ILU.
- \c "ILUT" : returns an instance of Ifpack_AdditiveSchwarz<Ifpack_ILUT>.
- \c "ILUT stand-alone" : returns an instance of Ifpack_ILUT.
- otherwise, Create() returns 0.

\note Objects in stand-alone mode cannot use reordering, variable overlap, and singleton filters.
However, their construction can be slightly faster than the non stand-alone counterpart. 

<P> The following fragment of code shows the
basic usage of this class.
\code
#include "Ifpack.h"

...

Ifpack Factory;

Epetra_RowMatrix* A; // A is FillComplete()'d.
string PrecType = "ILU"; // use incomplete LU on each process
int OverlapLevel = 1; // one row of overlap among the processes
Ifpack_Preconditioner* Prec = Factory.Create(PrecType, A, OverlapLevel);
assert (Prec != 0);

Teuchos::ParameterList List;
List.set("fact: level-of-fill", 5); // use ILU(5)

IFPACK_CHK_ERR(Prec->SetParameters(List));
IFPACK_CHK_ERR(Prec->Initialize());
IFPACK_CHK_ERR(Prec->Compute());

// now Prec can be used as AztecOO preconditioner
// like for instance
AztecOO AztecOOSolver(*Problem);

// specify solver
AztecOOSolver.SetAztecOption(AZ_solver,AZ_gmres);
AztecOOSolver.SetAztecOption(AZ_output,32);

// Set Prec as preconditioning operator
AztecOOSolver.SetPrecOperator(Prec);

// Call the solver
AztecOOSolver.Iterate(1550,1e-8);

// print information on stdout
cout << *Prec;

// delete the preconditioner
delete Prec;
\endcode

\author Marzio Sala, (formally) SNL org. 1414

\date Last updated on 25-Jan-05.
*/

class Ifpack {
public:

  /** \brief Enum for the type of preconditioner. */
  enum EPrecType {
    POINT_RELAXATION
    ,POINT_RELAXATION_STAND_ALONE
    ,BLOCK_RELAXATION
    ,BLOCK_RELAXATION_STAND_ALONE
    ,BLOCK_RELAXATION_STAND_ALONE_ILU
#ifdef HAVE_IFPACK_AMESOS
    ,BLOCK_RELAXATION_STAND_ALONE_AMESOS
    ,BLOCK_RELAXATION_AMESOS
    ,AMESOS
    ,AMESOS_STAND_ALONE
#endif // HAVE_IFPACK_AMESOS
    ,IC
    ,IC_STAND_ALONE
    ,ICT
    ,ICT_STAND_ALONE
    ,ILU
    ,ILU_STAND_ALONE
    ,ILUT
    ,ILUT_STAND_ALONE
#ifdef HAVE_IFPACK_SPARSKIT
    ,SPARSKIT
#endif // HAVE_IFPACK_SPARSKIT
    ,CHEBYSHEV
  };

  /** \brief . */
  static const int numPrecTypes =
    +5
#ifdef HAVE_IFPACK_AMESOS
    +4
#endif
    +8
#ifdef HAVE_IFPACK_SPARSKIT
    +1
#endif
    +1
    ;

  /** \brief List of the preconditioner types as enum values . */
  static const EPrecType precTypeValues[numPrecTypes];

  /** \brief List of preconditioner types as string values. */
  static const char* precTypeNames[numPrecTypes];

  /** \brief List of bools that determines if the precondtioner type supports
   * unsymmetric matrices. */
  static const bool supportsUnsymmetric[numPrecTypes];

  /** \brief Function that gives the string name for preconditioner given its
   * enumerication value. */
  static const char* toString(const EPrecType precType)
      { return precTypeNames[precType]; }

  /** \brief Creates an instance of Ifpack_Preconditioner given the enum value
   * of the preconditioner type (can not fail, no bad input possible).
   *
   * \param PrecType (In) - Enum value of preconditioner type to be created. 
   *
   * \param Matrix (In) - Matrix used to define the preconditioner
   *
   * \param overlap (In) - specified overlap, defaulted to 0.
   */
  static Ifpack_Preconditioner* Create(
    EPrecType PrecType, Epetra_RowMatrix* Matrix, const int overlap = 0
    );

  /** \brief Creates an instance of Ifpack_Preconditioner given the string
   * name of the preconditioner type (can fail with bad input).
   *
   * \param PrecType (In) - String name of preconditioner type to be created. 
   *
   * \param Matrix (In) - Matrix used to define the preconditioner
   *
   * \param overlap (In) - specified overlap, defaulted to 0.
   *
   * Returns <tt>0</tt> if the preconditioner with that input name does not
   * exist.  Otherwise, return a newly created preconditioner object.  Note
   * that the client is responsible for calling <tt>delete</tt> on the
   * returned object once it is finished using it!
   */
  Ifpack_Preconditioner* Create(const string PrecType,
				Epetra_RowMatrix* Matrix,
				const int overlap = 0);

  /** \brief Sets the options in List from the command line.
   *
   * Note: If you want full support for all parameters, consider reading in a
   * parameter list from an XML file as supported by the Teuchos helper
   * function <tt>Teuchos::updateParametersFromXmlFile()</tt> or
   * <tt>Teuchos::updateParametersFromXmlStream()</tt>.
   */
  int SetParameters(int argc, char* argv[],
                    Teuchos::ParameterList& List, string& PrecType,
                    int& Overlap);

};

#endif
