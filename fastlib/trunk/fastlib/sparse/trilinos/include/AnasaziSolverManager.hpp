// @HEADER
// ***********************************************************************
//
//                 Anasazi: Block Eigensolvers Package
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

#ifndef ANASAZI_SOLVERMANAGER_HPP
#define ANASAZI_SOLVERMANAGER_HPP

/*! \file AnasaziSolverManager.hpp
    \brief Pure virtual base class which describes the basic interface for a solver manager.
*/

#include "AnasaziConfigDefs.hpp"
#include "AnasaziTypes.hpp"
#include "AnasaziEigenproblem.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

/*! \class Anasazi::SolverManager
  \brief The Anasazi::SolverManager is a templated virtual base class that defines the
	basic interface that any solver manager will support.
*/

namespace Anasazi {

template<class ScalarType, class MV, class OP>
class SolverManager {
    
  public:

    //!@name Constructors/Destructor 
  //@{ 

  //! Empty constructor.
  SolverManager() {};

  //! Destructor.
  virtual ~SolverManager() {};
  //@}
  
  //! @name Accessor methods
  //@{ 

  virtual const Eigenproblem<ScalarType,MV,OP>& getProblem() const = 0;

  //@}

  //! @name Solver application methods
  //@{ 
    
  /*! \brief This method performs possibly repeated calls to the underlying eigensolver's iterate() routine
   * until the problem has been solved (as decided by the solver manager) or the solver manager decides to 
   * quit.
   *
   * \returns ::ReturnType specifying:
   *    - ::Converged: the eigenproblem was solved to the specification required by the solver manager.
   *    - ::Unconverged: the eigenproblem was not solved to the specification desired by the solver manager
  */
  virtual ReturnType solve() = 0;
  //@}
  
};

} // end Anasazi namespace

#endif /* ANASAZI_SOLVERMANAGER_HPP */
