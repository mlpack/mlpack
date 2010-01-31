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

/*! \file AnasaziOperator.hpp
  \brief  Templated virtual class for creating operators that can interface with the Anasazi::OperatorTraits class
*/

#ifndef ANASAZI_OPERATOR_HPP
#define ANASAZI_OPERATOR_HPP

#include "AnasaziConfigDefs.hpp"
#include "AnasaziOperatorTraits.hpp"
#include "AnasaziMultiVec.hpp"
#include "Teuchos_ScalarTraits.hpp"


namespace Anasazi {
  
/*!	
	\brief Anasazi's templated virtual class for constructing an operator that can interface with the 
	OperatorTraits class used by the eigensolvers.
	
	A concrete implementation of this class is necessary.  The user can create their own implementation
	if those supplied are not suitable for their needs.

	\author Ulrich Hetmaniuk, Rich Lehoucq, and Heidi Thornquist
*/
  template <class ScalarType>
  class Operator {
  public:
    //! @name Constructor/Destructor
    //@{ 
    //! Default constructor.
    Operator() {};
    
    //! Destructor.
    virtual ~Operator() {};
    //@}
    
    //! @name Operator application method
    //@{ 
    
    /*! \brief This method takes the Anasazi::MultiVec \c x and
      applies the operator to it resulting in the Anasazi::MultiVec \c y.
    */
    virtual void Apply ( const MultiVec<ScalarType>& x, MultiVec<ScalarType>& y ) const = 0;

    //@}
  };
  
  ////////////////////////////////////////////////////////////////////
  //
  // Implementation of the Anasazi::OperatorTraits for Anasazi::Operator 
  //                                               and Anasazi::MultiVec.
  //
  ////////////////////////////////////////////////////////////////////  
  
  /*! 
    \brief Template specialization of Anasazi::OperatorTraits class using Anasazi::Operator and Anasazi::MultiVec virtual
    base classes.

    Any class that inherits from Anasazi::Operator will be accepted by the Anasazi templated solvers due to this
    interface to the Anasazi::OperatorTraits class.
  */

  template <class ScalarType> 
  class OperatorTraits < ScalarType, MultiVec<ScalarType>, Operator<ScalarType> > 
  {
  public:
  
    //! @name Operator application method
    //@{ 

    /*! \brief This method takes the Anasazi::MultiVec \c x and
      applies the Anasazi::Operator \c Op to it resulting in the Anasazi::MultiVec \c y.
    */
    static void Apply ( const Operator<ScalarType>& Op, 
			      const MultiVec<ScalarType>& x, 
			      MultiVec<ScalarType>& y )
    { Op.Apply( x, y ); }

    //@}
    
  };
  
} // end of Anasazi namespace

#endif

// end of file AnasaziOperator.hpp
