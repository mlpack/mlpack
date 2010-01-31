// @HEADER
// ***********************************************************************
// 
//                    Teuchos: Common Tools Package
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

#ifndef Teuchos_XMLPARAMETERLISTREADER_H
#define Teuchos_XMLPARAMETERLISTREADER_H

/*! \file Teuchos_XMLParameterListReader.hpp
    \brief Writes an XML object to a parameter list
*/

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLObject.hpp"
#include "Teuchos_Utils.hpp"

namespace Teuchos
{

	/** \ingroup XML 
	 * \brief Writes an XML object to a parameter list
	 */

	class XMLParameterListReader
		{
		public:
      //! @name Constructors 
			//@{
      /** Construct a reader */
      XMLParameterListReader();
			//@}

      /** Write the given XML object to a parameter list */
      ParameterList toParameterList(const XMLObject& xml) const ;
	
		private:

		};
}
#endif

