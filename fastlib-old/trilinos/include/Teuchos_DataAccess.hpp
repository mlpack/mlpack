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

// Kris
// 07.08.03 -- Move into Teuchos package/namespace

#ifndef _TEUCHOS_DATAACCESS_HPP_
#define _TEUCHOS_DATAACCESS_HPP_

/*! \file Teuchos_DataAccess.hpp 
    \brief Teuchos::DataAccess Mode enumerable type
*/

namespace Teuchos {
	
	/*! \enum DataAccess
      If set to Copy, user data will be copied at construction.
      If set to View, user data will be encapsulated and used throughout
      the life of the object.
	*/
	
	enum DataAccess {
		Copy, /*!< User data will be copied at construction. */
		View /*!< User data will be encapsulated and used throughout the life of the object. */
	};

} // namespace Teuchos

#endif /* _TEUCHOS_DATAACCESS_HPP_ */
