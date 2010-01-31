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

#ifndef TEUCHOS_MPITRAITS_H
#define TEUCHOS_MPITRAITS_H

/*! \file Teuchos_MPITraits.hpp
 * \brief Declaration of a templated traits class for binding MPI types to
 * C++ types. This is for use with the MPIComm class and is supposed to compile
 * rgeardless of whether MPI has been enabled. If you need to convert directly to 
 * MPI types (e.g., MPI_INT), please refer to Teuchos_MPIRawTraits.hpp.
*/

#include "Teuchos_MPIComm.hpp"

namespace Teuchos
{
	using std::string;

	/** \ingroup MPI 
	 * \brief Templated traits class that binds MPI types to C++ types
	 * \note Template specializations exist for datatypes: <tt>char</tt>,
		<tt>int</tt>, <tt>float</tt>, and <tt>double</tt>.
	 */
	template <class T> class MPITraits
		{
		public:
			/** \brief Return the MPI data type of the template argument */
			static int type();
		};

#ifndef DOXYGEN_SHOULD_SKIP_THIS	
	/** \ingroup MPI 
	 * Binds MPI_INT to int
	 */
	template <> class MPITraits<int>
		{
		public:
			/** return the MPI data type of the template argument */
			static int type() {return MPIComm::INT;}
		};
	
	/** \ingroup MPI 
	 * Binds MPI_FLOAT to float
	 */
	template <> class MPITraits<float>
		{
		public:
			/** return the MPI data type of the template argument */
			static int type() {return MPIComm::FLOAT;}
		};
	
	/** \ingroup MPI 
	 * Binds MPI_DOUBLE to double
	 */
	template <> class MPITraits<double>
		{
		public:
			/** return the MPI data type of the template argument */
			static int type() {return MPIComm::DOUBLE;}
		};
	
	/** \ingroup MPI 
	 * Binds MPI_CHAR to char
	 */
	template <> class MPITraits<char>
		{
		public:
			/** return the MPI data type of the template argument */
			static int type() {return MPIComm::CHAR;}
		};

#endif //DOXYGEN_SHOULD_SKIP_THIS
	
} // namespace Teuchos

#endif
