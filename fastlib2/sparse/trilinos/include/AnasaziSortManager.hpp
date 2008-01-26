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

#ifndef ANASAZI_SORTMANAGER_HPP
#define ANASAZI_SORTMANAGER_HPP

/*!     \file AnasaziSortManager.hpp
        \brief Virtual base class which defines the interface between an eigensolver and a class whose
	job is the sorting of the computed eigenvalues
*/

/*!    \class Anasazi::SortManager
       \brief Anasazi's templated pure virtual class for managing the sorting of 
       approximate eigenvalues computed by the eigensolver.

       A concrete implementation of this class is necessary.  The user can create
       their own implementation if those supplied are not suitable for their needs.

       \author Ulrich Hetmaniuk, Rich Lehoucq, and Heidi Thornquist
*/

#include "AnasaziConfigDefs.hpp"
#include "AnasaziTypes.hpp"
#include "Teuchos_TestForException.hpp"



namespace Anasazi {

  //! @name LOBPCG Exceptions
  //@{ 
  /** \brief SortManagerError is thrown when the Anasazi::SortManager is unable to sort the numbers,
   *  due to some failure of the sort method or error in calling it.
   */
  class SortManagerError : public AnasaziError
  {public: SortManagerError(const std::string& what_arg) : AnasaziError(what_arg) {}};

  //@}

  template<class ScalarType, class MV, class OP>
  class Eigensolver;

  template<class ScalarType, class MV, class OP>
  class SortManager {
    
  public:
    
    //! Default Constructor
    SortManager() {};

    //! Destructor
    virtual ~SortManager() {};

    //! Sort the vector of eigenvalues, optionally returning the permutation vector.
    /**
       @param solver [in] Eigensolver that is calling the sorting routine

       @param n [in] Number of values in evals to be sorted.

       @param evals [in/out] Vector of length n containing the eigenvalues to be sorted

       @param perm [out] Vector of length n to store the permutation index (optional)
    */
    virtual void sort(Eigensolver<ScalarType,MV,OP>* solver, const int n, std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> &evals, std::vector<int> *perm = 0) const = 0;

    /*! \brief Sort the vectors of eigenpairs, optionally returning the permutation vector.

       This routine takes two vectors, one for each part of a complex
       eigenvalue. This is helpful for solving real, non-symmetric eigenvalue
       problems.

       @param solver [in] Eigensolver that is calling the sorting routine

       @param n [in] Number of values in r_evals,i_evals to be sorted.

       @param r_evals [in/out] Vector of length n containing the real part of the eigenvalues to be sorted 

       @param i_evals [in/out] Vector of length n containing the imaginary part of the eigenvalues to be sorted 

       @param perm [out] Vector of length n to store the permutation index (optional)
    */
    virtual void sort(Eigensolver<ScalarType,MV,OP>* solver, 
                      const int n,
                      std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> &r_evals, 
                      std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> &i_evals, 
                      std::vector<int> *perm = 0) const = 0;
    
  };
  
}

#endif // ANASAZI_SORTMANAGER_HPP

