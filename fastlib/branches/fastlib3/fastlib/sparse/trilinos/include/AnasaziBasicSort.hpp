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

/*! \file AnasaziBasicSort.hpp
  \brief Basic implementation of the Anasazi::SortManager class
*/

#ifndef ANASAZI_BASIC_SORT_HPP
#define ANASAZI_BASIC_SORT_HPP

/*!    \class Anasazi::BasicSort
       \brief An implementation of the Anasazi::SortManager that performs a collection
       of common sorting techniques.

       \author Chris Baker, Ulrich Hetmaniuk, Rich Lehoucq, and Heidi Thornquist
*/

#include "AnasaziConfigDefs.hpp"
#include "AnasaziSortManager.hpp"
#include "Teuchos_LAPACK.hpp"
#include "Teuchos_ScalarTraits.hpp"

namespace Anasazi {

  template<class ScalarType, class MV, class OP>
  class BasicSort : public SortManager<ScalarType,MV,OP> {
    
  public:
    
    //! Constructor
    /**
       @param which [in] The eigenvalues of interest for this eigenproblem.
       <ul>
       <li> "LM" - Largest Magnitude [ default ]
       <li> "SM" - Smallest Magnitude
       <li> "LR" - Largest Real 
       <li> "SR" - Smallest Real 
       <li> "LI" - Largest Imaginary 
       <li> "SI" - Smallest Imaginary 
       </ul>
    */
    BasicSort( const std::string which = "LM" ) {
      setSortType(which);
    }

    //! Destructor
    virtual ~BasicSort() {};

    //! Set sort type
    /**
       @param which [in] The eigenvalues of interest for this eigenproblem.
       <ul>
       <li> "LM" - Largest Magnitude [ default ]
       <li> "SM" - Smallest Magnitude
       <li> "LR" - Largest Real 
       <li> "SR" - Smallest Real 
       <li> "LI" - Largest Imaginary 
       <li> "SI" - Smallest Imaginary 
       </ul>
    */
    void setSortType( const std::string which ) { 
      which_ = which; 
      TEST_FOR_EXCEPTION(which_.compare("LM") && which_.compare("SM") &&
                         which_.compare("LR") && which_.compare("SR") &&
                         which_.compare("LI") && which_.compare("SI"), std::invalid_argument, 
                         "Anasazi::BasicSort::sort(): sorting order is not valid");
    };
    
    //! Sort the vector of eigenvalues, optionally returning the permutation vector.
    /**
       @param solver [in] Eigensolver that is calling the sorting routine

       @param n [in] Number of values in evals to be sorted.

       @param evals [in/out] Vector of length n containing the eigenvalues to be sorted

       @param perm [out] Vector of length n to store the permutation index (optional)
    */
    void sort(Eigensolver<ScalarType,MV,OP>* solver, const int n, std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> &evals, std::vector<int> *perm = 0) const;

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
    void sort(Eigensolver<ScalarType,MV,OP>* solver, 
              const int n,
              std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> &r_evals, 
              std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> &i_evals, 
              std::vector<int> *perm = 0) const;
    
  protected: 
    
    //! Sorting type
    /*! \note Sorting choices:
       <ul>
       <li> "LM" - Largest Magnitude [ default ]
       <li> "SM" - Smallest Magnitude
       <li> "LR" - Largest Real 
       <li> "SR" - Smallest Real 
       <li> "LI" - Largest Imaginary 
       <li> "SI" - Smallest Imaginary 
       </ul>
    */
    std::string which_;

  };

  template<class ScalarType, class MV, class OP>
  void BasicSort<ScalarType,MV,OP>::sort(Eigensolver<ScalarType,MV,OP>* solver, const int n, 
                              std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> &evals, 
                              std::vector<int> *perm) const
  {
    int i=0,j=0;

    TEST_FOR_EXCEPTION(evals.size() < (unsigned int) n,
                       std::invalid_argument, "Anasazi::BasicSort:sort(): eigenvalue vector size isn't consistent with n.");
    if (perm) {
      TEST_FOR_EXCEPTION(perm->size() < (unsigned int) n,
                         std::invalid_argument, "Anasazi::BasicSort:sort(): permutation vector size isn't consistent with n.");
    }

    // Temp integer for swapping the index of the permutation, used in all sorting types.
    int tempord=0;

    typedef typename Teuchos::ScalarTraits<ScalarType>::magnitudeType MagnitudeType;
    typedef Teuchos::ScalarTraits<MagnitudeType> MT;

    // Temp variable for swapping the eigenvalue used in all sorting types.
    MagnitudeType temp;

    Teuchos::LAPACK<int,MagnitudeType> lapack;

    //
    // Reset the permutation if it is required.
    //
    if (perm) {
      for (i=0; i < n; i++) {
        (*perm)[i] = i;
      }
    }
    //
    // These methods use an insertion sort method to circumvent recursive calls.
    //---------------------------------------------------------------
    // Sort eigenvalues in increasing order of magnitude
    //---------------------------------------------------------------
    if (!which_.compare("SM")) {
      for (j=1; j < n; j++) {
        temp = evals[j]; 
        if (perm) {
          tempord = (*perm)[j];
        }
        MagnitudeType temp2 = MT::magnitude(evals[j]);
        for (i=j-1; i >=0 && MT::magnitude(evals[i]) > temp2; i--) {
          evals[i+1] = evals[i];
          if (perm) {
            (*perm)[i+1]=(*perm)[i];
          }
        }
        evals[i+1] = temp; 
        if (perm) {
          (*perm)[i+1] = tempord;
        }
      }
      return;
    }
    //---------------------------------------------------------------
    // Sort eigenvalues in increasing order of real part
    //---------------------------------------------------------------
    if (!which_.compare("SR")) {
      for (j=1; j < n; j++) {
        temp = evals[j]; 
        if (perm) {
          tempord = (*perm)[j];
        }
        for (i=j-1; i >= 0 && evals[i] > temp; i--) {
          evals[i+1]=evals[i];
          if (perm) {
            (*perm)[i+1]=(*perm)[i];
          }
        }
        evals[i+1] = temp; 
        if (perm) {
          (*perm)[i+1] = tempord;
        }
      }
      return;
    }
    //---------------------------------------------------------------
    // Sort eigenvalues in increasing order of imaginary part
    // NOTE:  There is no implementation for this since this sorting
    // method assumes only real eigenvalues.
    //---------------------------------------------------------------
    TEST_FOR_EXCEPTION(!which_.compare("SI"), SortManagerError, 
                       "Anasazi::BasicSort::sort() with one arg assumes real eigenvalues");
    //---------------------------------------------------------------
    // Sort eigenvalues in decreasing order of magnitude
    //---------------------------------------------------------------
    if (!which_.compare("LM")) {
      for (j=1; j < n; j++) {
        temp = evals[j]; 
        if (perm) {
          tempord = (*perm)[j];
        }
        MagnitudeType temp2 = MT::magnitude(evals[j]);
        for (i=j-1; i >= 0 && MT::magnitude(evals[i]) < temp2; i--) {
          evals[i+1]=evals[i];
          if (perm) {
            (*perm)[i+1]=(*perm)[i];
          }
        }
        evals[i+1] = temp; 
        if (perm) {
          (*perm)[i+1] = tempord;
        }
      }
      return;
    }
    //---------------------------------------------------------------
    // Sort eigenvalues in decreasing order of real part
    //---------------------------------------------------------------
    if (!which_.compare("LR")) {
      for (j=1; j < n; j++) {
        temp = evals[j]; 
        if (perm) {
          tempord = (*perm)[j];
        }
        for (i=j-1; i >= 0 && evals[i]<temp; i--) {
          evals[i+1]=evals[i];
          if (perm) {
            (*perm)[i+1]=(*perm)[i];
          }
        }
        evals[i+1] = temp; 
        if (perm) {
          (*perm)[i+1] = tempord;
        }
      }
      return;
    }
    //---------------------------------------------------------------
    // Sort eigenvalues in decreasing order of imaginary part
    // NOTE:  There is no implementation for this since this templating
    // assumes only real eigenvalues.
    //---------------------------------------------------------------
    TEST_FOR_EXCEPTION(!which_.compare("LI"), SortManagerError, 
                       "Anasazi::BasicSort::sort() with one arg assumes real eigenvalues");
    
    // The character string held by this class is not valid.  
    TEST_FOR_EXCEPTION(true, std::logic_error, 
                       "Anasazi::BasicSort::sort(): sorting order is not valid");
  }


  template<class ScalarType, class MV, class OP>
  void BasicSort<ScalarType,MV,OP>::sort(Eigensolver<ScalarType,MV,OP>* solver, 
                                         const int n,
                                         std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> &r_evals, 
                                         std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> &i_evals, 
                                         std::vector<int> *perm) const 
  {
    typedef typename Teuchos::ScalarTraits<ScalarType>::magnitudeType MagnitudeType;
    typedef Teuchos::ScalarTraits<MagnitudeType> MT;

    TEST_FOR_EXCEPTION(r_evals.size() < (unsigned int) n || i_evals.size() < (unsigned int) n,
                       std::invalid_argument, "Anasazi::BasicSort:sort(): real and imaginary vector sizes aren't consistent with n.");
    if (perm) {
      TEST_FOR_EXCEPTION(perm->size() < (unsigned int) n,
                         std::invalid_argument, "Anasazi::BasicSort:sort(): permutation vector size isn't consistent with n.");
    }
    int i=0,j=0;
    int tempord=0;

    MagnitudeType temp, tempr, tempi;
    Teuchos::LAPACK<int,MagnitudeType> lapack;
    //
    // Reset the index
    //
    if (perm) {
      for (i=0; i < n; i++) {
        (*perm)[i] = i;
      }
    }
    //
    // These methods use an insertion sort method to circumvent recursive calls.
    //---------------------------------------------------------------
    // Sort eigenvalues in increasing order of magnitude
    //---------------------------------------------------------------
    if (!which_.compare("SM")) {
      for (j=1; j < n; j++) {
        tempr = r_evals[j]; tempi = i_evals[j]; 
        if (perm) {
          tempord = (*perm)[j];
        }
        temp=lapack.LAPY2(r_evals[j],i_evals[j]);
        for (i=j-1; i>=0 && lapack.LAPY2(r_evals[i],i_evals[i]) > temp; i--) {
          r_evals[i+1]=r_evals[i]; i_evals[i+1]=i_evals[i];
          if (perm) {
            (*perm)[i+1]=(*perm)[i];
          }
        }
        r_evals[i+1] = tempr; i_evals[i+1] = tempi; 
        if (perm) {
          (*perm)[i+1] = tempord;
        }
      }
      return;
    }
    //---------------------------------------------------------------
    // Sort eigenvalues in increasing order of real part
    //---------------------------------------------------------------
    if (!which_.compare("SR")) {
      for (j=1; j < n; j++) {
        tempr = r_evals[j]; tempi = i_evals[j]; 
        if (perm) {
          tempord = (*perm)[j];
        }
        for (i=j-1; i>=0 && r_evals[i]>tempr; i--) {
          r_evals[i+1]=r_evals[i]; i_evals[i+1]=i_evals[i];
          if (perm) {
            (*perm)[i+1]=(*perm)[i];
          }
        }
        r_evals[i+1] = tempr; i_evals[i+1] = tempi; 
        if (perm) {
          (*perm)[i+1] = tempord;
        }
      }
      return;
    }
    //---------------------------------------------------------------
    // Sort eigenvalues in increasing order of imaginary part
    //---------------------------------------------------------------
    if (!which_.compare("SI")) {
      for (j=1; j < n; j++) {
        tempr = r_evals[j]; tempi = i_evals[j]; 
        if (perm) {
          tempord = (*perm)[j];
        }
        for (i=j-1; i>=0 && i_evals[i]>tempi; i--) {
          r_evals[i+1]=r_evals[i]; i_evals[i+1]=i_evals[i];
          if (perm) {
            (*perm)[i+1]=(*perm)[i];
          }
        }
        r_evals[i+1] = tempr; i_evals[i+1] = tempi; 
        if (perm) {
          (*perm)[i+1] = tempord;
        }
      }
      return;
    }
    //---------------------------------------------------------------
    // Sort eigenvalues in decreasing order of magnitude
    //---------------------------------------------------------------
    if (!which_.compare("LM")) {
      for (j=1; j < n; j++) {
        tempr = r_evals[j]; tempi = i_evals[j]; 
        if (perm) {
          tempord = (*perm)[j];
        }
        temp=lapack.LAPY2(r_evals[j],i_evals[j]);
        for (i=j-1; i>=0 && lapack.LAPY2(r_evals[i],i_evals[i])<temp; i--) {
          r_evals[i+1]=r_evals[i]; i_evals[i+1]=i_evals[i];
          if (perm) {
            (*perm)[i+1]=(*perm)[i];
          }
        }
        r_evals[i+1] = tempr; i_evals[i+1] = tempi; 
        if (perm) {
          (*perm)[i+1] = tempord;
        }
      }        
      return;
    }
    //---------------------------------------------------------------
    // Sort eigenvalues in decreasing order of real part
    //---------------------------------------------------------------
    if (!which_.compare("LR")) {
      for (j=1; j < n; j++) {
        tempr = r_evals[j]; tempi = i_evals[j]; 
        if (perm) {
          tempord = (*perm)[j];
        }
        for (i=j-1; i>=0 && r_evals[i]<tempr; i--) {
          r_evals[i+1]=r_evals[i]; i_evals[i+1]=i_evals[i];
          if (perm) {
            (*perm)[i+1]=(*perm)[i];
          }
        }
        r_evals[i+1] = tempr; i_evals[i+1] = tempi; 
        if (perm) {
          (*perm)[i+1] = tempord;
        }
      }        
      return;
    }
    //---------------------------------------------------------------
    // Sort eigenvalues in decreasing order of imaginary part
    //---------------------------------------------------------------
    if (!which_.compare("LI")) {
      for (j=1; j < n; j++) {
        tempr = r_evals[j]; tempi = i_evals[j]; 
        if (perm) {
          tempord = (*perm)[j];
        }
        for (i=j-1; i>=0 && i_evals[i]<tempi; i--) {
          r_evals[i+1]=r_evals[i]; i_evals[i+1]=i_evals[i];
          if (perm) {
            (*perm)[i+1]=(*perm)[i];
          }
        }
        r_evals[i+1] = tempr; i_evals[i+1] = tempi; 
        if (perm) {
          (*perm)[i+1] = tempord;
        }
      }
      return;
    }

    TEST_FOR_EXCEPTION(true, std::logic_error, 
                       "Anasazi::BasicSort::sort(): sorting order is not valid");
  }
  
  
} // namespace Anasazi

#endif // ANASAZI_BASIC_SORT_HPP

