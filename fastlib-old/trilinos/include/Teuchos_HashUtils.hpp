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

#ifndef TEUCHOS_HASHUTILS_H
#define TEUCHOS_HASHUTILS_H

/*! \file Teuchos_HashUtils.hpp
    \brief Utilities for generating hashcodes
*/

#include "Teuchos_ConfigDefs.hpp"

namespace Teuchos
{
  using std::string;

  /**
   * \ingroup Containers
   * \brief Utilities for generating hashcodes.
   */

  class HashUtils
    {
    public:
      /* Get the next prime in a sequence of hashtable sizes */
      static int nextPrime(int newCapacity);

    private:

      // sequence of primes generated via mathematica:
      // Table[Prime[Round[1.5^x]], {x, 8, 36}]
      static const int primeCount_;
      static const int primes_[];
      /*={101, 163, 271, 443, 733, 1187, 1907, 3061,
        4919, 7759, 12379, 19543, 30841, 48487, 75989,
        119089, 185971, 290347, 452027, 703657, 1093237,
        1695781, 2627993, 4067599, 6290467, 9718019,
        15000607, 23133937, 35650091};*/
    };

  /** \relates HashUtils 
      \brief Standard interface for getting the hash code of an object 
  */
  template <class T> int hashCode(const T& x);

  /** \relates HashUtils 
      \brief Get the hash code of an int 
  */
  template <> inline int hashCode(const int& x) 
    {
      return x;
    }

  /** \relates HashUtils  
      \brief Get the hash code of a double 
  */
  template <> inline int hashCode(const double& x)
    {
      return (int) x;
    }

  /** \relates HashUtils  
      \brief Get the hash code of a bool 
  */
  template <> inline int hashCode(const bool& x)
    {
      return (int) x;
    }


  /** \relates HashUtils 
      \brief Get the hash code of a std::string 
  */
  template <> inline int hashCode(const std::string& x)
    {
      const char* str = x.c_str();
      int len = x.length();
      int step = len/4 + 1;
      int base = 1;
      int rtn = 0;

      for (int i=0; i<len/2; i+=step)
        {
          rtn += base*(int) str[i];
          base *= 128;
          rtn += base*(int) str[len-i-1];
          base *= 128;
        }

      return rtn;
    }



}
#endif // TEUCHOS_HASHUTILS_H
