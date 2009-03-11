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

#ifndef TEUCHOS_UTILS_H
#define TEUCHOS_UTILS_H

/*! \file Teuchos_Utils.hpp
    \brief A utilities class for Teuchos
*/

#include "Teuchos_ConfigDefs.hpp"

/*! \class Teuchos::Utils
    \brief This class provides some basic std::string and floating-point utilities for Teuchos
*/

namespace Teuchos
{
  using std::string;

  class Utils
    {
    public:

      /** \brief print a description of the current build. */
      static void aboutBuild();

      /** \brief Set a number to zero if it is less than
       * <tt>getChopVal()</tt>. */
      static double chop(const double& x);

      /** \brief Get the chopping value, below which numbers are considered to
       * be zero. */
      static double getChopVal() {return chopVal_;}

      /** \brief Set the chopping value, below which numbers are considered to
       * be zero. */
      static void setChopVal(double chopVal) {chopVal_ = chopVal;}

      /** \brief Determine if a char is whitespace or not. */
      static bool isWhiteSpace( const char c )
        { return ( c==' ' || c =='\t' || c=='\n' ); }

      /** \brief Trim whitespace from beginning and end of std::string. */
      static std::string trimWhiteSpace( const std::string& str );

      /** \brief Write a double as a std::string. */
      static std::string toString(const double& x);

      /** \brief Write an int as a std::string. */
      static std::string toString(const int& x);

      /** \brief Write an unsigned int as a std::string. */
      static std::string toString(const unsigned int& x);

      /** \brief pi. */
#ifdef M_PI
      static double pi() {return M_PI;}
#else
      static double pi() {return 3.14159265358979323846;}
#endif

      /** \brief Get a parallel file name extention . */
      static std::string getParallelExtension(
        int    procRank = -1
        ,int   numProcs = -1
        );

    private:
      static double chopVal_;
    };

  /** \relates Utils */
  inline std::string toString(const int& x) {return Utils::toString(x);}

  /** \relates Utils */
  inline std::string toString(const unsigned int& x) {return Utils::toString(x);}

  /** \relates Utils */
  inline std::string toString(const double& x) {return Utils::toString(x);}

  /** \relates Utils */
  inline std::string toString(const std::string& x) {return x;}

} // end namespace Teuchos

#endif


