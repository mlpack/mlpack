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

#ifndef _TEUCHOS_PROMOTION_TRAITS_HPP_
#define _TEUCHOS_PROMOTION_TRAITS_HPP_

#include "Teuchos_ConfigDefs.hpp"

namespace Teuchos {

template <class A, class B>
class PromotionTraits
{
public:
};

//Specialization
template <class T> class PromotionTraits<T,T> {
public:
  typedef T promote;
};

#define PT_SPEC(type1,type2,type3) \
template <> class PromotionTraits< type1 , type2 > { \
public: \
    typedef type3 promote; \
}; \
template <> class PromotionTraits< type2 , type1 > { \
public: \
    typedef type3 promote; \
};                                               

#if defined(HAVE_COMPLEX) && defined(HAVE_TEUCHOS_COMPLEX)
PT_SPEC(double,std::complex<float>,std::complex<double>)
PT_SPEC(float,std::complex<double>,std::complex<double>)
PT_SPEC(float,std::complex<float>,std::complex<float>)
PT_SPEC(double,std::complex<double>,std::complex<double>)
#endif
PT_SPEC(double,float,double)
PT_SPEC(double,long,double)
PT_SPEC(double,int,double)
PT_SPEC(float,long,float)
PT_SPEC(float,int,float)

// ToDo: Add specializations for extended precision types!

#undef PT_SPEC

} // Teuchos namespace

#endif // _TEUCHOS_PROMOTION_TRAITS_HPP_
