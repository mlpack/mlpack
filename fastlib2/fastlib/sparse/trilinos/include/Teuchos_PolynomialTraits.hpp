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

#ifndef TEUCHOS_POLYNOMIAL_TRAITS_HPP
#define TEUCHOS_POLYNOMIAL_TRAITS_HPP

#include "Teuchos_RCP.hpp"

namespace Teuchos {

  //! Traits class for polynomial coefficients in Teuchos::Polynomial.
  /*!
   * This class provides traits for implementing Teuchos::Polynomial.  The
   * default template definition here will work for any scalar type.  Any other
   * coefficient type for Teuchos::Polynomial should provide a specialization
   * of this traits class for that type that mirrors the default definition 
   * below.
   */
  template <typename Scalar>
  class PolynomialTraits {
  public:

    //! Typename of coefficients
    typedef Scalar coeff_type;

    //! Typename of scalars
    typedef Scalar scalar_type;

    //! Clone a coefficient
    static inline Teuchos::RCP<coeff_type> clone(const coeff_type& c) {
      return Teuchos::rcp(new coeff_type(c));
    }

    //! Copy a coefficient
    static inline void copy(const coeff_type& x, coeff_type* y) {
      *y = x;
    }

    //! Assign a scalar to a coefficient
    static inline void assign(coeff_type* y, const scalar_type& alpha) {
     *y = alpha;
    }

    //! y = x + beta*y
    static inline void update(coeff_type* y, const coeff_type& x, 
			      const scalar_type& beta) {
      *y = x + beta*(*y);
    }

  }; // class PolynomialTraits

} // end namespace Teuchos

#endif  // TEUCHOS_POLYNOMIAL_TRAITS_HPP
