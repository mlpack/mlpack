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

#ifndef TEUCHOS_POLYNOMIAL_HPP
#define TEUCHOS_POLYNOMIAL_HPP

#include "Teuchos_PolynomialDecl.hpp"
#include "Teuchos_ScalarTraits.hpp"

template <typename CoeffT>
Teuchos::Polynomial<CoeffT>::Polynomial(unsigned int deg,
					const CoeffT& cloneCoeff, 
					unsigned int reserve) :
  d(deg)
{
  if (reserve > d)
    sz = reserve+1;
  else
    sz = d+1;

  coeff.resize(sz);
  for (unsigned int i=0; i<sz; i++)
    coeff[i] = PolynomialTraits<CoeffT>::clone(cloneCoeff);
}

template <typename CoeffT>
Teuchos::Polynomial<CoeffT>::Polynomial(unsigned int deg, 
					unsigned int reserve) :
  d(deg)
{
  if (reserve > d)
    sz = reserve+1;
  else
    sz = d+1;

  coeff.resize(sz);
}

template <typename CoeffT>
Teuchos::Polynomial<CoeffT>::~Polynomial()
{
}

template <typename CoeffT>
void 
Teuchos::Polynomial<CoeffT>::setDegree(unsigned int deg) 
{
  d = deg;
  if (d+1 > sz) {
    coeff.resize(d+1);
    if (coeff[0] != Teuchos::null) {
      for (unsigned int i=sz; i<d+1; i++)
	coeff[i] = PolynomialTraits<CoeffT>::clone(*coeff[0]);
    }
    sz = d+1;
  }
}

template <typename CoeffT>
Teuchos::RCP<CoeffT>
Teuchos::Polynomial<CoeffT>::getCoefficient(unsigned int i) 
{
#ifdef TEUCHOS_DEBUG
  TEST_FOR_EXCEPTION(i > d, 
		     std::out_of_range,
		     "Polynomial<CoeffT>::getCoefficient(i): " << 
		     "Error, coefficient i = " << i << 
		     " is not in range, degree = " << d << "." );
#endif
  return coeff[i];
}

template <typename CoeffT>
Teuchos::RCP<const CoeffT>
Teuchos::Polynomial<CoeffT>::getCoefficient(unsigned int i) const
{
#ifdef TEUCHOS_DEBUG
  TEST_FOR_EXCEPTION(i > d, 
		     std::out_of_range,
		     "Polynomial<CoeffT>::getCoefficient(i): " << 
		     "Error, coefficient i = " << i << 
		     " is not in range, degree = " << d << "." );
#endif
  return coeff[i];
}

template <typename CoeffT>
void
Teuchos::Polynomial<CoeffT>::setCoefficient(unsigned int i, const CoeffT& v) 
{
#ifdef TEUCHOS_DEBUG
  TEST_FOR_EXCEPTION(i > d, 
		     std::out_of_range,
		     "Polynomial<CoeffT>::setCoefficient(i,v): " << 
		     "Error, coefficient i = " << i << 
		     " is not in range, degree = " << d << "." );
  TEST_FOR_EXCEPTION(coeff[i] == Teuchos::null, 
		     std::runtime_error,
		     "Polynomial<CoeffT>::setCoefficient(i,v): " << 
		     "Error, coefficient i = " << i << " is null!");
#endif
  PolynomialTraits<CoeffT>::copy(v, coeff[i].get());
}

template <typename CoeffT>
void
Teuchos::Polynomial<CoeffT>::setCoefficientPtr(
	                          unsigned int i,
	                          const Teuchos::RCP<CoeffT>& v)
{
#ifdef TEUCHOS_DEBUG
  TEST_FOR_EXCEPTION(i > d, 
		     std::out_of_range,
		     "Polynomial<CoeffT>::setCoefficientPtr(i,v): " << 
		     "Error, coefficient i = " << i << 
		     " is not in range, degree = " << d << "." );
#endif
  coeff[i] = v;
}

template <typename CoeffT>
void
Teuchos::Polynomial<CoeffT>::evaluate(
			  typename Teuchos::Polynomial<CoeffT>::scalar_type t, 
			  CoeffT* x, CoeffT* xdot) const
{
  bool evaluate_xdot = (xdot != NULL);

#ifdef TEUCHOS_DEBUG
  for (unsigned int i=0; i<=d; i++)
    TEST_FOR_EXCEPTION(coeff[i] == Teuchos::null, 
		       std::runtime_error,
		       "Polynomial<CoeffT>::evaluate(): " << 
		       "Error, coefficient i = " << i << " is null!");
#endif

  // Initialize x, xdot with coeff[d]
  PolynomialTraits<CoeffT>::copy(*coeff[d], x);
  if (evaluate_xdot) {
    if (d > 0)
      PolynomialTraits<CoeffT>::copy(*coeff[d], xdot);
    else
      PolynomialTraits<CoeffT>::assign(
				 xdot, 
				 Teuchos::ScalarTraits<scalar_type>::zero());
  }

  // If this is a degree 0 polynomial, we're done
  if (d == 0)
    return;

  for (int k=d-1; k>=0; --k) {
    // compute x = coeff[k] + t*x
    PolynomialTraits<CoeffT>::update(x, *coeff[k], t);

    // compute xdot = x + t*xdot
    if (evaluate_xdot && k > 0)
      PolynomialTraits<CoeffT>::update(xdot, *x, t);
  }
}

#endif  // TEUCHOS_VECTOR_POLYNOMIAL_HPP
