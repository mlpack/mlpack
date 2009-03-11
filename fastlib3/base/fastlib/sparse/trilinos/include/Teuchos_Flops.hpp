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

#ifndef TEUCHOS_FLOPS_HPP
#define TEUCHOS_FLOPS_HPP

/*! \file Teuchos_Flops.hpp 
    \brief Object for providing basic support and consistent interfaces for 
	counting/reporting floating-point operations performed in Teuchos computational
	classes.
*/

/*! \class Teuchos::Flops
    \brief The Teuchos Floating Point Operations Class.

    The Teuchos_Flops class provides basic support and consistent interfaces
    for counting and reporting floating point operations performed in 
    the Teuchos computational classes.  All classes based on the Teuchos::CompObject
    can count flops by the user creating an Teuchos::Flops object and calling the SetFlopCounter()
    method for an Teuchos_CompObject. 
*/

namespace Teuchos
{
class Flops
{    
  public:

    //! @name Constructor/Destructor.
  //@{ 

  //! Default Constructor.
  /*! Creates a Flops instance. This instance can be queried for
      the number of floating point operations performed for the associated
      \e this object.
  */
  Flops();

  //! Copy Constructor.
  /*! Makes an exact copy of an existing Flops instance.
  */
  Flops(const Flops &flops);

  //! Destructor.
  /*! Completely deletes a Flops object.
  */
  virtual ~Flops();

  //@}

  //! @name Accessor methods.
  //@{ 

  //! Returns the number of floating point operations with \e this object and resets the count.
  double flops() const { return flops_; }

  //@}

  //! @name Reset methods.
  //@{ 

  //! Resets the number of floating point operations to zero for \e this multi-std::vector.
  void resetFlops() {flops_ = 0.0;}

  //@}

  friend class CompObject;

 protected:

  mutable double flops_;

  //! @name Updating methods.
  //@{ 
  //! Increment Flop count for \e this object from an int
  void updateFlops(int addflops) const {flops_ += (double) addflops; }

  //! Increment Flop count for \e this object from a long int
  void updateFlops(long int addflops) const {flops_ += (double) addflops; }

  //! Increment Flop count for \e this object from a double
  void updateFlops(double addflops) const {flops_ += (double) addflops; }

  //! Increment Flop count for \e this object from a float
  void updateFlops(float addflops) const {flops_ += (double) addflops; }

  //@}

 private:
  
};

  // #include "Teuchos_Flops.cpp"

} // namespace Teuchos

#endif // end of TEUCHOS_FLOPS_HPP
