
//@HEADER
/*
************************************************************************

              Epetra: Linear Algebra Services Package 
                Copyright (2001) Sandia Corporation

Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
license for use of this work by or on behalf of the U.S. Government.

This library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation; either version 2.1 of the
License, or (at your option) any later version.
 
This library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
 
You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
USA
Questions? Contact Michael A. Heroux (maherou@sandia.gov) 

************************************************************************
*/
//@HEADER

#ifndef EPETRA_COMPOBJECT_H
#define EPETRA_COMPOBJECT_H

//! Epetra_CompObject: Functionality and data that is common to all computational classes.

/*! The Epetra_CompObject is a base class for all Epetra computational objects.  It provides the basic
    mechanisms and interface specifications for floating point operations using Epetra_Flops objects.

*/
#include "Epetra_Object.h"
#include "Epetra_Flops.h"
//==========================================================================
class Epetra_CompObject {

  public:

    //! @name Constructors/Destructor
  //@{ 
  //! Basic Epetra_CompObject constuctor.
  Epetra_CompObject();

  //! Epetra_CompObject copy constructor.
  
  Epetra_CompObject(const Epetra_CompObject& Source);
  
  
  //! Epetra_CompObject destructor.  
  virtual ~Epetra_CompObject();
  //@}

  //! @name Set/Get counter method
  //@{ 
  //! Set the internal Epetra_Flops() pointer.
  void SetFlopCounter(const Epetra_Flops & FlopCounter) {FlopCounter_= (Epetra_Flops *) &FlopCounter; return;}
  //! Set the internal Epetra_Flops() pointer to the flop counter of another Epetra_CompObject.
  void SetFlopCounter(const Epetra_CompObject & CompObject) {FlopCounter_= (Epetra_Flops *) (CompObject.GetFlopCounter()); return;}
  //! Set the internal Epetra_Flops() pointer to 0 (no flops counted).
  void UnsetFlopCounter() {FlopCounter_= 0; return;}
  //! Get the pointer to the  Epetra_Flops() object associated with this object, returns 0 if none.
  Epetra_Flops * GetFlopCounter() const {return(FlopCounter_);}
  //@}

  //! @name Set flop count methods
  //@{ 
  //! Resets the number of floating point operations to zero for \e this multi-vector.
  void ResetFlops() const {if (FlopCounter_!=0) FlopCounter_->ResetFlops(); return;}

  //! Returns the number of floating point operations with \e this multi-vector.
  double Flops() const {if (FlopCounter_!=0) return(FlopCounter_->Flops()); else return(0.0);}
  //@}

  //! @name Update flop count methods
  //@{ 
  //! Increment Flop count for \e this object
  void UpdateFlops(int Flops) const {if (FlopCounter_!=0) FlopCounter_->UpdateFlops(Flops); return;}

  //! Increment Flop count for \e this object
  void UpdateFlops(long int Flops) const {if (FlopCounter_!=0) FlopCounter_->UpdateFlops(Flops); return;}

  //! Increment Flop count for \e this object
  void UpdateFlops(double Flops) const {if (FlopCounter_!=0) FlopCounter_->UpdateFlops(Flops); return;}

  //! Increment Flop count for \e this object
  void UpdateFlops(float Flops) const {if (FlopCounter_!=0) FlopCounter_->UpdateFlops(Flops); return;}
  //@}

  Epetra_CompObject& operator=(const Epetra_CompObject& src)
    {
      FlopCounter_ = src.FlopCounter_;
      return(*this);
    }

 protected:


  Epetra_Flops * FlopCounter_;

};

#endif /* EPETRA_COMPOBJECT_H */
