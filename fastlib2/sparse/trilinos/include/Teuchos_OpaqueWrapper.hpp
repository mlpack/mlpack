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

#ifndef TEUCHOS_OPAQUE_WRAPPER_HPP
#define TEUCHOS_OPAQUE_WRAPPER_HPP

#include "Teuchos_RCP.hpp"

//#define TEUCHOS_OPAQUE_WRAPPER_ANNOUNCE_FREE

#ifdef TEUCHOS_OPAQUE_WRAPPER_ANNOUNCE_FREE
#  include "Teuchos_VerboseObject.hpp"
#endif // TEUCHOS_OPAQUE_WRAPPER_ANNOUNCE_FREE

namespace Teuchos {

/** \brief Base class for wrapped opaque objects.
 *
 * This base class allows opaque objects to be wrapped by a real object that
 * you can then take an address off.  This is needed in order to wrap an
 * opaque object in a RCP for example.
 *
 * For example, MPI uses the opaque object idiom for handling things like
 * MPI_Comm, and MPI_Op.  Some implementations implement these opaque object
 * handles and just integers.  This causes many problems with used with the
 * RCP (just try wrapping an MPI_Comm object directly in a RCP
 * and see what happens yourself and see what happens).
 *
 * For example, to wrap MPI_COMM_WORLD in a RCP, you would do
 * <tt>opaqueWrapper(MPI_COMM_WORLD)</tt> and that is it.
 *
 * Consider what would happen if you tried to directly wrap the MPI_Comm
 * MPI_COMM_WORLD in a RCP.  On some implementations like MPICH,
 * MPI_Comm is just a typedef to an integer and MPI_COMM_WORLD is just a define
 * to a literal interger.  In this case, the expression
 * <tt>rcp(&MPI_COMM_WORLD)</tt> would not even compile (try this on your
 * version of MPICH).  To make this compile, we might try something like:
 
 \code

  Teuchos::RCP<MPI_Comm> getMpiCommPtr()
  {
    MPI_Comm comm = MPI_COMM_WORLD;
    return Teuchos::rcp(&comm,false);
  }

 \endcode

 * Of course the above code would result in a disaster when the stack variable
 * <tt>comm</tt>, which is just an integer in MPICH, was destroyed and
 * reclaimed.  The <tt>RCP</tt> returned from getMpiCommPtr() would
 * contain a raw pointer an <tt>int</tt> object that was now being used for
 * something else and would no longer have the integer value of a valid
 * MPI_Comm object.
 *
 * The following implementation would most likely work but is pretty ugly:
 
 \code

  Teuchos::RCP<MPI_Comm> getMpiCommPtr()
  {
    MPI_Comm *comm = new MPI_Comm(MPI_COMM_WORLD);
    return Teuchos::rcp(&comm);
  }

 \endcode

 * The above implementation of getMPiCommPtr() would work with MPICH but it is
 * unclear how this would work with other implementations of MPI (but it
 * should work for these also).  However, this is pretty ugly to do.
 *
 * There are other issues that crop up also when you play these types of games.
 *
 * Therefore, just use <tt>opaqueWrapper()</tt> create wrap opaque objects in
 * RCP objects and be sure that this will go smoothly.
 *
 * \ingroup teuchos_mem_mng_grp
 */
template <class Opaque>
class OpaqueWrapper {
public:
  /** \brief . */
  OpaqueWrapper( Opaque opaque )
    : opaque_(opaque)
    {}
  /** \brief . */
  operator Opaque () const
    { return opaque_; }
  /** \brief . */
  Opaque operator()() const
    { return opaque_; }
protected:
  Opaque  opaque_; // Bad in general but ...
private:
  OpaqueWrapper(); // Not defined
  OpaqueWrapper(const OpaqueWrapper&); // Not defined
  OpaqueWrapper& operator=(const OpaqueWrapper&); // Not defined
}; 

/** \brief Subclass for wrapped opaque objects with a free function.
 *
 * This subclass allows a client to easily wrap any opaque object that needs a
 * function to free it.  This function (or function object) must be callable
 * as:

 \code
  opaqueFree(&opaque);
 \endcode

 * Again, this is typical for the opaque objects implemented in MPI for
 * instance.  For example, in order to delete an MPI_Comm object created by
 * the user (not MPI_COMM_WORLD), you must call the function MPI_Comm_free().
 *
 * \relates OpaqueWrapper
 */
template <class Opaque, class OpaqueFree>
class OpaqueWrapperWithFree : public OpaqueWrapper<Opaque> {
public:
  OpaqueWrapperWithFree( Opaque opaque, OpaqueFree opaqueFree )
    : OpaqueWrapper<Opaque>(opaque), opaqueFree_(opaqueFree)
    {}
  ~OpaqueWrapperWithFree()
    {
      if(opaqueFree_) {
#ifdef TEUCHOS_OPAQUE_WRAPPER_ANNOUNCE_FREE
        Teuchos::RCP<Teuchos::FancyOStream>
          out = Teuchos::VerboseObjectBase::getDefaultOStream();
        Teuchos::OSTab tab(out);
        *out << "\nOpaqueWrapperWithFree::~OpaqueWrapperWithFree(): Freeing opaque object"
             << " of type " << TypeNameTraits<Opaque>::name() << "!\n";
#endif // TEUCHOS_OPAQUE_WRAPPER_ANNOUNCE_FREE
        opaqueFree_(&this->opaque_);
      }
    }
private:
  OpaqueFree opaqueFree_;
  OpaqueWrapperWithFree(); // Not defined
  OpaqueWrapperWithFree(const OpaqueWrapperWithFree&); // Not defined
  OpaqueWrapperWithFree& operator=(const OpaqueWrapperWithFree&); // Not defined
}; 

/** \brief Helper function created a new <tt>OpaqueWrapper</tt> object without
 * a free function.
 *
 * \relates OpaqueWrapper
 */
template <class Opaque>
inline
RCP<OpaqueWrapper<Opaque> >
opaqueWrapper( Opaque opaque)
{
  return rcp(new OpaqueWrapper<Opaque>(opaque));
}

/** \brief Helper function created a new <tt>OpaqueWrapper</tt> object with a
 * free function.
 *
 * \relates OpaqueWrapper
 */
template <class Opaque, class OpaqueFree>
inline
RCP<OpaqueWrapper<Opaque> >
opaqueWrapper( Opaque opaque, OpaqueFree opaqueFree)
{
  return rcp(new OpaqueWrapperWithFree<Opaque,OpaqueFree>(opaque,opaqueFree));
}

} // namespace Teuchos

#endif	// TEUCHOS_OPAQUE_WRAPPER_HPP
