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

#ifndef TEUCHOS_REDUCTION_OP_HELPERS_HPP
#define TEUCHOS_REDUCTION_OP_HELPERS_HPP

#include "Teuchos_ReductionOp.hpp"
#include "Teuchos_SerializationTraitsHelpers.hpp"
#include "Teuchos_SerializerHelpers.hpp"

namespace Teuchos {

/** \brief Decorator class that uses traits to convert to and from
 * <tt>char[]</tt> to typed buffers for objects that use value semantics and
 * then call a type-specific reduction object.
 *
 * ToDo: Finish Documentation!
 */
template<typename Ordinal, typename T>
class CharToValueTypeReductionOp : public ValueTypeReductionOp<Ordinal,char>
{
public:
  /** \brief . */
  CharToValueTypeReductionOp(
    const RCP<const ValueTypeReductionOp<Ordinal,T> >  &reductOp
    );
  /** \brief . */
  void reduce(
    const Ordinal     charCount
    ,const char       charInBuffer[]
    ,char             charInoutBuffer[]
    ) const;
private:
  RCP<const ValueTypeReductionOp<Ordinal,T> >  reductOp_;
  // Not defined and not to be called!
  CharToValueTypeReductionOp();
  CharToValueTypeReductionOp(const CharToValueTypeReductionOp&);
  CharToValueTypeReductionOp& operator=(const CharToValueTypeReductionOp&);
};

/** \brief Decorator class that uses a strategy object to convert to and from
 * <tt>char[]</tt> to typed buffers for objects that use reference semantics
 * and then call a type-specific reduction object.
 *
 * ToDo: Finish Documentation!
 */
template<typename Ordinal, typename T>
class CharToReferenceTypeReductionOp : public ValueTypeReductionOp<Ordinal,char>
{
public:
  /** \brief . */
  CharToReferenceTypeReductionOp(
    const RCP<const Serializer<Ordinal,T> >                 &serializer
    ,const RCP<const ReferenceTypeReductionOp<Ordinal,T> >  &reductOp
    );
  /** \brief . */
  void reduce(
    const Ordinal     charCount
    ,const char       charInBuffer[]
    ,char             charInoutBuffer[]
    ) const;
private:
  RCP<const Serializer<Ordinal,T> >                serializer_;
  RCP<const ReferenceTypeReductionOp<Ordinal,T> >  reductOp_;
  // Not defined and not to be called!
  CharToReferenceTypeReductionOp();
  CharToReferenceTypeReductionOp(const CharToReferenceTypeReductionOp&);
  CharToReferenceTypeReductionOp& operator=(const CharToReferenceTypeReductionOp&);
};

// /////////////////////////////////////
// Template implementations

//
// CharToValueTypeReductionOp
//

template<typename Ordinal, typename T>
CharToValueTypeReductionOp<Ordinal,T>::CharToValueTypeReductionOp(
  const RCP<const ValueTypeReductionOp<Ordinal,T> >  &reductOp
  )
  :reductOp_(reductOp)
{}

template<typename Ordinal, typename T>
void CharToValueTypeReductionOp<Ordinal,T>::reduce(
  const Ordinal     charCount
  ,const char       charInBuffer[]
  ,char             charInoutBuffer[]
  ) const
{
  ConstValueTypeDeserializationBuffer<Ordinal,T>
    inBuffer(charCount,charInBuffer);
  ValueTypeDeserializationBuffer<Ordinal,T>
    inoutBuffer(charCount,charInoutBuffer);
  reductOp_->reduce(
    inBuffer.getCount(),inBuffer.getBuffer(),inoutBuffer.getBuffer()
    );
}

//
// CharToReferenceTypeReductionOp
//

template<typename Ordinal, typename T>
CharToReferenceTypeReductionOp<Ordinal,T>::CharToReferenceTypeReductionOp(
  const RCP<const Serializer<Ordinal,T> >                 &serializer
  ,const RCP<const ReferenceTypeReductionOp<Ordinal,T> >  &reductOp
  )
  :serializer_(serializer), reductOp_(reductOp)
{}

template<typename Ordinal, typename T>
void CharToReferenceTypeReductionOp<Ordinal,T>::reduce(
  const Ordinal     charCount
  ,const char       charInBuffer[]
  ,char             charInoutBuffer[]
  ) const
{
  ConstReferenceTypeDeserializationBuffer<Ordinal,T>
    inBuffer(*serializer_,charCount,charInBuffer);
  ReferenceTypeDeserializationBuffer<Ordinal,T>
    inoutBuffer(*serializer_,charCount,charInoutBuffer);
  reductOp_->reduce(
    inBuffer.getCount(),inBuffer.getBuffer(),inoutBuffer.getBuffer()
    );
}

} // namespace Teuchos

#endif // TEUCHOS_REDUCTION_OP_HELPERS_HPP
