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

#ifndef TEUCHOS_COMM_HELPERS_HPP
#define TEUCHOS_COMM_HELPERS_HPP

#include "Teuchos_Comm.hpp"
#include "Teuchos_CommUtilities.hpp"
#include "Teuchos_SerializationTraitsHelpers.hpp"
#include "Teuchos_ReductionOpHelpers.hpp"
#include "Teuchos_SerializerHelpers.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_OrdinalTraits.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_TypeNameTraits.hpp"

namespace Teuchos {

//
// Teuchos::Comm Helper Functions
//

/** \brief Enumeration for selecting from a set of pre-defined reduction
 * operations.
 *
 * \relates Comm
 */
enum EReductionType {
  REDUCE_SUM     ///< Sum
  ,REDUCE_MIN    ///< Min
  ,REDUCE_MAX    ///< Max
  ,REDUCE_AND    ///< Logical AND
};

/** \brief Convert to std::string representation.
 *
 * \relates EReductionType
 */
inline
const char* toString( const EReductionType reductType )
{
  switch(reductType) {
    case REDUCE_SUM: return "REDUCE_SUM";
    case REDUCE_MIN: return "REDUCE_MIN";
    case REDUCE_MAX: return "REDUCE_MAX";
    case REDUCE_AND: return "REDUCE_AND";
   default: TEST_FOR_EXCEPT(true);
  }
  return 0; // Will never be called
}

/** \brief Get the process rank.
 *
 * \relates Comm
 */
template<typename Ordinal>
int rank(const Comm<Ordinal>& comm);

/** \brief Get the number of processes in the communicator.
 *
 * \relates Comm
 */
template<typename Ordinal>
int size(const Comm<Ordinal>& comm);

/** \brief Barrier.
 *
 * \relates Comm
 */
template<typename Ordinal>
void barrier(const Comm<Ordinal>& comm);

/** \brief Broadcast array of objects that use value semantics.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
void broadcast(
  const Comm<Ordinal>& comm
  ,const int rootRank, const Ordinal count, Packet buffer[]
  );

/** \brief Broadcast single object that use value semantics.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
void broadcast(
  const Comm<Ordinal>& comm
  ,const int rootRank, Packet *object
  );

/** \brief Broadcast array of objects that use reference semantics.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
void broadcast(
  const Comm<Ordinal>& comm, const Serializer<Ordinal,Packet> &serializer
  ,const int rootRank, const Ordinal count, Packet*const buffer[]
  );

/** \brief Gather array of objects that use value semantics from every process
 * to every process.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
void gatherAll(
  const Comm<Ordinal>& comm
  ,const Ordinal sendCount, const Packet sendBuffer[]
  ,const Ordinal recvCount, Packet recvBuffer[]
  );

/** \brief Gather array of objects that use reference semantics from every
 * process to every process.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
void gatherAll(
  const Comm<Ordinal>& comm, const Serializer<Ordinal,Packet> &serializer
  ,const Ordinal sendCount, const Packet*const sendBuffer[]
  ,const Ordinal recvCount, Packet*const recvBuffer[]
  );

/** \brief Collective reduce all of array of objects using value semantics
 * using a user-defined reduction operator.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
void reduceAll(
  const Comm<Ordinal>& comm, const ValueTypeReductionOp<Ordinal,Packet> &reductOp
  ,const Ordinal count, const Packet sendBuffer[], Packet globalReducts[]
  );

/** \brief Collective reduce all of array of objects using value semantics
 * using a pre-defined reduction type.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
void reduceAll(
  const Comm<Ordinal>& comm, const EReductionType reductType
  ,const Ordinal count, const Packet sendBuffer[], Packet globalReducts[]
  );

/** \brief Collective reduce all for single object using value semantics using
 * a pre-defined reduction type.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
void reduceAll(
  const Comm<Ordinal>& comm, const EReductionType reductType
  ,const Packet &send, Packet *globalReduct
  );

/** \brief Collective reduce all for array of objects using reference
 * semantics.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
void reduceAll(
  const Comm<Ordinal>& comm, const Serializer<Ordinal,Packet> &serializer
  ,const ReferenceTypeReductionOp<Ordinal,Packet> &reductOp
  ,const Ordinal count, const Packet*const sendBuffer[], Packet*const globalReducts[]
  );

/** \brief Reduce and Scatter array of objects that use value semantics using
 * a user-defined reduction object.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
void reduceAllAndScatter(
  const Comm<Ordinal>& comm, const ValueTypeReductionOp<Ordinal,Packet> &reductOp
  ,const Ordinal sendCount, const Packet sendBuffer[] 
  ,const Ordinal recvCounts[], Packet myGlobalReducts[]
  );

/** \brief Reduce and Scatter array of objects that use value semantics using
 * a a pre-defined reduction type.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
void reduceAllAndScatter(
  const Comm<Ordinal>& comm, const EReductionType reductType
  ,const Ordinal sendCount, const Packet sendBuffer[] 
  ,const Ordinal recvCounts[], Packet myGlobalReducts[]
  );

/** \brief Reduce and Scatter array of objects that use reference semantics
 * using a user-defined reduction object.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
void reduceAllAndScatter(
  const Comm<Ordinal>& comm, const Serializer<Ordinal,Packet> &serializer
  ,const ReferenceTypeReductionOp<Ordinal,Packet> &reductOp
  ,const Ordinal sendCount, const Packet*const sendBuffer[] 
  ,const Ordinal recvCounts[], Packet*const myGlobalReducts[]
  );

/** \brief Scan/Reduce array of objects that use value semantics using a
 * user-defined reduction operator.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
void scan(
  const Comm<Ordinal>& comm, const ValueTypeReductionOp<Ordinal,Packet> &reductOp
  ,const Ordinal count, const Packet sendBuffer[], Packet scanReducts[]
  );

/** \brief Scan/Reduce array of objects using value semantics using a
 * predefined reduction type.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
void scan(
  const Comm<Ordinal>& comm, const EReductionType reductType
  ,const Ordinal count, const Packet sendBuffer[], Packet scanReducts[]
  );

/** \brief Scan/Reduce single object using value semantics using a predefined
 * reduction type.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
void scan(
  const Comm<Ordinal>& comm, const EReductionType reductType
  ,const Packet &send, Packet *scanReduct
  );

/** \brief Scan/Reduce array of objects that use reference semantics using a
 * user-defined reduction operator.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
void scan(
  const Comm<Ordinal>& comm, const Serializer<Ordinal,Packet> &serializer
  ,const ReferenceTypeReductionOp<Ordinal,Packet> &reductOp
  ,const Ordinal count, const Packet*const sendBuffer[], Packet*const scanReducts[]
  );

/** \brief Send objects that use values semantics to another process.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
void send(
  const Comm<Ordinal>& comm
  ,const Ordinal count, const Packet sendBuffer[], const int destRank
  );

/** \brief Send a single object that use values semantics to another process.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
void send(
  const Comm<Ordinal>& comm
  ,const Packet &send, const int destRank
  );

/** \brief Send objects that use reference semantics to another process.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
void send(
  const Comm<Ordinal>& comm, const Serializer<Ordinal,Packet> &serializer
  ,const Ordinal count, const Packet*const sendBuffer[], const int destRank
  );

/** \brief Receive objects that use values semantics from another process.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
int receive(
  const Comm<Ordinal>& comm
  ,const int sourceRank, const Ordinal count, Packet recvBuffer[] 
  );

/** \brief Receive a single object that use values semantics from another process.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
int receive(
  const Comm<Ordinal>& comm
  ,const int sourceRank, Packet *recv 
  );

/** \brief Receive objects that use reference semantics from another process.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
int receive(
  const Comm<Ordinal>& comm, const Serializer<Ordinal,Packet> &serializer
  ,const int sourceRank, const Ordinal count, Packet*const recvBuffer[] 
  );

//
// Standard reduction subclasses for objects that use value semantics
//

/** \brief Standard summation operator for types with value semantics.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
class SumValueReductionOp : public ValueTypeReductionOp<Ordinal,Packet>
{
public:
  /** \brief . */
  void reduce(
    const Ordinal     count
    ,const Packet     inBuffer[]
    ,Packet           inoutBuffer[]
    ) const;
};

/** \brief Standard min operator for types with value semantics.
 *
 * Note, this class object will throw an std::exception when used with a packet
 * type where <tt>ScalarTraits<Packet>::isComparable==false</tt> but it will
 * still compile.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
class MinValueReductionOp : public ValueTypeReductionOp<Ordinal,Packet>
{
public:
  /** \brief . */
  void reduce(
    const Ordinal     count
    ,const Packet     inBuffer[]
    ,Packet           inoutBuffer[]
    ) const;
};

/** \brief Standard Max operator for types with value semantics.
 *
 * Note, this class object will throw an std::exception when used with a packet
 * type where <tt>ScalarTraits<Packet>::isComparable==false</tt> but it will
 * still compile.
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
class MaxValueReductionOp : public ValueTypeReductionOp<Ordinal,Packet>
{
public:
  /** \brief . */
  void reduce(
    const Ordinal     count
    ,const Packet     inBuffer[]
    ,Packet           inoutBuffer[]
    ) const;
};



/** \brief Standard logical AND operator for booleans
 *
 * \relates Comm
 */
template<typename Ordinal, typename Packet>
class ANDValueReductionOp : public ValueTypeReductionOp<Ordinal,Packet>
{
public:
  /** \brief . */
  void reduce(
    const Ordinal     count
    ,const Packet       inBuffer[]
    ,Packet             inoutBuffer[]
    ) const;
};





// ////////////////////////////////////////////////////////////
// Implementation details (not for geneal users to mess with)

//
// ReductionOp Utilities
//

namespace MixMaxUtilities {

template<bool isComparable, typename Ordinal, typename Packet>
class Min {};

template<typename Ordinal, typename Packet>
class Min<true,Ordinal,Packet> {
public:
  static void min(
    const Ordinal     count
    ,const Packet     inBuffer[]
    ,Packet           inoutBuffer[]
    )
    {
      for( int i = 0; i < count; ++i )
        inoutBuffer[i] = TEUCHOS_MIN(inoutBuffer[i],inBuffer[i]);
    }
};

template<typename Ordinal, typename Packet>
class Min<false,Ordinal,Packet> {
public:
  static void min(
    const Ordinal     count
    ,const Packet     inBuffer[]
    ,Packet           inoutBuffer[]
    )
    {
      TEST_FOR_EXCEPTION(
        true,std::logic_error
        ,"Error, the type "<<ScalarTraits<Packet>::name()
        <<" does not support comparison operations!"
        );
    }
};

template<bool isComparable, typename Ordinal, typename Packet>
class Max {};

template<typename Ordinal, typename Packet>
class Max<true,Ordinal,Packet> {
public:
  static void max(
    const Ordinal     count
    ,const Packet     inBuffer[]
    ,Packet           inoutBuffer[]
    )
    {
      for( int i = 0; i < count; ++i )
        inoutBuffer[i] = TEUCHOS_MAX(inoutBuffer[i],inBuffer[i]);
    }
};

template<typename Ordinal, typename Packet>
class Max<false,Ordinal,Packet> {
public:
  static void max(
    const Ordinal     count
    ,const Packet     inBuffer[]
    ,Packet           inoutBuffer[]
    )
    {
      TEST_FOR_EXCEPTION(
        true,std::logic_error
        ,"Error, the type "<<ScalarTraits<Packet>::name()
        <<" does not support comparison operations!"
        );
    }
};

template<bool isComparable, typename Ordinal, typename Packet>
class AND {};

template<typename Ordinal, typename Packet>
class AND<true,Ordinal,Packet> {
public:
  static void And(
                   const Ordinal     count
                   ,const Packet     inBuffer[]
                   ,Packet           inoutBuffer[]
                   )
  {
    for( int i = 0; i < count; ++i )
      inoutBuffer[i] = inoutBuffer[i] && inBuffer[i];
  }
};

template<typename Ordinal, typename Packet>
class AND<false,Ordinal,Packet> {
public:
  static void And(
    const Ordinal     count
    ,const Packet     inBuffer[]
    ,Packet           inoutBuffer[]
    )
    {
      TEST_FOR_EXCEPTION(
        true,std::logic_error
        ,"Error, the type "<<ScalarTraits<Packet>::name()
        <<" does not support logical AND operations!"
        );
    }
};

} // namespace MixMaxUtilities

template<typename Ordinal, typename Packet>
void SumValueReductionOp<Ordinal,Packet>::reduce(
  const Ordinal     count
  ,const Packet     inBuffer[]
  ,Packet           inoutBuffer[]
  ) const
{
  for( int i = 0; i < count; ++i )
    inoutBuffer[i] += inBuffer[i];
}

template<typename Ordinal, typename Packet>
void MinValueReductionOp<Ordinal,Packet>::reduce(
  const Ordinal     count
  ,const Packet     inBuffer[]
  ,Packet           inoutBuffer[]
  ) const
{
  typedef ScalarTraits<Packet> ST;
  MixMaxUtilities::Min<ST::isComparable,Ordinal,Packet>::min(
    count,inBuffer,inoutBuffer
    );
}

template<typename Ordinal, typename Packet>
void MaxValueReductionOp<Ordinal,Packet>::reduce(
  const Ordinal     count
  ,const Packet     inBuffer[]
  ,Packet           inoutBuffer[]
  ) const
{
  typedef ScalarTraits<Packet> ST;
  MixMaxUtilities::Max<ST::isComparable,Ordinal,Packet>::max(
    count,inBuffer,inoutBuffer
    );
}

template<typename Ordinal, typename Packet>
void ANDValueReductionOp<Ordinal,Packet>::reduce(
  const Ordinal     count
  ,const Packet     inBuffer[]
  ,Packet           inoutBuffer[]
  ) const
{
  typedef ScalarTraits<Packet> ST;
  MixMaxUtilities::AND<ST::isComparable,Ordinal,Packet>::And(
    count,inBuffer,inoutBuffer
    );
}

} // namespace Teuchos

// //////////////////////////
// Template implemenations

//
// ReductionOp utilities
//

namespace Teuchos {


// Not for the general user to use!  I am returning a raw ReducionOp* pointer
// to avoid the overhead of using RCP.  However, given the use case
// this is just fine since I can just use std::auto_ptr to make sure things
// are deleted correctly.
template<typename Ordinal, typename Packet>
ValueTypeReductionOp<Ordinal,Packet>* createOp( const EReductionType reductType )
{
  typedef ScalarTraits<Packet> ST;
  switch(reductType) {
    case REDUCE_SUM: {
      return new SumValueReductionOp<Ordinal,Packet>();
      break;
    }
    case REDUCE_MIN: {
      TEST_FOR_EXCEPT(!ST::isComparable);
      return new MinValueReductionOp<Ordinal,Packet>();
      break;
    }
    case REDUCE_MAX: {
      TEST_FOR_EXCEPT(!ST::isComparable);
      return new MaxValueReductionOp<Ordinal,Packet>();
      break;
    }
    case REDUCE_AND: {
      return new ANDValueReductionOp<Ordinal, Packet>();
      break;
    }
    default:
      TEST_FOR_EXCEPT(true);
  }
  return 0; // Will never be called!
}

} // namespace Teuchos

//
// Teuchos::Comm wrapper functions
//

template<typename Ordinal>
int Teuchos::rank(const Comm<Ordinal>& comm)
{
  return comm.getRank();
}

template<typename Ordinal>
int Teuchos::size(const Comm<Ordinal>& comm)
{
  return comm.getSize();
}

template<typename Ordinal>
void Teuchos::barrier(const Comm<Ordinal>& comm)
{
  TEUCHOS_COMM_TIME_MONITOR(
    "Teuchos::CommHelpers: barrier<"
    <<OrdinalTraits<Ordinal>::name()
    <<">()"
    );
  comm.barrier();
}

template<typename Ordinal, typename Packet>
void Teuchos::broadcast(
  const Comm<Ordinal>& comm
  ,const int rootRank, const Ordinal count, Packet buffer[]
  )
{
  TEUCHOS_COMM_TIME_MONITOR(
    "Teuchos::CommHelpers: broadcast<"
    <<OrdinalTraits<Ordinal>::name()<<","<<ScalarTraits<Packet>::name()
    <<">( value type )"
    );
  ValueTypeSerializationBuffer<Ordinal,Packet>
    charBuffer(count,buffer);
  comm.broadcast(
    rootRank,charBuffer.getBytes(),charBuffer.getCharBuffer()
    );
}

template<typename Ordinal, typename Packet>
void Teuchos::broadcast(
  const Comm<Ordinal>& comm
  ,const int rootRank, Packet *object
  )
{
  broadcast(comm,rootRank,1,object);
}

template<typename Ordinal, typename Packet>
void Teuchos::broadcast(
  const Comm<Ordinal>& comm, const Serializer<Ordinal,Packet> &serializer
  ,const int rootRank, const Ordinal count, Packet*const buffer[]
  )
{
  TEUCHOS_COMM_TIME_MONITOR(
    "Teuchos::CommHelpers: broadcast<"
    <<OrdinalTraits<Ordinal>::name()<<","<<TypeNameTraits<Packet>::name()
    <<">( reference type )"
    );
  ReferenceTypeSerializationBuffer<Ordinal,Packet>
    charBuffer(serializer,count,buffer);
  comm.broadcast(
    rootRank,charBuffer.getBytes(),charBuffer.getCharBuffer()
    );
}

template<typename Ordinal, typename Packet>
void Teuchos::gatherAll(
  const Comm<Ordinal>& comm
  ,const Ordinal sendCount, const Packet sendBuffer[]
  ,const Ordinal recvCount, Packet recvBuffer[]
  )
{
  TEUCHOS_COMM_TIME_MONITOR(
    "Teuchos::CommHelpers: gatherAll<"
    <<OrdinalTraits<Ordinal>::name()<<","<<ScalarTraits<Packet>::name()
    <<">( value type )"
    );
  ConstValueTypeSerializationBuffer<Ordinal,Packet>
    charSendBuffer(sendCount,sendBuffer);
  ValueTypeSerializationBuffer<Ordinal,Packet>
    charRecvBuffer(recvCount,recvBuffer);
  comm.gatherAll(
    charSendBuffer.getBytes(),charSendBuffer.getCharBuffer()
    ,charRecvBuffer.getBytes(),charRecvBuffer.getCharBuffer()
    );
}

template<typename Ordinal, typename Packet>
void Teuchos::gatherAll(
  const Comm<Ordinal>& comm, const Serializer<Ordinal,Packet> &serializer
  ,const Ordinal sendCount, const Packet*const sendBuffer[]
  ,const Ordinal recvCount, Packet*const recvBuffer[]
  )
{
  TEST_FOR_EXCEPT(true); // ToDo: Implement and test when needed!
}

template<typename Ordinal, typename Packet>
void Teuchos::reduceAll(
  const Comm<Ordinal>& comm, const ValueTypeReductionOp<Ordinal,Packet> &reductOp
  ,const Ordinal count, const Packet sendBuffer[], Packet globalReducts[]
  )
{
  TEUCHOS_COMM_TIME_MONITOR(
    "Teuchos::CommHelpers: reduceAll<"
    <<OrdinalTraits<Ordinal>::name()<<","<<ScalarTraits<Packet>::name()
    <<">( value type, user-defined op )"
    );
  ConstValueTypeSerializationBuffer<Ordinal,Packet>
    charSendBuffer(count,sendBuffer);
  ValueTypeSerializationBuffer<Ordinal,Packet>
    charGlobalReducts(count,globalReducts);
  CharToValueTypeReductionOp<Ordinal,Packet>
    _reductOp(rcp(&reductOp,false));
  comm.reduceAll(
    _reductOp,charSendBuffer.getBytes(),charSendBuffer.getCharBuffer()
    ,charGlobalReducts.getCharBuffer()
    );
}

template<typename Ordinal, typename Packet>
void Teuchos::reduceAll(
  const Comm<Ordinal>& comm, const EReductionType reductType
  ,const Ordinal count, const Packet sendBuffer[], Packet globalReducts[]
  )
{
  TEUCHOS_COMM_TIME_MONITOR(
    "Teuchos::CommHelpers: reduceAll<"
    <<OrdinalTraits<Ordinal>::name()<<","<<ScalarTraits<Packet>::name()
    <<">( value type, "<<toString(reductType)<<" )"
    );
  std::auto_ptr<ValueTypeReductionOp<Ordinal,Packet> >
    reductOp(createOp<Ordinal,Packet>(reductType));
  reduceAll(comm,*reductOp,count,sendBuffer,globalReducts);
}

template<typename Ordinal, typename Packet>
void Teuchos::reduceAll(
  const Comm<Ordinal>& comm, const EReductionType reductType
  ,const Packet &send, Packet *globalReduct
  )
{
  reduceAll(comm,reductType,1,&send,globalReduct);
}

template<typename Ordinal, typename Packet>
void Teuchos::reduceAll(
  const Comm<Ordinal>& comm, const Serializer<Ordinal,Packet> &serializer
  ,const ReferenceTypeReductionOp<Ordinal,Packet> &reductOp
  ,const Ordinal count, const Packet*const sendBuffer[], Packet*const globalReducts[]
  )
{
  TEUCHOS_COMM_TIME_MONITOR(
    "Teuchos::CommHelpers: reduceAll<"
    <<OrdinalTraits<Ordinal>::name()<<","<<TypeNameTraits<Packet>::name()
    <<">( reference type )"
    );
  ConstReferenceTypeSerializationBuffer<Ordinal,Packet>
    charSendBuffer(serializer,count,sendBuffer);
  ReferenceTypeSerializationBuffer<Ordinal,Packet>
    charGlobalReducts(serializer,count,globalReducts);
  CharToReferenceTypeReductionOp<Ordinal,Packet>
    _reductOp(rcp(&serializer,false),rcp(&reductOp,false));
  comm.reduceAll(
    _reductOp,charSendBuffer.getBytes(),charSendBuffer.getCharBuffer()
    ,charGlobalReducts.getCharBuffer()
    );
}

template<typename Ordinal, typename Packet>
void Teuchos::reduceAllAndScatter(
  const Comm<Ordinal>& comm, const ValueTypeReductionOp<Ordinal,Packet> &reductOp
  ,const Ordinal sendCount, const Packet sendBuffer[] 
  ,const Ordinal recvCounts[], Packet myGlobalReducts[]
  )
{
  TEUCHOS_COMM_TIME_MONITOR(
    "Teuchos::CommHelpers: reduceAllAndScatter<"
    <<OrdinalTraits<Ordinal>::name()<<","<<ScalarTraits<Packet>::name()
    <<">( value type, user-defined op )"
    );
#ifdef TEUCHOS_DEBUG
  Ordinal sumRecvCounts = 0;
  const int size = Teuchos::size(comm);
  for( Ordinal i = 0; i < size; ++i )
    sumRecvCounts += recvCounts[i];
  TEST_FOR_EXCEPT(!(sumRecvCounts==sendCount));
#endif
  const int rank = Teuchos::rank(comm);
  ConstValueTypeSerializationBuffer<Ordinal,Packet>
    charSendBuffer(sendCount,sendBuffer);
  ValueTypeSerializationBuffer<Ordinal,Packet>
    charMyGlobalReducts(recvCounts[rank],myGlobalReducts);
  CharToValueTypeReductionOp<Ordinal,Packet>
    _reductOp(rcp(&reductOp,false));
  const Ordinal
    blockSize = charSendBuffer.getBytes()/sendCount;
  comm.reduceAllAndScatter(
    _reductOp,charSendBuffer.getBytes(),charSendBuffer.getCharBuffer()
    ,recvCounts,blockSize,charMyGlobalReducts.getCharBuffer()
    );
}

template<typename Ordinal, typename Packet>
void Teuchos::reduceAllAndScatter(
  const Comm<Ordinal>& comm, const EReductionType reductType
  ,const Ordinal sendCount, const Packet sendBuffer[] 
  ,const Ordinal recvCounts[], Packet myGlobalReducts[]
  )
{
  TEUCHOS_COMM_TIME_MONITOR(
    "Teuchos::CommHelpers: reduceAllAndScatter<"
    <<OrdinalTraits<Ordinal>::name()<<","<<ScalarTraits<Packet>::name()
    <<">( value type, "<<toString(reductType)<<" )"
    );
  std::auto_ptr<ValueTypeReductionOp<Ordinal,Packet> >
    reductOp(createOp<Ordinal,Packet>(reductType));
  reduceAllAndScatter(
    comm,*reductOp,sendCount,sendBuffer,recvCounts,myGlobalReducts
    );
}

template<typename Ordinal, typename Packet>
void Teuchos::reduceAllAndScatter(
  const Comm<Ordinal>& comm, const Serializer<Ordinal,Packet> &serializer
  ,const ReferenceTypeReductionOp<Ordinal,Packet> &reductOp
  ,const Ordinal sendCount, const Packet*const sendBuffer[] 
  ,const Ordinal recvCounts[], Packet*const myGlobalReducts[]
  )
{
  TEST_FOR_EXCEPT(true); // ToDo: Implement and test when needed!
}

template<typename Ordinal, typename Packet>
void Teuchos::scan(
  const Comm<Ordinal>& comm, const ValueTypeReductionOp<Ordinal,Packet> &reductOp
  ,const Ordinal count, const Packet sendBuffer[], Packet scanReducts[]
  )
{
  TEUCHOS_COMM_TIME_MONITOR(
    "Teuchos::CommHelpers: scan<"
    <<OrdinalTraits<Ordinal>::name()<<","<<ScalarTraits<Packet>::name()
    <<">( value type, user-defined op )"
    );
  ConstValueTypeSerializationBuffer<Ordinal,Packet>
    charSendBuffer(count,sendBuffer);
  ValueTypeSerializationBuffer<Ordinal,Packet>
    charScanReducts(count,scanReducts);
  CharToValueTypeReductionOp<Ordinal,Packet>
    _reductOp(rcp(&reductOp,false));
  comm.scan(
    _reductOp,charSendBuffer.getBytes(),charSendBuffer.getCharBuffer()
    ,charScanReducts.getCharBuffer()
    );
}

template<typename Ordinal, typename Packet>
void Teuchos::scan(
  const Comm<Ordinal>& comm, const EReductionType reductType
  ,const Ordinal count, const Packet sendBuffer[], Packet scanReducts[]
  )
{
  TEUCHOS_COMM_TIME_MONITOR(
    "Teuchos::CommHelpers: scan<"
    <<OrdinalTraits<Ordinal>::name()<<","<<ScalarTraits<Packet>::name()
    <<">( value type, "<<toString(reductType)<<" )"
    );
  std::auto_ptr<ValueTypeReductionOp<Ordinal,Packet> >
    reductOp(createOp<Ordinal,Packet>(reductType));
  scan(comm,*reductOp,count,sendBuffer,scanReducts);
}

template<typename Ordinal, typename Packet>
void Teuchos::scan(
  const Comm<Ordinal>& comm, const EReductionType reductType
  ,const Packet &send, Packet *globalReduct
  )
{
  scan(comm,reductType,1,&send,globalReduct);
}

template<typename Ordinal, typename Packet>
void Teuchos::scan(
  const Comm<Ordinal>& comm, const Serializer<Ordinal,Packet> &serializer
  ,const ReferenceTypeReductionOp<Ordinal,Packet> &reductOp
  ,const Ordinal count, const Packet*const sendBuffer[], Packet*const scanReducts[]
  )
{
  TEST_FOR_EXCEPT(true); // ToDo: Implement and test when needed!
}

template<typename Ordinal, typename Packet>
void Teuchos::send(
  const Comm<Ordinal>& comm
  ,const Ordinal count, const Packet sendBuffer[], const int destRank
  )
{
  TEUCHOS_COMM_TIME_MONITOR(
    "Teuchos::CommHelpers: send<"
    <<OrdinalTraits<Ordinal>::name()<<","<<ScalarTraits<Packet>::name()
    <<">( value type )"
    );
  ConstValueTypeSerializationBuffer<Ordinal,Packet>
    charSendBuffer(count,sendBuffer);
  comm.send(
    charSendBuffer.getBytes(),charSendBuffer.getCharBuffer()
    ,destRank
    );
}

template<typename Ordinal, typename Packet>
void Teuchos::send(
  const Comm<Ordinal>& comm
  ,const Packet &send, const int destRank
  )
{
  Teuchos::send<Ordinal,Packet>(comm,1,&send,destRank);
}

template<typename Ordinal, typename Packet>
void Teuchos::send(
  const Comm<Ordinal>& comm, const Serializer<Ordinal,Packet> &serializer
  ,const Ordinal count, const Packet*const sendBuffer[], const int destRank
  )
{
  TEST_FOR_EXCEPT(true); // ToDo: Implement and test when needed!
}

template<typename Ordinal, typename Packet>
int Teuchos::receive(
  const Comm<Ordinal>& comm
  ,const int sourceRank, const Ordinal count, Packet recvBuffer[] 
  )
{
  TEUCHOS_COMM_TIME_MONITOR(
    "Teuchos::CommHelpers: receive<"
    <<OrdinalTraits<Ordinal>::name()<<","<<ScalarTraits<Packet>::name()
    <<">( value type )"
    );
  ValueTypeSerializationBuffer<Ordinal,Packet>
    charRecvBuffer(count,recvBuffer);
  return comm.receive(
    sourceRank
    ,charRecvBuffer.getBytes(),charRecvBuffer.getCharBuffer()
    );
}

template<typename Ordinal, typename Packet>
int Teuchos::receive(
  const Comm<Ordinal>& comm
  ,const int sourceRank, Packet *recv 
  )
{
  return Teuchos::receive<Ordinal,Packet>(comm,sourceRank,1,recv);
}

template<typename Ordinal, typename Packet>
int Teuchos::receive(
  const Comm<Ordinal>& comm, const Serializer<Ordinal,Packet> &serializer
  ,const int sourceRank, const Ordinal count, Packet*const recvBuffer[] 
  )
{
  TEST_FOR_EXCEPT(true); // ToDo: Implement and test when needed!
}

#endif // TEUCHOS_COMM_HELPERS_HPP
