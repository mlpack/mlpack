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

#ifndef TEUCHOS_SERIAL_COMM_HPP
#define TEUCHOS_SERIAL_COMM_HPP

#include "Teuchos_Comm.hpp"
#include "Teuchos_OrdinalTraits.hpp"

namespace Teuchos {

/** \brief Concrete serial communicator subclass.
 *
 * ToDo: Finish documentation!
 */
template<typename Ordinal>
class SerialComm : public Comm<Ordinal> {
public:

  //! @name Constructors 
  //@{

  /** \brief . */
  SerialComm();

  //@}

  //! @name Overridden from Comm 
  //@{

  /** \brief . */
  int getRank() const;
  /** \brief . */
  int getSize() const;
  /** \brief . */
  void barrier() const;
  /** \brief . */
  void broadcast(
    const int rootRank, const Ordinal bytes, char buffer[]
    ) const;
  /** \brief . */
  void gatherAll(
    const Ordinal sendBytes, const char sendBuffer[]
    ,const Ordinal recvBytes, char recvBuffer[]
    ) const;
  /** \brief . */
  void reduceAll(
    const ValueTypeReductionOp<Ordinal,char> &reductOp
    ,const Ordinal bytes, const char sendBuffer[], char globalReducts[]
    ) const;
  /** \brief . */
  void reduceAllAndScatter(
    const ValueTypeReductionOp<Ordinal,char> &reductOp
    ,const Ordinal sendBytes, const char sendBuffer[]
    ,const Ordinal recvCounts[], const Ordinal blockSize, char myGlobalReducts[]
    ) const;
  /** \brief . */
	void scan(
    const ValueTypeReductionOp<Ordinal,char> &reductOp
    ,const Ordinal bytes, const char sendBuffer[], char scanReducts[]
    ) const;
  /** \brief . */
  void send(
    const Ordinal bytes, const char sendBuffer[], const int destRank
    ) const;
  /** \brief . */
  int receive(
    const int sourceRank, const Ordinal bytes, char recvBuffer[]
    ) const;

  //@}

  //! @name Overridden from Describable 
  //@{

  /** \brief . */
  std::string description() const;

  //@}
	
};

// ////////////////////////
// Implementations

// Constructors

template<typename Ordinal>
SerialComm<Ordinal>::SerialComm()
{}

// Overridden from Comm
  
template<typename Ordinal>
int SerialComm<Ordinal>::getRank() const
{
  return 0;
}
  
template<typename Ordinal>
int SerialComm<Ordinal>::getSize() const
{
  return 1;
}
  
template<typename Ordinal>
void SerialComm<Ordinal>::barrier() const
{
  // Nothing to do
}
  
template<typename Ordinal>
void SerialComm<Ordinal>::broadcast(
  const int rootRank, const Ordinal bytes, char buffer[]
  ) const
{
  // Nothing to do
}
  
template<typename Ordinal>
void SerialComm<Ordinal>::gatherAll(
  const Ordinal sendBytes, const char sendBuffer[]
  ,const Ordinal recvBytes, char recvBuffer[]
  ) const
{
#ifdef TEUCHOS_DEBUG
  TEST_FOR_EXCEPT(!(sendBytes==recvBytes));
#endif
  std::copy(sendBuffer,sendBuffer+sendBytes,recvBuffer);
}
  
template<typename Ordinal>
void SerialComm<Ordinal>::reduceAll(
  const ValueTypeReductionOp<Ordinal,char> &reductOp
  ,const Ordinal bytes, const char sendBuffer[], char globalReducts[]
  ) const
{
  std::copy(sendBuffer,sendBuffer+bytes,globalReducts);
}
  
template<typename Ordinal>
void SerialComm<Ordinal>::reduceAllAndScatter(
  const ValueTypeReductionOp<Ordinal,char> &reductOp
  ,const Ordinal sendBytes, const char sendBuffer[]
  ,const Ordinal recvCounts[], const Ordinal blockSize, char myGlobalReducts[]
  ) const
{
#ifdef TEUCHOS_DEBUG
  TEST_FOR_EXCEPT( recvCounts==NULL || blockSize*recvCounts[0] != sendBytes ); 
#endif
  std::copy(sendBuffer,sendBuffer+sendBytes,myGlobalReducts);
}
  
template<typename Ordinal>
void SerialComm<Ordinal>::scan(
  const ValueTypeReductionOp<Ordinal,char> &reductOp
  ,const Ordinal bytes, const char sendBuffer[], char scanReducts[]
  ) const
{
  std::copy(sendBuffer,sendBuffer+bytes,scanReducts);
}
  
template<typename Ordinal>
void SerialComm<Ordinal>::send(
  const Ordinal bytes, const char sendBuffer[], const int destRank
  ) const
{
  TEST_FOR_EXCEPTION(
    true, std::logic_error
    ,"SerialComm<Ordinal>::send(...): Error, you can not call send(...) when you"
    " only have one process!"
    );
}
  
template<typename Ordinal>
int SerialComm<Ordinal>::receive(
  const int sourceRank, const Ordinal bytes, char recvBuffer[]
  ) const
{
  TEST_FOR_EXCEPTION(
    true, std::logic_error
    ,"SerialComm<Ordinal>::receive(...): Error, you can not call receive(...) when you"
    " only have one process!"
    );
  // The next line will never be reached, but a return is required on some platforms
  return 0; 
}

// Overridden from Describable

template<typename Ordinal>
std::string SerialComm<Ordinal>::description() const
{
  std::ostringstream oss;
  oss << "Teuchos::SerialComm<"<<OrdinalTraits<Ordinal>::name()<<">";
  return oss.str();
}

} // namespace Teuchos

#endif // TEUCHOS_SERIAL_COMM_HPP
