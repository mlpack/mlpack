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

#ifndef TEUCHOS_COMM_HPP
#define TEUCHOS_COMM_HPP

#include "Teuchos_ReductionOp.hpp"

namespace Teuchos {

/** \brief Abstract interface class for a basic communication channel between
 * one or more processes.
 *
 * This interface is templated on the ordinal type but only deals with buffers
 * of untyped data represented as arrays <tt>char</tt> type. All reduction
 * operations that are initiated by the concreate communicator object are
 * performed by user-defined <tt>ReductOpBase</tt> objects.  It is the
 * responsibility of the <tt>ReductOpBase</tt> object to know what the currect
 * data type is, to perform casts or serializations/unserializations to and
 * from <tt>char[]</tt> buffers, and to know how to reduce the objects
 * correctly.  It is strictly up to the client to correctly convert data types
 * to <tt>char[]</tt> arrays but there is a great deal of helper code to make
 * this easy and safe.
 *
 * ToDo: Finish documentation!
 */
template<typename Ordinal>
class Comm : virtual public Describable {
public:
  
  //! @name Query functions 
  //@{

  /** \brief Returns the rank of this process.
   *
   * <b>Postconditions:</b><ul>
   * <li><tt>0 <= return && return < this->getSize()</tt>
   * </ul>
   */
  virtual int getRank() const = 0;

  /** \brief Returns the number of processes that make up this communicator.
   *
   * <b>Postconditions:</b><ul>
   * <li><tt>return > 0</tt>
   * </ul>
   */
  virtual int getSize() const = 0;
  
  //@}

  //! @name Collective Operations 
  //@{

  /** \brief Pause every process in <tt>*this</tt> communicator until all the
   * processes reach this point.
   */
  virtual void barrier() const = 0;
  
  /** \brief Broadcast values from the root process to the slave processes.
   *
   * \param  rootRank
   *           [in] The rank of the root process.
   * \param  count
   *           [in] The number of bytes in <tt>buffer[]</tt>.
   * \param  buffer
   *           [in/out] Array (length <tt>bytes</tt>) of packed data.  Must be set on input
   *           on the root processes with rank <tt>root</tt>.  On output, each processs,
   *           including the root process contains the data.
   *
   * <b>Preconditions:</b><ul>
   * <li><tt>0 <= rootRank && rootRank < this->getSize()</tt>
   * </ul>
   */
  virtual void broadcast(
    const int rootRank, const Ordinal bytes, char buffer[]
    ) const = 0;

  /** \brief Gather values from each process to collect on all processes.
   *
   * \param  sendBytes
   *           [in] Number of entires in <tt>sendBuffer[]</tt> on input.
   * \param  sendBuffer
   *           [in] Array (length <tt>sendBytes</tt>) of data being sent from each process.
   * \param  recvBytes
   *           [in] Number of entires in <tt>recvBuffer[]</tt> which must be
   *           equal to <tt>sendBytes*this->getSize()</tt>.  This field is just here
   *           for debug checking.
   * \param  recvBuffer
   *           [out] Array (length <tt>recvBytes</tt>) of all of the entires
   *           sent from each processes.  Specifically, <tt>recvBuffer[sendBytes*j+i]</tt>,
   *           for <tt>j=0...this->getSize()-1</tt> and <tt>i=0...sendBytes-1</tt>,
   *           is the entry <tt>sendBuffer[i]</tt> from process with rank <tt>j</tt>.
   *
   * <b>Preconditions:</b><ul>
   * <li><tt>recvBytes==sendBytes*this->getSize()</tt>
   * </ul>
   */
  virtual void gatherAll(
    const Ordinal sendBytes, const char sendBuffer[]
    ,const Ordinal recvBytes, char recvBuffer[]
    ) const = 0;

  /** \brief Global reduction.
   *
   * \param  reductOp
   *           [in] The user-defined reduction operation
   * \param  bytes
   *           [in] The length of the buffers <tt>sendBuffer[]</tt> and <tt>globalReducts[]</tt>.
   * \param  sendBuffer
   *           [in] Array (length <tt>bytes</tt>) of the data contributed from each process.
   * \param  globalReducts
   *           [out] Array (length <tt>bytes</tt>) of the global reduction from each process.
   */
  virtual void reduceAll(
    const ValueTypeReductionOp<Ordinal,char> &reductOp
    ,const Ordinal bytes, const char sendBuffer[], char globalReducts[]
    ) const = 0;

  /** \brief Global reduction combined with a scatter.
   *
   * \param  reductOp
   *           [in] The user-defined reduction operation.
   * \param  sendBytes
   *           [in] The number of entires in <tt>sendBuffer[]</tt>.  This must be the same
   *           in each process.
   * \param  sendBuffer
   *           [in] Array (length <tt>sendBytes</tt>) of the data contributed from each process.
   * \param  recvCounts
   *           [in] Array (length <tt>this->getSize()</tt>) which gives the number of element
   *           blocks of block size <tt>blockSize</tt> from the global reduction that will be
   *           recieved in each process.
   * \param  blockSize
   *           [in] Gives the block size for interpreting <tt>recvCount</tt>
   * \param  myGlobalReducts
   *           [out] Array (length <tt>blockSize*recvBytes[rank]</tt>) of the global reductions gathered
   *           in this process.
   *
   * <b>Preconditions:</b><ul>
   * <li><tt>sendBytes == blockSize*sum(recvCounts[i],i=0...this->getSize()-1)</tt>
   * </ul>
   */
  virtual void reduceAllAndScatter(
    const ValueTypeReductionOp<Ordinal,char> &reductOp
    ,const Ordinal sendBytes, const char sendBuffer[]
    ,const Ordinal recvCounts[], const Ordinal blockSize, char myGlobalReducts[]
    ) const = 0;

  /** \brief Scan reduction.
   *
   * \param  reductOp
   *           [in] The user-defined reduction operation
   * \param  bytes
   *           [in] The length of the buffers <tt>sendBuffer[]</tt> and <tt>scanReducts[]</tt>.
   * \param  sendBuffer
   *           [in] Array (length <tt>bytes</tt>) of the data contributed from each process.
   * \param  scanReducts
   *           [out] Array (length <tt>bytes</tt>) of the reduction up to and including
   *           this process.
   */
	virtual void scan(
    const ValueTypeReductionOp<Ordinal,char> &reductOp
    ,const Ordinal bytes, const char sendBuffer[], char scanReducts[]
    ) const = 0;

  //! @name Point-to-Point Operations 
  //@{

  /** \brief Blocking send of data from this process to another process.
   *
   * \param  bytes
   *           [in] The number of bytes of data being passed between processes.
   * \param  sendBuffer
   *           [in] Array (length <tt>bytes</tt>) of data being sent from this process.
   *           This buffer can be immediately destroyed or reused as soon as the function
   *           exits (that is why this function is "blocking").
   * \param  destRank
   *           [in] The rank of the process to recieve the data.
   *
   * <b>Preconditions:</b><ul>
   * <li><tt>0 <= destRank && destRank < this->getSize()</tt>
   * <li><tt>destRank != this->getRank()</tt>
   * </ul>
   */
  virtual void send(
    const Ordinal bytes, const char sendBuffer[], const int destRank
    ) const = 0;

  /** \brief Blocking receive of data from this process to another process.
   *
   * \param  sourceRank
   *           [in] The rank of the process to recieve the data from.  If <tt>sourceRank < 0</tt> then
   *           data will be recieved from any process.
   * \param  bytes
   *           [in] The number of bytes of data being passed between processes.
   * \param  recvBuffer
   *           [out] Array (length <tt>bytes</tt>) of data being received from this process.
   *           This buffer can be immediately used to access the data as soon as the function
   *           exits (that is why this function is "blocking").
   *
   * <b>Preconditions:</b><ul>
   * <li>[<tt>sourceRank >= 0] <tt>sourceRank < this->getSize()</tt>
   * <li><tt>sourceRank != this->getRank()</tt>
   * </ul>
   *
   * \return Returns the senders rank.
   */
  virtual int receive(
    const int sourceRank, const Ordinal bytes, char recvBuffer[]
    ) const = 0;
  
  //@}
	
}; // class Comm

} // namespace Teuchos

#endif // TEUCHOS_COMM_HPP
