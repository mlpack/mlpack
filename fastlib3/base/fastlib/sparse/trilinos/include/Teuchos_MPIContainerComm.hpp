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

#ifndef TEUCHOS_MPICONTAINERCOMM_H
#define TEUCHOS_MPICONTAINERCOMM_H

/*! \file Teuchos_MPIContainerComm.hpp
  \brief Object representation of an MPI communicator for templated containers
*/

#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_MPIComm.hpp"
#include "Teuchos_MPITraits.hpp"

namespace Teuchos
{
  /** \ingroup MPI
   * \brief Object representation of an MPI communicator for templated containers
   * \note Template specialization exists for <tt>std::string</tt>.
   * @author Kevin Long
   */

  template <class T> class MPIContainerComm
  {
  public:

    //! Broadcast a single object 
    static void bcast(T& x, int src, const MPIComm& comm);

    //! Broadcast an array of objects
    static void bcast(Array<T>& x, int src, const MPIComm& comm);

    //! Broadcast an array of arrays 
    static void bcast(Array<Array<T> >& x,
                      int src, const MPIComm& comm);

    //! Gather to all processors
    static void allGather(const T& outgoing,
                          Array<T>& incoming,
                          const MPIComm& comm);

    //! All-to-all scatter/gather for an array of objects
    static void allToAll(const Array<T>& outgoing,
                         Array<Array<T> >& incoming,
                         const MPIComm& comm);

    //! All-to-all scatter/gather for an array of arrays
    static void allToAll(const Array<Array<T> >& outgoing,
                         Array<Array<T> >& incoming,
                         const MPIComm& comm);

    /** Gatherv: gather arrays of data to the root processor */
    static void gatherv(const Array<T>& outgoing,
                        Array<Array<T> >& incoming,
                        int rootRank,
                        const MPIComm& comm);

    //! Sum local values from all processors with rank < myRank
    static void accumulate(const T& localValue, Array<T>& sums, T& total,
                           const MPIComm& comm);

  private:
    //! Build a 1D array and an offset list from a 2D array
    static void getBigArray(const Array<Array<T> >& x,
                            Array<T>& bigArray,
                            Array<int>& offsets);

    //! Reassemble a 2D array from a 1D array and an offset table
    static void getSmallArrays(const Array<T>& bigArray,
                               const Array<int>& offsets,
                               Array<Array<T> >& x);


  };


#ifndef DOXYGEN_SHOULD_SKIP_THIS
  /** \ingroup MPI
   * Specialiaztion of MPIContainerComm<T> to std::string
   */
  template <> class MPIContainerComm<std::string>
  {
  public:
    static void bcast(std::string& x, int src, const MPIComm& comm);

    /** bcast an array of objects */
    static void bcast(Array<std::string>& x, int src, const MPIComm& comm);

    /** bcast an array of arrays  */
    static void bcast(Array<Array<std::string> >& x,
                      int src, const MPIComm& comm);

    /** AllGather: each process sends a single object to all other procs */
    static void allGather(const std::string& outgoing,
                          Array<std::string>& incoming,
                          const MPIComm& comm);

    /** Gatherv: gather arrays of strings to the root processor */
    static void gatherv(const Array<std::string>& outgoing,
                        Array<Array<std::string> >& incoming,
                        int rootRank,
                        const MPIComm& comm);

    /** get a single big array of characters from an array of strings,
     * packing the structural description into the header of the packed
     * array as follows:
     * \code
     * [numStrings, offset0, offset1, ..., offsetN, char data]
     * \endcode 
     */
    static void pack(const Array<std::string>& x,
                     Array<char>& packed);

    /** recover an array of strings from a single big array and
     * and offset table */
    static void unpack(const Array<char>& packed,
                       Array<std::string>& x);
  private:
    /** get a single big array of characters from an array of strings */
    static void getBigArray(const Array<std::string>& x,
                            Array<char>& bigArray,
                            Array<int>& offsets);

    /** recover an array of strings from a single big array and
     * and offset table */
    static void getStrings(const Array<char>& bigArray,
                           const Array<int>& offsets,
                           Array<std::string>& x);
  };

#endif // DOXYGEN_SHOULD_SKIP_THIS

  /* --------- generic functions for primitives ------------------- */

  template <class T> inline void MPIContainerComm<T>::bcast(T& x, int src,
                                                            const MPIComm& comm)
  {
    comm.bcast((void*)&x, 1, MPITraits<T>::type(), src);
  }


  /* ----------- generic functions for arrays of primitives ----------- */

  template <class T>
  inline void MPIContainerComm<T>::bcast(Array<T>& x, int src, const MPIComm& comm)
  {
    int len = x.length();
    MPIContainerComm<int>::bcast(len, src, comm);

    if (comm.getRank() != src)
      {
        x.resize(len);
      }
    if (len==0) return;

    /* then broadcast the contents */
    comm.bcast((void*) &(x[0]), (int) len,
               MPITraits<T>::type(), src);
  }



  /* ---------- generic function for arrays of arrays ----------- */

  template <class T>
  inline void MPIContainerComm<T>::bcast(Array<Array<T> >& x, int src, const MPIComm& comm)
  {
    Array<T> bigArray;
    Array<int> offsets;

    if (src==comm.getRank())
      {
        getBigArray(x, bigArray, offsets);
      }

    bcast(bigArray, src, comm);
    MPIContainerComm<int>::bcast(offsets, src, comm);

    if (src != comm.getRank())
      {
        getSmallArrays(bigArray, offsets, x);
      }
  }

  /* ---------- generic gather and scatter ------------------------ */

  template <class T> inline
  void MPIContainerComm<T>::allToAll(const Array<T>& outgoing,
                                     Array<Array<T> >& incoming,
                                     const MPIComm& comm)
  {
    int numProcs = comm.getNProc();

    // catch degenerate case
    if (numProcs==1)
      {
        incoming.resize(1);
        incoming[0] = outgoing;
        return;
      }

    Array<T> sb(numProcs * outgoing.length());
    Array<T> rb(numProcs * outgoing.length());

    T* sendBuf = new T[numProcs * outgoing.length()];
    TEST_FOR_EXCEPTION(sendBuf==0, 
                       std::runtime_error, "Comm::allToAll failed to allocate sendBuf");

    T* recvBuf = new T[numProcs * outgoing.length()];
    TEST_FOR_EXCEPTION(recvBuf==0, 
                       std::runtime_error, "Comm::allToAll failed to allocate recvBuf");

    int i;
    for (i=0; i<numProcs; i++)
      {
        for (int j=0; j<outgoing.length(); j++)
          {
            sendBuf[i*outgoing.length() + j] = outgoing[j];
          }
      }



    comm.allToAll(sendBuf, outgoing.length(), MPITraits<T>::type(),
                  recvBuf, outgoing.length(), MPITraits<T>::type());

    incoming.resize(numProcs);

    for (i=0; i<numProcs; i++)
      {
        incoming[i].resize(outgoing.length());
        for (int j=0; j<outgoing.length(); j++)
          {
            incoming[i][j] = recvBuf[i*outgoing.length() + j];
          }
      }

    delete [] sendBuf;
    delete [] recvBuf;
  }

  template <class T> inline
  void MPIContainerComm<T>::allToAll(const Array<Array<T> >& outgoing,
                                     Array<Array<T> >& incoming, const MPIComm& comm)
  {
    int numProcs = comm.getNProc();

    // catch degenerate case
    if (numProcs==1)
      {
        incoming = outgoing;
        return;
      }

    int* sendMesgLength = new int[numProcs];
    TEST_FOR_EXCEPTION(sendMesgLength==0, 
                       std::runtime_error, "failed to allocate sendMesgLength");
    int* recvMesgLength = new int[numProcs];
    TEST_FOR_EXCEPTION(recvMesgLength==0, 
                       std::runtime_error, "failed to allocate recvMesgLength");

    int p = 0;
    for (p=0; p<numProcs; p++)
      {
        sendMesgLength[p] = outgoing[p].length();
      }
    
    comm.allToAll(sendMesgLength, 1, MPIComm::INT,
                  recvMesgLength, 1, MPIComm::INT);


    int totalSendLength = 0;
    int totalRecvLength = 0;
    for (p=0; p<numProcs; p++)
      {
        totalSendLength += sendMesgLength[p];
        totalRecvLength += recvMesgLength[p];
      }

    T* sendBuf = new T[totalSendLength];
    TEST_FOR_EXCEPTION(sendBuf==0, 
                       std::runtime_error, "failed to allocate sendBuf");
    T* recvBuf = new T[totalRecvLength];
    TEST_FOR_EXCEPTION(recvBuf==0, 
                       std::runtime_error, "failed to allocate recvBuf");

    int* sendDisp = new int[numProcs];
    TEST_FOR_EXCEPTION(sendDisp==0, 
                       std::runtime_error, "failed to allocate sendDisp");
    int* recvDisp = new int[numProcs];
    TEST_FOR_EXCEPTION(recvDisp==0, 
                       std::runtime_error, "failed to allocate recvDisp");

    int count = 0;
    sendDisp[0] = 0;
    recvDisp[0] = 0;

    for (p=0; p<numProcs; p++)
      {
        for (int i=0; i<outgoing[p].length(); i++)
          {
            sendBuf[count] = outgoing[p][i];
            count++;
          }
        if (p>0)
          {
            sendDisp[p] = sendDisp[p-1] + sendMesgLength[p-1];
            recvDisp[p] = recvDisp[p-1] + recvMesgLength[p-1];
          }
      }

    comm.allToAllv(sendBuf, sendMesgLength,
                   sendDisp, MPITraits<T>::type(),
                   recvBuf, recvMesgLength,
                   recvDisp, MPITraits<T>::type());

    incoming.resize(numProcs);
    for (p=0; p<numProcs; p++)
      {
        incoming[p].resize(recvMesgLength[p]);
        for (int i=0; i<recvMesgLength[p]; i++)
          {
            incoming[p][i] = recvBuf[recvDisp[p] + i];
          }
      }

    delete [] sendBuf;
    delete [] sendMesgLength;
    delete [] sendDisp;
    delete [] recvBuf;
    delete [] recvMesgLength;
    delete [] recvDisp;
  }

  template <class T> inline
  void MPIContainerComm<T>::allGather(const T& outgoing, Array<T>& incoming,
                                      const MPIComm& comm)
  {
    int nProc = comm.getNProc();
    incoming.resize(nProc);

    if (nProc==1)
      {
        incoming[0] = outgoing;
      }
    else
      {
        comm.allGather((void*) &outgoing, 1, MPITraits<T>::type(),
                       (void*) &(incoming[0]), 1, MPITraits<T>::type());
      }
  }

  template <class T> inline
  void MPIContainerComm<T>::accumulate(const T& localValue, Array<T>& sums,
                                       T& total,
                                       const MPIComm& comm)
  {
    Array<T> contributions;
    allGather(localValue, contributions, comm);
    sums.resize(comm.getNProc());
    sums[0] = 0;
    total = contributions[0];

    for (int i=0; i<comm.getNProc()-1; i++)
      {
        total += contributions[i+1];
        sums[i+1] = sums[i] + contributions[i];
      }
  }




  template <class T> inline
  void MPIContainerComm<T>::getBigArray(const Array<Array<T> >& x, Array<T>& bigArray,
                                        Array<int>& offsets)
  {
    offsets.resize(x.length()+1);
    int totalLength = 0;

    for (int i=0; i<x.length(); i++)
      {
        offsets[i] = totalLength;
        totalLength += x[i].length();
      }
    offsets[x.length()] = totalLength;

    bigArray.resize(totalLength);

    for (int i=0; i<x.length(); i++)
      {
        for (int j=0; j<x[i].length(); j++)
          {
            bigArray[offsets[i]+j] = x[i][j];
          }
      }
  }

  template <class T> inline
  void MPIContainerComm<T>::getSmallArrays(const Array<T>& bigArray,
                                           const Array<int>& offsets,
                                           Array<Array<T> >& x)
  {
    x.resize(offsets.length()-1);
    for (int i=0; i<x.length(); i++)
      {
        x[i].resize(offsets[i+1]-offsets[i]);
        for (int j=0; j<x[i].length(); j++)
          {
            x[i][j] = bigArray[offsets[i] + j];
          }
      }
  }


#ifndef DOXYGEN_SHOULD_SKIP_THIS

  /* --------------- std::string specializations --------------------- */

  inline void MPIContainerComm<std::string>::bcast(std::string& x,
                                              int src, const MPIComm& comm)
  {
    int len = x.length();
    MPIContainerComm<int>::bcast(len, src, comm);

    x.resize(len);
    comm.bcast((void*)&(x[0]), len, MPITraits<char>::type(), src);
  }


  inline void MPIContainerComm<std::string>::bcast(Array<std::string>& x, int src,
                                              const MPIComm& comm)
  {
    /* begin by packing all the data into a big char array. This will
     * take a little time, but will be cheaper than multiple MPI calls */
    Array<char> bigArray;
    Array<int> offsets;
    if (comm.getRank()==src)
      {
        getBigArray(x, bigArray, offsets);
      }

    /* now broadcast the big array and the offsets */
    MPIContainerComm<char>::bcast(bigArray, src, comm);
    MPIContainerComm<int>::bcast(offsets, src, comm);

    /* finally, reassemble the array of strings */
    if (comm.getRank() != src)
      {
        getStrings(bigArray, offsets, x);
      }
  }

  inline void MPIContainerComm<std::string>::bcast(Array<Array<std::string> >& x,
                                              int src, const MPIComm& comm)
  {
    int len = x.length();
    MPIContainerComm<int>::bcast(len, src, comm);

    x.resize(len);
    for (int i=0; i<len; i++)
      {
        MPIContainerComm<std::string>::bcast(x[i], src, comm);
      }
  }


  inline void MPIContainerComm<std::string>::allGather(const std::string& outgoing,
                                                  Array<std::string>& incoming,
                                                  const MPIComm& comm)
  {
    int nProc = comm.getNProc();

    int sendCount = outgoing.length();

    incoming.resize(nProc);

    int* recvCounts = new int[nProc];
    int* recvDisplacements = new int[nProc];

    /* share lengths with all procs */
    comm.allGather((void*) &sendCount, 1, MPIComm::INT,
                   (void*) recvCounts, 1, MPIComm::INT);


    int recvSize = 0;
    recvDisplacements[0] = 0;
    for (int i=0; i<nProc; i++)
      {
        recvSize += recvCounts[i];
        if (i < nProc-1)
          {
            recvDisplacements[i+1] = recvDisplacements[i]+recvCounts[i];
          }
      }

    char* recvBuf = new char[recvSize];

    comm.allGatherv((void*) outgoing.c_str(), sendCount, MPIComm::CHAR,
                    recvBuf, recvCounts, recvDisplacements, MPIComm::CHAR);

    for (int j=0; j<nProc; j++)
      {
        char* start = recvBuf + recvDisplacements[j];
        char* tmp = new char[recvCounts[j]+1];
        std::memcpy(tmp, start, recvCounts[j]);
        tmp[recvCounts[j]] = '\0';
        incoming[j] = std::string(tmp);
        delete [] tmp;
      }
    
    delete [] recvCounts;
    delete [] recvDisplacements;
    delete [] recvBuf;
  }
  
  inline void MPIContainerComm<std::string>::gatherv(const Array<std::string>& outgoing,
                                                Array<Array<std::string> >& incoming,
                                                int root,
                                                const MPIComm& comm)
  {
    int nProc = comm.getNProc();

    Array<char> packedLocalArray;
    pack(outgoing, packedLocalArray);

    int sendCount = packedLocalArray.size();

    /* gather the message sizes from all procs */
    Array<int> recvCounts(nProc);
    Array<int> recvDisplacements(nProc);

    comm.gather((void*) &sendCount, 1, MPIComm::INT,
                (void*) &(recvCounts[0]), 1, MPIComm::INT, root);
    
    /* compute the displacements */
    int recvSize = 0;
    if (root == comm.getRank())
      {
        recvDisplacements[0] = 0;
        for (int i=0; i<nProc; i++)
          {
            recvSize += recvCounts[i];
            if (i < nProc-1)
              {
                recvDisplacements[i+1] = recvDisplacements[i]+recvCounts[i];
              }
          }
      }

    /* set the size to 1 on non-root procs */
    Array<char> recvBuf(std::max(1,recvSize));
    

    void* sendBuf = (void*) &(packedLocalArray[0]);
    void* inBuf = (void*) &(recvBuf[0]);
    int* inCounts = inCounts = &(recvCounts[0]);
    int* inDisps = inDisps = &(recvDisplacements[0]);

    /* gather the packed data */
    comm.gatherv( sendBuf, sendCount, MPIComm::CHAR,
                  inBuf, inCounts, inDisps,
                  MPIComm::CHAR, root);

    /* on the root, unpack the data */
    if (comm.getRank()==root)
      {
        incoming.resize(nProc);
        for (int j=0; j<nProc; j++)
          {
            char* start = &(recvBuf[0]) + recvDisplacements[j];
            Array<char> tmp(recvCounts[j]+1);
            std::memcpy(&(tmp[0]), start, recvCounts[j]);
            tmp[recvCounts[j]] = '\0';
            unpack(tmp, incoming[j]);
          }
      }
                 
                 
  }


  inline void MPIContainerComm<std::string>::getBigArray(const Array<std::string>& x,
                                                    Array<char>& bigArray,
                                                    Array<int>& offsets)
  {
    offsets.resize(x.length()+1);
    int totalLength = 0;

    for (int i=0; i<x.length(); i++)
      {
        offsets[i] = totalLength;
        totalLength += x[i].length();
      }
    offsets[x.length()] = totalLength;

    bigArray.resize(totalLength);

    for (int i=0; i<x.length(); i++)
      {
        for (unsigned int j=0; j<x[i].length(); j++)
          {
            bigArray[offsets[i]+j] = x[i][j];
          }
      }
  }

  inline void MPIContainerComm<std::string>::pack(const Array<std::string>& x,
                                             Array<char>& bigArray)
  {
    Array<int> offsets(x.size()+1);
    int headerSize = (x.size()+2) * sizeof(int);

    int totalLength = headerSize;

    for (int i=0; i<x.length(); i++)
      {
        offsets[i] = totalLength;
        totalLength += x[i].length();
      }
    offsets[x.length()] = totalLength;

    /* The array will be packed as follows:
     * [numStrs, offset1, ... offsetN, characters data] 
     */

    bigArray.resize(totalLength);

    int* header = reinterpret_cast<int*>( &(bigArray[0]) );
    header[0] = x.size();
    for (unsigned int i=0; i<=x.size(); i++)
      {
        header[i+1] = offsets[i];
      }

    for (int i=0; i<x.length(); i++)
      {
        for (unsigned int j=0; j<x[i].length(); j++)
          {
            bigArray[offsets[i]+j] = x[i][j];
          }
      }
  }

  inline void MPIContainerComm<std::string>::unpack(const Array<char>& packed,
                                             Array<std::string>& x)
  {
    const int* header = reinterpret_cast<const int*>( &(packed[0]) );

    x.resize(header[0]);
    Array<int> offsets(x.size()+1);
    for (unsigned int i=0; i<=x.size(); i++) offsets[i] = header[i+1];

    for (unsigned int i=0; i<x.size(); i++)
      {
        x[i].resize(offsets[i+1]-offsets[i]);
        for (unsigned int j=0; j<x[i].length(); j++)
          {
            x[i][j] = packed[offsets[i] + j];
          }
      }
  }

  inline void MPIContainerComm<std::string>::getStrings(const Array<char>& bigArray,
                                                   const Array<int>& offsets,
                                                   Array<std::string>& x)
  {
    x.resize(offsets.length()-1);
    for (int i=0; i<x.length(); i++)
      {
        x[i].resize(offsets[i+1]-offsets[i]);
        for (unsigned int j=0; j<x[i].length(); j++)
          {
            x[i][j] = bigArray[offsets[i] + j];
          }
      }
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS

}


#endif


