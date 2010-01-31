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

#ifndef TEUCHOS_HASHSET_H
#define TEUCHOS_HASHSET_H

/*! \file Teuchos_HashSet.hpp
    \brief Templated hashtable-based set
*/

#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_HashUtils.hpp"

namespace Teuchos
{
  using std::string;


  /** \ingroup Containers
   * \brief Templated hashtable-based set.

    HashSet is a hashtable-based set, similar to the STL set class
   * or the Java HashSet class.
   */
  template<class Key> class HashSet
    {
    public:

      //! Create an empty HashSet
      inline HashSet(int capacity=19);

      //! Check for the presence of a key
      inline bool containsKey(const Key& key) const ;

      //! Put a new object into the table.
      inline void put(const Key& key);

      //! Remove from the table the element given by key.
      inline void remove(const Key& key);

      //! Get the number of elements in the table
      inline int size() const {return count_;}

      //! Get list of keys in Array form
      inline Array<Key> arrayify() const ;

      //! Get list of keys in Array form
      inline void arrayify(Array<Key>& keys) const ;

      //! Write to a std::string
      inline std::string toString() const ;

    private:
      /** rebuild the hashtable when the size has changed */
      inline void rehash();
      /** get the next prime number near a given capacity */
      inline int nextPrime(int newCap) const ;

      Array<Array<Key> > data_;
      int count_;
      int capacity_;
      mutable Key mostRecentKey_;
    };


  /** \relates HashSet 
      \brief Write HashSet to a stream 
  */
  template<class Key>
    std::ostream& operator<<(std::ostream& os, const HashSet<Key>& h);

  template<class Key> inline
    std::string toString(const HashSet<Key>& h) {return h.toString();}


  template<class Key> inline
    HashSet<Key>::HashSet(int capacity)
    : data_(), count_(0), capacity_(HashUtils::nextPrime(capacity))
    {
      data_.resize(capacity_);
    }

  template<class Key> inline
    bool HashSet<Key>::containsKey(const Key& key) const
    {
      const Array<Key>& candidates
        = data_[hashCode(key) % capacity_];

      for (int i=0; i<candidates.length(); i++)
        {
          const Key& c = candidates[i];
          if (c == key)
            {
              return true;
            }
        }
      return false;
    }

  template<class Key> inline
    void HashSet<Key>::put(const Key& key)
    {
      int index = hashCode(key) % capacity_;

      Array<Key>& local = data_[index];

      // check for duplicate key
      for (int i=0; i<local.length(); i++)
        {
          if (local[i] == key)
            {
              return;
            }
        }

      // no duplicate key, so increment element count by one.
      count_++;

      // check for need to resize.
      if (count_ > capacity_)
        {
          capacity_ = HashUtils::nextPrime(capacity_+1);
          rehash();
          // recaluate index
          index = hashCode(key) % capacity_;
        }

      data_[index].append(key);
    }



  template<class Key> inline
    void HashSet<Key>::rehash()
    {
      Array<Array<Key> > tmp(capacity_);

      for (int i=0; i<data_.length(); i++)
        {
          for (int j=0; j<data_[i].length(); j++)
            {
              int newIndex = hashCode(data_[i][j]) % capacity_;
              tmp[newIndex].append(data_[i][j]);
            }
        }

      data_ = tmp;
    }

  template<class Key> inline
    Array<Key> HashSet<Key>::arrayify() const
    {
      Array<Key> rtn;
      rtn.reserve(size());

      for (int i=0; i<data_.length(); i++)
        {
          for (int j=0; j<data_[i].length(); j++)
            {
              rtn.append(data_[i][j]);
            }
        }

      return rtn;
    }

  template<class Key> inline
    void HashSet<Key>::arrayify(Array<Key>& rtn) const
    {
      rtn.resize(0);

      for (int i=0; i<data_.length(); i++)
        {
          for (int j=0; j<data_[i].length(); j++)
            {
              rtn.append(data_[i][j]);
            }
        }
    }

  template<class Key>  inline
    std::string HashSet<Key>::toString() const
    {
      std::string rtn = "HashSet[";

      bool first = true;

      for (int i=0; i<data_.length(); i++)
        {
          for (int j=0; j<data_[i].length(); j++)
            {
              if (!first) rtn += ", ";
              first = false;
              rtn += Teuchos::toString(data_[i][j]);
            }
        }
      rtn += "]";
      return rtn;
    }


  template<class Key> inline
    void HashSet<Key>::remove(const Key& key)
    {
      TEST_FOR_EXCEPTION(!containsKey(key),
                         std::runtime_error,
                         "HashSet<Key>::remove: key " 
                         << Teuchos::toString(key) 
                         << " not found in HashSet"
                         << toString());

      count_--;
      int h = hashCode(key) % capacity_;
      Array<Key>& candidates = data_[h];

      for (int i=0; i<candidates.length(); i++)
        {
          if (candidates[i] == key)
            {
              candidates.remove(i);
              break;
            }
        }
    }



  template<class Key>  inline
    std::ostream& operator<<(std::ostream& os, const HashSet<Key>& h)
    {
      return os << h.toString();
    }


}

#endif // TEUCHOS_HASHSET_H
