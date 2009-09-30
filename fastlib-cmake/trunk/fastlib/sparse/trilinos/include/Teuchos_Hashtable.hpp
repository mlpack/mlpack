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

#ifndef TEUCHOS_HASHTABLE_H
#define TEUCHOS_HASHTABLE_H

/*! \file Teuchos_Hashtable.hpp
    \brief Templated hashtable class
*/

#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_HashUtils.hpp"

namespace Teuchos
{
  using std::string;

  /** \ingroup Containers
   * \brief Helper class for Teuchos::Hashtable, representing a single <key, value> pair.
   */
  template<class Key, class Value> class HashPair
    {
    public:
      //! Empty constructor 
      inline HashPair() : key_(), value_() {;}
      //! Basic <key, value> constructor
      inline HashPair(const Key& key, const Value& value)
        : key_(key), value_(value) {;}

      //! Templated key variable
      Key key_;
      //! Templated value variable
      Value value_;
    };

  /**
     \ingroup Containers
     \brief Templated hashtable class.
     @author Kevin Long
  */
  template<class Key, class Value> class Hashtable
    {
    public:

      //! Create an empty Hashtable
      inline Hashtable(int capacity=101, double rehashDensity = 0.8);

      //! Check for the presence of a key
      inline bool containsKey(const Key& key) const ;

      //! Get the value indexed by key
      inline const Value& get(const Key& key) const ;

      //! Put a new (key, value) pair in the table.
      inline void put(const Key& key, const Value& value);

      //! Remove from the table the element given by key.
      inline void remove(const Key& key);

      //! Get the number of elements in the table
      inline int size() const {return count_;}

      //! Get lists of keys and values in Array form
      inline void arrayify(Array<Key>& keys, Array<Value>& values) const ;

      //! Return the average degeneracy (average number of entries per hash code).
      inline double avgDegeneracy() const {return avgDegeneracy_;}

      //! Return the density of the hashtable (num entries / capacity)
      inline double density() const {return ((double)count_)/((double) capacity_);}

      //! Set the density at which to do a rehash
      inline void setRehashDensity(double rehashDensity);

      //! Write to a std::string
      inline std::string toString() const ;

    private:

      inline void rehash();
      inline int nextPrime(int newCap) const ;
      inline void accumulateAvgFill(int n) const ;


      Array<Array<HashPair<Key, Value> > > data_;
      int count_;
      int capacity_;
      mutable Value mostRecentValue_;
      mutable Key mostRecentKey_;

      mutable int nHits_;
      mutable double avgDegeneracy_;
      double rehashDensity_;
    };

  template<class Key, class Value>
  std::string toString(const Hashtable<Key, Value>& h);

  /** \relates Hashtable 
      \brief Write Hashtable to a stream
  */
  template<class Key, class Value>
  std::ostream& operator<<(std::ostream& os, const Hashtable<Key, Value>& h);

  template<class Key, class Value> inline
    Hashtable<Key, Value>::Hashtable(int capacity, double rehashDensity)
    : data_(), count_(0), capacity_(HashUtils::nextPrime(capacity)),
    nHits_(0), avgDegeneracy_(0), rehashDensity_(rehashDensity)
    {
      data_.resize(capacity_);
    }

  template<class Key, class Value> inline
    bool Hashtable<Key, Value>::containsKey(const Key& key) const
    {
      const Array<HashPair<Key, Value> >& candidates
        = data_[hashCode(key) % capacity_];
      
      for (int i=0; i<candidates.length(); i++)
        {
          const HashPair<Key, Value>& c = candidates[i];
          if (c.key_ == key)
            {
              //          (Key&) mostRecentKey_ = key;
              //(Value&) mostRecentValue_ = c.value_;
              return true;
            }
        }
      return false;
    }

  template<class Key, class Value> inline
    void Hashtable<Key, Value>::put(const Key& key, const Value& value)
    {
      int index = hashCode(key) % capacity_;
      
      Array<HashPair<Key, Value> >& local = data_[index];
      
      // check for duplicate key
      for (int i=0; i<local.length(); i++)
        {
          if (local[i].key_ == key)
            {
              local[i].value_ = value;
              return;
            }
        }
      
      // no duplicate key, so increment element count by one.
      count_++;
      
      // check for need to resize.
      if ((double) count_ > rehashDensity_ * (double) capacity_)
        {
          capacity_ = HashUtils::nextPrime(capacity_+1);
          rehash();
          // recaluate index
          index = hashCode(key) % capacity_;
        }
      
      data_[index].append(HashPair<Key, Value>(key, value));
    }



  template<class Key, class Value> inline
    void Hashtable<Key, Value>::rehash()
    {
      Array<Array<HashPair<Key, Value> > > tmp(capacity_);

      for (int i=0; i<data_.length(); i++)
        {
          for (int j=0; j<data_[i].length(); j++)
            {
              int newIndex = hashCode(data_[i][j].key_) % capacity_;
              tmp[newIndex].append(data_[i][j]);
            }
        }
      
      data_ = tmp;
    }


  template<class Key, class Value> inline
    void Hashtable<Key, Value>::arrayify(Array<Key>& keys, Array<Value>& values) const
    {
      keys.reserve(size());
      values.reserve(size());

      for (int i=0; i<data_.length(); i++)
        {
          for (int j=0; j<data_[i].length(); j++)
            {
              keys.append(data_[i][j].key_);
              values.append(data_[i][j].value_);
            }
        }
    }

  template<class Key, class Value>  inline
  std::string Hashtable<Key, Value>::toString() const 
  {
    Array<Key> keys;
    Array<Value> values;
    arrayify(keys, values);
    
    std::string rtn = "[";
    for (int i=0; i<keys.length(); i++)
      {
        rtn += "{" + Teuchos::toString(keys[i]) + ", " + Teuchos::toString(values[i])
          + "}";
        if (i < keys.length()-1) rtn += ", ";
      }
    rtn += "]";
    
    return rtn;
  }

  template<class Key, class Value>  inline
    std::string toString(const Hashtable<Key, Value>& h)
    {
      Array<Key> keys;
      Array<Value> values;
      h.arrayify(keys, values);

      std::string rtn = "[";
      for (int i=0; i<keys.length(); i++)
        {
          rtn += "{" + Teuchos::toString(keys[i]) + ", " + Teuchos::toString(values[i])
            + "}";
          if (i < keys.length()-1) rtn += ", ";
        }
      rtn += "]";

      return rtn;
    }

  template<class Key, class Value> inline
    const Value& Hashtable<Key, Value>::get(const Key& key) const
    {
      TEST_FOR_EXCEPTION(!containsKey(key),
                         std::runtime_error,
                         "Hashtable<Key, Value>::get: key " 
                         << Teuchos::toString(key) 
                         << " not found in Hashtable"
                         << toString());
      
      const Array<HashPair<Key, Value> >& candidates
        = data_[hashCode(key) % capacity_];

      accumulateAvgFill(candidates.length());
      
      for (int i=0; i<candidates.length(); i++)
        {
          const HashPair<Key, Value>& c = candidates[i];
          if (c.key_ == key)
            {
              return c.value_;
            }
        }
      return mostRecentValue_;
    }


  template<class Key, class Value> inline
    void Hashtable<Key, Value>::remove(const Key& key)
    {
      TEST_FOR_EXCEPTION(!containsKey(key),
                         std::runtime_error,
                         "Hashtable<Key, Value>::remove: key " 
                         << Teuchos::toString(key) 
                         << " not found in Hashtable"
                         << toString());

      count_--;
      int h = hashCode(key) % capacity_;
      const Array<HashPair<Key, Value> >& candidates = data_[h];

      for (int i=0; i<candidates.length(); i++)
        {
          const HashPair<Key, Value>& c = candidates[i];
          if (c.key_ == key)
            {
              data_[h].remove(i);
              break;
            }
        }
    }

  template<class Key, class Value> inline
  void Hashtable<Key, Value>::accumulateAvgFill(int n) const
  {
    avgDegeneracy_ = ((double) nHits_)/(nHits_ + 1.0) * avgDegeneracy_ + ((double) n)/(nHits_ + 1.0);
    nHits_++;
    }

  template<class Key, class Value>  inline
    std::ostream& operator<<(std::ostream& os, const Hashtable<Key, Value>& h)
    {
      return os << toString(h);
    }


}

#endif // TEUCHOS_HASHTABLE_H
