/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @file heap.h
 *
 * Simple priority queue implementation.
 */

#ifndef COLLECTIONS_HEAP_H
#define COLLECTIONS_HEAP_H

#include <vector>
#include <algorithm>
#include "../base/base.h"

/**
 * Priority queue implemented as a heap.
 *
 * Note that a heap isn't really a collection, but actually it is just a
 * prioritization data structure.  Storing large objects in will incur copy
 * overheads, so Key and Value should probably be either primitives
 * (integers, floats, pointers) or small structures.
 */
template <typename TKey, typename TValue = Empty>
class MinHeap {

 public:
  typedef TKey Key;
  typedef TValue Value;

 private:
  struct Entry {
    TKey key;
    TValue value;

	 /** 
	  * Actually >. Cheap hack.
	  * 
	  * FIXME: Replace with comp function used by std::heap methods.
	  */
	 inline bool operator< (const Entry& other) const {
		 return key > other.key;
	 }
	 /**
	  * Standard assignment operator.
	  */
	 inline Entry& operator= (const Entry& other) {
		 key = other.key;
		 value = other.value;
		 return *this;
	 }
  };

  std::vector<Entry> entries_;
 
 public:

  MinHeap(const unsigned int size=0) {
	  entries_.reserve(size);
  }

  /**
   * Initializes an empty priority queue. Deprecated, use constructor instead.
   */
  void Init() {
  }

  /**
   * Detects whether this queue is empty.
   */
  bool is_empty() const {
    return entries_.size() == 0;
  }

  /**
   * Places a value at the specified priority.
   *
   * @param key the priority
   * @param value the value associated with the priority
   */
  void Put(const Key key, const Value& value) {
    Entry entry;

    entry.key = key;
    entry.value = value;

	 entries_.push_back( entry );
	std::push_heap( entries_.begin(), entries_.end());
  }

  /**
   * Pops and returns the lowest element off the heap.
   *
   * @return the value associated with the highest priority
   */
  inline Value Pop() {
    Value t = entries_[0].value;

	  std::pop_heap( entries_.begin(), entries_.end() );
	  entries_.pop_back();

    return t;
  }

  /**
   * Gets the value at the top of the heap.
   */
  Value top() const {
    return entries_[0].value;
  }

  /**
   * Gets the key at the top of the heap.
   */
  Key top_key() const {
    return entries_[0].key;
  }
  
  /**
   * Replaces the top item on the heap.
   */
  void set_top(const Value& v) {
    entries_[0].value = v;
  }

  /**
   * Gets the size of the heap.
   */
  index_t size() const {
    return entries_.size();
  }

};

#endif
