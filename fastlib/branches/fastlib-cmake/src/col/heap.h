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

#include "arraylist.h"

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
  // TODO: A copiable heap probably isn't a bad idea

 public:
  typedef TKey Key;
  typedef TValue Value;

 private:
  struct Entry {
    TKey key;
    TValue value;

    OBJECT_TRAVERSAL_SHALLOW(Entry) {
      OT_OBJ(key);
      OT_OBJ(value);
    }
  };

  ArrayList<Entry> entries_;

  /*OBJECT_TRAVERSAL(MinHeap) {
    OT_OBJ(entries_);
  }*/

 public:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    //ArrayList to be replaced by std::vector
    //ar & entries_;
  }

  /**
   * Initializes an empty priority queue.
   */
  void Init() {
    entries_.Init();
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
  void Put(Key key, Value value) {
    Entry entry;

    entries_.PushBack();

    entry.key = key;
    entry.value = value;

    WalkUp_(entry, entries_.size() - 1);
  }

  /**
   * Pops and returns the lowest element off the heap.
   *
   * @return the value associated with the highest priority
   */
  Value Pop() {
    Value t = entries_[0].value;

    PopOnly();

    return t;
  }

  /**
   * Removes the lowest element from the heap.
   *
   * Simply pops the top value on the queue, without
   * returning it.
   */
  void PopOnly() {
    Entry entry;
    entries_.PopBackInit(&entry);

    if (likely(entries_.size() != 0)) {
      WalkDown_(entry, 0);
    }
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
  void set_top(Value v) {
    entries_[0].value = v;
  }

  /**
   * Gets the size of the heap.
   */
  index_t size() const {
    return entries_.size();
  }

 private:
  static index_t ChildIndex_(index_t i) {
    return (i << 1) + 1;
  }

  static index_t ParentIndex_(index_t i) {
    return (i - 1) >> 1;
  }

  index_t WalkDown_(const Entry& entry, index_t i) {
    Key key = entry.key;
    Entry *entries = entries_.begin();
    index_t last = entries_.size() - 1;

    for (;;) {
      index_t c = ChildIndex_(i);

      if (unlikely(c > last)) {
        break;
      }

      // TODO: This "if" can be avoided if we're more intelligent...
      if (likely(c != last)) {
        c += entries[c + 1].key < entries[c].key ? 1 : 0;
      }

      if (key <= entries[c].key) {
        break;
      }

      entries[i] = entries[c];
      i = c;
    }

    entries[i] = entry;

    return i;
  }

  index_t WalkUp_(const Entry& entry, index_t i) {
    Key key = entry.key;
    Entry *entries = entries_.begin();

    for (;;) {
      index_t p;

      if (unlikely(i == 0)) {
        break; // highly unlikely, we found the best!
      }

      p = ParentIndex_(i);

      if (key >= entries[p].key) {
        break;
      }

      entries[i] = entries[p];

      i = p;
    }

    entries[i] = entry;

    return i;
  }
};

#endif
