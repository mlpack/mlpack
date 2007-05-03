// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
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
  };
  
  ArrayList<Entry> entries_;
  
  OT_DEF(MinHeap) {
    OT_MY_OBJECT(entries_);
  }
  
 public:
  MinHeap() {}
  ~MinHeap() {}
  
  /**
   * Copy constructor (for use in collections only!).
   */
  MinHeap(const MinHeap& other) {
    Copy(other);
  }
  CC_ASSIGNMENT_OPERATOR(MinHeap);
 
  /**
   * Initializes an empty priority queue.
   */
  void Init() {
    entries_.Init();
  }
  
  /**
   * Serializes this heap.
   *
   * Currently only works for things that are bit-copiable, containing no
   * pointers.
   */
  template<typename Serializer>
  void Serialize(Serializer *s) const {
    entries_.Serialize(s);
  }
  
  /**
   * Initializes this heap, deserializing from the given source.
   */
  template<typename Deserializer>
  void Deserialize(Deserializer *s) {
    entries_.Deserialize(s);
  }
  
  /**
   * Places a value at the specified priority.
   *
   * @param key the priority
   * @param value the value associated with the priority
   */
  void Put(Key key, Value value) {
    Entry entry;
    
    entries_.AddBack();
    
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
    Entry entry = *entries_.PopBackPtr();
    
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
  
  /**
   * Copies another MinHeap.
   */
  void Copy(const MinHeap& other) {
    entries_.Copy(other.entries_);
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
