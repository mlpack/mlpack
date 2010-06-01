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
 * @file queue.h
 *
 * A first-in first-out queue.
 */

#ifndef COL_QUEUE_H
#define COL_QUEUE_H

#include <deque>

/**
 * A FIFO queue.
 *
 * A wrapper for std::deque.
 */
template<typename T>
class Queue {

 private:
	std::deque<T> queue;

 public:
  /**
   * Does nothing. Deprecated.
   */
  void Init() {
  }

  /**
   * Adds an element to the tail end of the queue, but not initializing it.
   *
   * @return a pointer to the default-constructed but uninitialized value
   */
/*  T *Add() {
    Node *node = new Node();
    *tailp_ = node;
    tailp_ = &node->next;
    return &node->data;
  }
*/
  /**
   * Adds the specified element to the tail end.
   *
   * @return a pointer to the tail end, which contains the given parameter
   */
  T *Add(const T& value=NULL) {
		queue.push_back(value);
		return &queue.back();
  }

  /**
   * Pops from the head of the queue, not returning anything.
   */
  void PopOnly() {
		queue.pop_front();
  }

  /**
   * Pops from the head of the queue, returning a copy of the item.
   */
  T Pop() {
		T c = queue.front();
		queue.pop_front();
		return c;
  }

  /**
   * Determines if the queue is empty.
   */
  bool is_empty() const {
    return queue.empty();
  }

  /**
   * Gets the element at the head of the queue.
   */
  const T& top() const {
    return queue.front();
  }

  /**
   * Gets the element at the head of the queue.
   */
  T& top() {
    return queue.front();
  }

  /**
   * Clears all elements from the queue. 
   */
  void Clear() {
		queue.erase();
  }
};

#endif
