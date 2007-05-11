// Copyright (C) 2001 Octavian Procopiuc
//
// File:    ami_block.h
// Authors: Octavian Procopiuc <tavi@cs.duke.edu>
//
// Definition and implementation of the AMI_block class.
//
// $Id: ami_block.h,v 1.9 2005/01/21 17:29:26 tavi Exp $
//

#ifndef _AMI_BLOCK_H
#define _AMI_BLOCK_H

// Get definitions for working with Unix and Windows
#include <portability.h>

// The AMI_block_base class.
#include <ami_block_base.h>

// The b_vector class.
#include <b_vector.h>
 
template<class E, class I, class BTECOLL = BTE_COLLECTION >
class AMI_block: public AMI_block_base<BTECOLL> {
  protected:
  using AMI_block_base<BTECOLL>::bid_;
  using AMI_block_base<BTECOLL>::dirty_;
  using AMI_block_base<BTECOLL>::pdata_;
  using AMI_block_base<BTECOLL>::per_;
  using AMI_block_base<BTECOLL>::pcoll_;
  
  public:
  using AMI_block_base<BTECOLL>::bid;
  
  //  typedef typename BTECOLL::block_id_t id_t;

  // The array of links.
  b_vector<AMI_bid> lk;

  // The array of elements.
  b_vector<E> el;

  typedef typename b_vector<AMI_bid>::iterator  lk_iterator;
  typedef typename b_vector<E>::iterator  el_iterator;

public:
  // Compute the capacity of the el vector statically (but you have to
  // give the correct block size and number of links!). It can also be
  // used to figure out how many elements would fit into a block with
  // a given size and a given number of links.
  static size_t el_capacity(size_t block_size, size_t links);

  // Constructor.  Read and initialize a block with a given ID. If the
  // ID is missing or 0, a new block is created.
  AMI_block(AMI_collection_single<BTECOLL>* pacoll, size_t links, AMI_bid bid = 0);

  // Get a reference to the info field.
  I* info();
  const I* info() const;
};


////////// ***Implementation*** ///////////



////////////////////////////////////
////////// **AMI_block** ///////////
////////////////////////////////////

template<class E, class I, class BTECOLL>
size_t AMI_block<E,I,BTECOLL>::el_capacity(size_t block_size, size_t links) {
  return (size_t) ((block_size - sizeof(I) - links * sizeof(AMI_bid)) / sizeof(E));
}

template<class E, class I, class BTECOLL>
AMI_block<E,I,BTECOLL>::AMI_block(AMI_collection_single<BTECOLL>* pacoll, 
		  size_t links, AMI_bid _bid):
  AMI_block_base<BTECOLL>(pacoll, _bid), 
  lk((AMI_bid*)pdata_, links),
  el((E*) ((char*) pdata_ + links * sizeof(AMI_bid)), 
     el_capacity(pcoll_->block_size(), links))
{
}

template<class E, class I, class BTECOLL>
I* AMI_block<E,I,BTECOLL>::info() {
  return (I*) (((char*) pdata_ + (lk.capacity()*sizeof(AMI_bid) + 
				  el.capacity()*sizeof(E))));
}

template<class E, class I, class BTECOLL>
const I*  AMI_block<E,I,BTECOLL>::info() const {
  return (I*) (((char*) pdata_ + (lk.capacity()*sizeof(AMI_bid) + 
				  el.capacity()*sizeof(E))));
}

#endif // _AMI_BLOCK_H
