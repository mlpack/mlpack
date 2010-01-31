/*
 * =====================================================================================
 * 
 *       Filename:  kdnode.h
 * 
 *    Description:  A simple kdnode apporach
 * 
 *        Version:  1.0
 *        Created:  02/09/2007 08:25:15 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef KDNODE_H_
#define KDNODE_H_
#include "bounding_box.h"
#include "node.h"
template<typename PRECISION, 
         typename IDPRECISION,
         typename ALLOCATOR, 
				 bool diagnostic>
class KdNode : public  
    Node<PRECISION, IDPRECISION, 
	       HyperRectangle, 
		   	 KdNode<PRECISION, IDPRECISION, 
				        ALLOCATOR, diagnostic>,	
			   ALLOCATOR, diagnostic> {
 public:
	typedef Node<PRECISION, 
					     IDPRECISION, 
							 HyperRectangle, 
							 KdNode<PRECISION, IDPRECISION, 
				              ALLOCATOR, diagnostic>,
							 ALLOCATOR, diagnostic> Node_t; 
  // Use this for node
  KdNode(typename Node_t::Pivot_t *pivot, IDPRECISION node_id) : 
		Node_t(pivot, node_id) {
	 }
	// Use this for leaf
  KdNode(typename Node_t::Pivot_t *pivot, IDPRECISION node_id,
                DataReader<PRECISION, IDPRECISION> *data) : 
		Node_t(pivot, node_id, data) {
	}

};


#endif  //KDNODE_H_
