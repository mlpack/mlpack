/*
 * =====================================================================================
 * 
 *       Filename:  GeneralNode.h
 * 
 *    Description   Specialization of node
 * 
 *        Version:  1.0
 *        Created:  02/03/2007 03:55:40 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */
#ifndef GENERAL_NODE_H_
#define GENERAL_NODE_H_

#include "node.h"
template<typename PRECISION, 
         typename IDPRECISION,
         template<typename, typename, bool> class BOUNDINGBOX,
         typename ALLOCATOR, 
				 bool diagnostic>
class GeneralNode : public  
    Node<PRECISION, IDPRECISION, 
	       BOUNDINGBOX, 
		   	 GeneralNode<PRECISION, IDPRECISION, 
				                    BOUNDINGBOX, ALLOCATOR,
													  diagnostic>,	
			   ALLOCATOR, diagnostic> {
 public:
	typedef Node<PRECISION, 
					     IDPRECISION, 
							 BOUNDINGBOX, 
							 GeneralNode<PRECISION, IDPRECISION, 
				                    BOUNDINGBOX, ALLOCATOR,
													  diagnostic>,
							 ALLOCATOR, diagnostic> Node_t; 
  // Use this for node
  GeneralNode(typename Node_t::Pivot_t *pivot, IDPRECISION node_id) : 
		Node_t(pivot, node_id) {
	 }
	// Use this for leaf
  GeneralNode(typename Node_t::Pivot_t *pivot, IDPRECISION node_id,
                DataReader<PRECISION, IDPRECISION> *data) : 
		Node_t(pivot, node_id, data) {
	}

};

#endif //GENERAL_NODE_H
