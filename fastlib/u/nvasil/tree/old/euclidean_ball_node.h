/*
 * =====================================================================================
 * 
 *       Filename:  euclidean_ball_node.h
 * 
 *    Description:
 * 
 *        Version:  1.0
 *        Created:  02/11/2007 11:09:46 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef EUCLIDEAN_BALL_NODE_H_
#define EUCLIDEAN_BALL_NODE_H_
#include "hyper_ball.h"
#include "node.h"
#include "euclidean_metric.h"

template<typename PRECISION, 
	       typename ALLOCATOR, 
				 bool diagnostic>
class EuclideanHyperBall : public HyperBall<PRECISION,
	                                          EuclideanMetric<PRECISION>, 
                                            ALLOCATOR,
																		        diagnostic> {
 public:
	typedef	HyperBall<PRECISION, 
					          EuclideanMetric<PRECISION>,
									 	ALLOCATOR, diagnostic> HyperBall_t;
	EuclideanHyperBall(typename HyperBall_t::PivotData &pv) : 
		HyperBall<PRECISION, EuclideanMetric<PRECISION>, ALLOCATOR, diagnostic>(pv) {
	}

};


template<typename PRECISION, 
         typename IDPRECISION,
         typename ALLOCATOR, 
				 bool diagnostic>
class EuclideanBallNode : public  
    Node<PRECISION, IDPRECISION, 
	       EuclideanHyperBall, 
		   	 EuclideanBallNode<PRECISION, IDPRECISION, 
				        ALLOCATOR, diagnostic>,	
			   ALLOCATOR, diagnostic> {
 public:
	typedef Node<PRECISION, 
					     IDPRECISION, 
							 EuclideanHyperBall, 
							 EuclideanBallNode<PRECISION, IDPRECISION, 
				                         ALLOCATOR, diagnostic>,
							 ALLOCATOR, diagnostic> Node_t; 
  typedef EuclideanMetric<PRECISION> Metric_t;
  // Use this for node
  EuclideanBallNode(typename Node_t::Pivot_t *pivot, IDPRECISION node_id) : 
		Node_t(pivot, node_id) {
	 }
	// Use this for leaf
  EuclideanBallNode(typename Node_t::Pivot_t *pivot, IDPRECISION node_id,
                DataReader<PRECISION, IDPRECISION> *data) : 
		Node_t(pivot, node_id, data) {
	}

};




#endif  // EUCLIDEAN_BALL_NODE_H_

