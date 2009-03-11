/**
 * @file factor_graph.h
 *
 * This file contains definition of a factor graph
 * 
 */

#ifndef GM_FACTOR_GRAPH_H
#define GM_FACTOR_GRAPH_H

#include "bipartie.h"
#include <fastlib/fastlib.h>

/** A factor graph is a bipartie graph with nodes on the left
 *  and factors on the right. Messages are passed back and forth
 *  between nodes and factors.
 *  For this template to work, the following members MUST be implemented
 *
 *  Class Node: 
 *    typedef RangeType;
 *    RangeType GetRange();  --> to be compared to Factor.GetRange(i_arg)
 *    RangeType GetVal(const SymbolType& sym) 
 *         --> return numerical value corresponding to the symbol
 *             which can be anything (e.g. numeric, string)
 *
 *  Class Factor:
 *    typedef RangeType;
 *    RangeType GetRange(i_arg);
 *    index_t n_agrs();      
 *         --> to be compared to bgraph_.n_factornodes(i_factor)
 *    double GetFactorVal(); 
 *         --> compute the factor given its arguments
 *    void SetArg(index_t i_arg, RangeType val);
 *         --> to set certain argument of the factor
 *    RangeType GetArg(index_t i_arg);
 *
 */
template <class Node, class Factor, 
	  class MessageNode2Factor, class MessageFactor2Node>
class FactorGraph {
 public:
  typedef Node NodeType;
  typedef Factor FactorType;
  typedef MessageNode2Factor MessageNode2FactorType;
  typedef MessageFactor2Node MessageFactor2NodeType;
  typedef BipartieGraph<MessageNode2FactorType, MessageFactor2NodeType> BGraphType;

 private:
  /** List of nodes in the graph */
  ArrayList<NodeType> nodes_;

  /** Observed values of variable, -1 for unseen nodes */
  ArrayList<typename NodeType::RangeType> observeds_;

  /** List of factors in the graph */
  ArrayList<FactorType> factors_;

  /** The bipartie graph structure */
  BGraphType bgraph_;

  /** The Z constant - common denominator - normalization constant */
  double Z_;

  OT_DEF(FactorGraph) {
    OT_MY_OBJECT(nodes_);
    OT_MY_OBJECT(observeds_);
    OT_MY_OBJECT(factors_);
    OT_MY_OBJECT(bgraph_);
    OT_MY_OBJECT(Z_);
  }

 public:
  /** Initialize a factor graph from a set of nodes and factors
   *  and factor-to-node connections 
   */
  void Init(const ArrayList<NodeType>& nodes, const ArrayList<FactorType>& factors,
	    const ArrayList<ArrayList<index_t> >& factor2node) {
    for (index_t i_factor = 0; i_factor < factors.size(); i_factor++)
      DEBUG_ASSERT(factors[i_factor].n_args() == factor2node[i_factor].size());

    nodes_.InitCopy(nodes);
    observeds_.Init(nodes_.size());

    factors_.InitCopy(factors);

    bgraph_.Init(nodes_.size(), factor2node);

    DEBUG_ASSERT(factors_.size() == bgraph_.n_factors());
    for (index_t i_factor = 0; i_factor < bgraph_.n_factors(); i_factor++) {
      DEBUG_ASSERT(factors_[i_factor].n_args() == 
		   bgraph_.n_factornodes(i_factor));
      for (index_t i_edge = 0; 
	   i_edge < bgraph_.n_factornodes(i_factor); i_edge++) {
	index_t i_node = bgraph_.node(i_factor, i_edge);
	DEBUG_ASSERT(factors_[i_factor].GetRange(i_edge) ==
		     nodes_[i_node].GetRange());
      }
    }
   
    Z_ = 1; // initialize the normalization factor
  }

  /** Calculate factor value using arguments stored in its args_ */
  double GetFactorVal(index_t i_factor) {
    double val = factors_[i_factor].GetVal();
    //printf("    getval = %f\n", val);
    return val;
  }

  inline FactorType& GetFactor(index_t i_factor) {
    return factors_[i_factor];
  }

  inline NodeType& GetNode(index_t i_node) {
    return nodes_[i_node];
  }

  inline BGraphType& GetBGraph() {
    return bgraph_;
  }

  /** Set arguments */
  //  void SetArgs(index_t i_factor, index_t i_edge, index_t val) {
  //    args_[i_factor][i_edge] = val;
  //  }

  /** Set arguments from observed values */
  void SetArgsFromObserved(index_t i_factor) {
    for (index_t i_edge = 0; 
	 i_edge < bgraph_.n_factornodes(i_factor); i_edge++) {
      index_t i_node = bgraph_.node(i_factor, i_edge);
      factors_[i_factor].SetArg(i_edge, observeds_[i_node]);
    }
  }

  /** Set observed values */
  void SetObserveds(const ArrayList<index_t>& observeds) {
    DEBUG_ASSERT(observeds_.size() == observeds.size());
    for (index_t i_node = 0; i_node < observeds_.size(); i_node++)
      observeds_[i_node] = observeds[i_node];
  }

  /** Set observed values from string symbol */
  void SetObserveds(const ArrayList<String>& observed_syms) {
    DEBUG_ASSERT(observeds_.size() == observed_syms.size());
    for (index_t i_node = 0; i_node < observeds_.size(); i_node++)
      observeds_[i_node] = nodes_[i_node].GetVal(observed_syms[i_node]);
  }

  double GetZ() {
    return Z_;
  }

  void SetZ(double Z) {
    Z_ = Z;
  }

  /** Calculate the joint product of all factors in the graph
   *  given a realization of all variables/nodes
   */
  double jointProduct() {
    double rval = 1;
    for (index_t i_factor = 0; i_factor < factors_.size(); i_factor++) {
      SetArgsFromObserved(i_factor);
      rval *= GetFactorVal(i_factor);
    }
    return rval;
  }

};

#endif
