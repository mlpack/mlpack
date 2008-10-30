/**
 * @file gm.h
 *
 * This file contains definition of a factor graph
 * 
 */

#ifndef GM_FACTOR_GRAPH_H
#define GM_FACTOR_GRAPH_H

#include "bipartie.h"
#include <fastlib/fastlib.h>
#include <queue>

class Node {
  /** The symbols for each values of the variable represented by the node */
  ArrayList<String> symbols_;

  OT_DEF(Node) {
    OT_MY_OBJECT(symbols_);
  }

  friend class FactorGraph;

 public:
  /** Initialize the set of symbols */
  void Init(const String* symbols, index_t n_syms) {
    symbols_.InitCopy(symbols, n_syms);
  }

  /** Get a symbol string */
  String GetSymbol(index_t i) {
    DEBUG_ASSERT(i < symbols_.size());
    return symbols_[i];
  }

  /** Set a symbol string */
  void SetSymbol(index_t i, const String& sym) {
    DEBUG_ASSERT(i < symbols_.size());    
    symbols_[i] = sym;
  }

  /** Get the index of a symbol 
   *  return -1 if not found
   */
  index_t GetIndex(const String& sym) {
    for (int i = 0; i < symbols_.size(); i++)
      if (symbols_[i] == sym) return i;
    return -1; // not found;
  }

  index_t GetRange() {
    return symbols_.size();
  }
};

class Factor {
  /** The ranges of each variable of the factor 
   *  (length of symbols_ in each node)
   */
  ArrayList<index_t> ranges_;

  /** Probabilities/values table (multi-dimentional) */
  Vector vals_;

  OT_DEF(Factor) {
    OT_MY_OBJECT(ranges_);
    OT_MY_OBJECT(vals_);
  }

  friend class FactorGraph;

 public:
  /** Calculate index from position which is stored 
   *  in column-wise manner
   */
  index_t GetIndex(const ArrayList<index_t>& pos) {
    DEBUG_ASSERT(pos.size() == ranges_.size());

    index_t n = pos.size()-1;
    index_t rpos = pos[n];
    DEBUG_ASSERT(pos[n] < ranges_[n] && pos[n] >= 0);
    for (int i = n-1; i >= 0; i--) {
      DEBUG_ASSERT(pos[i] < ranges_[i] && pos[i] >= 0);
      rpos = rpos*ranges_[i]+pos[i];
    }

    return rpos;
  }

  /** Get probability/value at a position in the table */
  double GetVal(const ArrayList<index_t>& pos) {
    index_t rpos = GetIndex(pos);
    return vals_[rpos];
  }

  /** Get probability/value from index in the table */
  double GetVal(index_t pos) {
    return vals_[pos];
  }

  /** Set probability/value at a position in the table */
  void SetVal(const ArrayList<index_t>& pos, double val) {
    index_t rpos = GetIndex(pos);
    vals_[rpos] = val;
  }

  /** Set probability/val from index in the table */
  void SetVal(index_t pos, double val) {
    vals_[pos] = val;
  }

  index_t GetRange(index_t i_arg) {
    DEBUG_ASSERT(i_arg < n_args());
    return ranges_[i_arg];
  }

  /** Initialize the factor with ranges specified */
  void Init(const ArrayList<index_t>& ranges) {
    index_t len = 1;

    for (int i = 0; i < ranges.size(); i++)
      len *= ranges[i];

    ranges_.InitCopy(ranges);
    vals_.Init(len);
  }

  /** Initialize the factor with ranges and probabilities/values specified */
  void Init(const ArrayList<index_t>& ranges, const Vector& vals) {
    index_t len = 1;

    for (int i = 0; i < ranges.size(); i++)
      len *= ranges[i];

    DEBUG_ASSERT(len == vals.length());

    ranges_.InitCopy(ranges);
    vals_.Copy(vals);
  }

  /** Initialize the factor with ranges and probabilities/values specified */
  void Init(const index_t* ranges, index_t n_nodes, const double* vals) {
    index_t len = 1;

    for (int i = 0; i < n_nodes; i++)
      len *= ranges[i];

    ranges_.InitCopy(ranges, n_nodes);
    vals_.Copy(vals, len);
  }

  index_t n_args() {
    return ranges_.size();
  }
};


class FactorGraph {
  /** List of nodes in the graph */
  ArrayList<Node> nodes_;

  /** Observed values of variable, -1 for unseen nodes */
  ArrayList<index_t> observeds_;

  /** List of factors in the graph */
  ArrayList<Factor> factors_;

  /** Temporary storage of factors' arguments */
  ArrayList<ArrayList<index_t> > args_;

  /** The bipartie graph structure */
  BipartieGraph<ArrayList<double>, ArrayList<double> > bgraph_;

  /** The Z constant - common denominator - normalization constant */
  double Z;

  OT_DEF(FactorGraph) {
    OT_MY_OBJECT(nodes_);
    OT_MY_OBJECT(observeds_);
    OT_MY_OBJECT(factors_);
    OT_MY_OBJECT(args_);
    OT_MY_OBJECT(bgraph_);
    OT_MY_OBJECT(Z);
  }

 public:
  /** Initialize a factor graph from a set of nodes and factors
   *  and factor-to-node connections 
   */
  void Init(const ArrayList<Node>& nodes, const ArrayList<Factor>& factors,
	    const ArrayList<ArrayList<index_t> >& factor2node) {
    DEBUG_ASSERT(factors.size() == factor2node.size());
    for (int i_factor = 0; i_factor < factors.size(); i_factor++)
      DEBUG_ASSERT(factors[i_factor].ranges_.size() ==
		   factor2node[i_factor].size());

    nodes_.InitCopy(nodes);
    observeds_.Init(nodes_.size());

    factors_.InitCopy(factors);
    args_.InitCopy(factor2node);

    bgraph_.Init(nodes_.size(), factor2node);

    for (index_t i_factor = 0; i_factor < bgraph_.n_factors(); i_factor++) {
      DEBUG_ASSERT(factors_[i_factor].ranges_.size() ==
		   bgraph_.n_factornodes(i_factor));
      for (index_t i_edge = 0; 
	   i_edge < bgraph_.n_factornodes(i_factor); i_edge++) {
	index_t i_node = bgraph_.node(i_factor, i_edge);
	DEBUG_ASSERT(factors_[i_factor].ranges_[i_edge] ==
		     nodes_[i_node].symbols_.size());
      }
    }

    // prepare spaces for messages between factor and node and vice versa
    for (index_t i_factor = 0; i_factor < bgraph_.n_factors(); i_factor++) {
      for (index_t i_edge = 0; 
	   i_edge < bgraph_.n_factornodes(i_factor); i_edge++)
	bgraph_.msg_factor2node(i_factor, i_edge).Init
	  (factors_[i_factor].GetRange(i_edge));
    }

    for (index_t i_node = 0; i_node < bgraph_.n_nodes(); i_node++) {
      for (index_t i_edge = 0; 
	   i_edge < bgraph_.n_nodefactors(i_node); i_edge++)
	bgraph_.msg_node2factor(i_node, i_edge).Init
	  (nodes_[i_node].GetRange());
    }
    
    // choose an order in which messages are passed inside the graph
    bgraph_.BreadthFirstSearchOrder();

    Z = 1; // initialize the normalization factor
  }

  /** Calculate factor value using arguments stored in args_ */
  double GetFactor(index_t i_factor) {
    double val = factors_[i_factor].GetVal(args_[i_factor]);
    //printf("    getval = %f\n", val);
    return val;
  }

  /** Set arguments */
  void SetArgs(index_t i_factor, index_t i_edge, index_t val) {
    args_[i_factor][i_edge] = val;
  }

  /** Set arguments from observed values */
  void SetArgsFromObserved(index_t i_factor) {
    for (index_t i_edge = 0; 
	 i_edge < bgraph_.n_factornodes(i_factor); i_edge++) {
      index_t i_node = bgraph_.node(i_factor, i_edge);
      SetArgs(i_factor, i_edge, observeds_[i_node]);
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
      observeds_[i_node] = nodes_[i_node].GetIndex(observed_syms[i_node]);
  }

  /** Calculate the joint product of all factors in the graph
   *  given a realization of all variables/nodes
   */
  double jointProduct() {
    double rval = 1;
    for (index_t i_factor = 0; i_factor < factors_.size(); i_factor++) {
      SetArgsFromObserved(i_factor);
      rval *= GetFactor(i_factor);
    }
    return rval;
  }

  /** Message passing from node to factor */
  void PassMessageNode2Factor(index_t i_node, index_t i_edge) {
    for (index_t val = 0; val < nodes_[i_node].GetRange(); val++) {
      double s = 1;
      for (index_t i = 0; i < bgraph_.n_nodefactors(i_node); i++)
	if (i != i_edge) {
	  index_t i_factor = bgraph_.factor(i_node, i);
	  index_t c_edge = bgraph_.factor_cedge(i_node, i);
	  s *= bgraph_.msg_factor2node(i_factor, c_edge)[val];
	}
      bgraph_.msg_node2factor(i_node, i_edge)[val] = s;
      //printf("  val = %d s = %f\n", val, s);
    }
  }

  /** Message passing from factor to node */
  void PassMessageFactor2Node(index_t i_factor, index_t i_edge) {
    index_t i_node = bgraph_.node(i_factor, i_edge);
    for (index_t val = 0; val < nodes_[i_node].GetRange(); val++) {
      args_[i_factor][i_edge] = val;
      double s = 0;
      VisitFactorArg(i_factor, i_edge, 0, 1, s);
      bgraph_.msg_factor2node(i_factor, i_edge)[val] = s;
      //printf("  val = %d s = %f\n", val, s);
    }
  }

  /** Recursive procedure to calculate the message from
   *  factor to node
   */
  void VisitFactorArg(index_t i_factor, index_t i_edge, index_t i, 
		      double term, double& sum) {
    if (i >= factors_[i_factor].n_args()) {
      sum += term*GetFactor(i_factor);
      return;
    }
    if (i_edge != i) {
      index_t i_node = bgraph_.node(i_factor, i);
      index_t c_edge = bgraph_.node_cedge(i_factor, i);
      for (index_t val = 0; val < nodes_[i_node].GetRange(); val++) {
	args_[i_factor][i] = val;
	VisitFactorArg(i_factor, i_edge, i+1, 
		       term*bgraph_.msg_node2factor(i_node, c_edge)[val], sum);
      }
    }
    else
      VisitFactorArg(i_factor, i_edge, i+1, term, sum);
  }

  /** Pass messages through entire graph using order */
  void PassMessages(bool reverse) {
    if (reverse) {
      for (index_t i = 0; i < bgraph_.n_orderedges(); i++) {
	index_t x = bgraph_.GetOrderEdgeFirst(i);
	index_t y = bgraph_.GetOrderEdgeSecond(i);
	//printf("x = %d y = %d\n", x, y);
	if (x < nodes_.size())
	  PassMessageNode2Factor(x, y);
	else 
	  PassMessageFactor2Node(x-nodes_.size(), y);
      }
    }
    else {
      for (index_t i = bgraph_.n_orderedges()-1; i >= 0; i--) {
	index_t x = bgraph_.GetOrderEdgeFirst(i);
	index_t y = bgraph_.GetOrderEdgeSecond(i);
	//printf("x = %d y = %d\n", x, y);
	if (x < nodes_.size()) {
	  index_t i_factor = bgraph_.factor(x, y);
	  index_t c_edge = bgraph_.factor_cedge(x, y);
	  //printf("i_factor = %d c_edge = %d\n", i_factor, c_edge);
	  PassMessageFactor2Node(i_factor, c_edge);
	}
	else {
	  index_t i_node = bgraph_.node(x-bgraph_.n_nodes(), y);
	  index_t c_edge = bgraph_.node_cedge(x-bgraph_.n_nodes(), y);
	  //printf("i_node = %d c_edge = %d\n", i_node, c_edge);
	  PassMessageNode2Factor(i_node, c_edge);
	}
      }
    }
  }

  /** Calculate marginal sum at each node
   *  assuming that messages are passed through the graph
   *  on every edge in both direction
   *  return constant Z
   */

  double NodeMarginalSum(index_t i_node, ArrayList<double>* sum) {
    Z = 0;
    sum->Init(nodes_[i_node].GetRange());
    for (index_t val = 0; val < nodes_[i_node].GetRange(); val++) {
      double s = 1;
      for (index_t i_edge = 0; 
	   i_edge < bgraph_.n_nodefactors(i_node); i_edge++){
	index_t i_factor = bgraph_.factor(i_node, i_edge);
	index_t c_edge =bgraph_.factor_cedge(i_node, i_edge);
	s *= bgraph_.msg_factor2node(i_factor, c_edge)[val];
      }
      (*sum)[val] = s;
      Z += s;
    }
    return Z;
  } 
  
};

#endif
