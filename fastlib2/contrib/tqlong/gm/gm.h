/**
 * @file gm.h
 *
 * This file contains definition of the sum-product algorithm on a 
 * discrete-variable factor graph
 * 
 */

#ifndef GM_SUM_PRODUCT_ALGORITHM_DISCRETE_FACTOR_GRAPH_H
#define GM_SUM_PRODUCT_ALGORITHM_DISCRETE_FACTOR_GRAPH_H

#include <fastlib/fastlib.h>
#include "bipartie.h"
#include "factor_graph.h"

class Node {
 public:
  typedef index_t RangeType;
  typedef String SymbolType;

 private:
  /** The symbols for each values of the variable represented by the node */
  ArrayList<SymbolType> symbols_;

  OT_DEF(Node) {
    OT_MY_OBJECT(symbols_);
  }

 public:
  /** Initialize the set of symbols */
  void Init(const SymbolType* symbols, index_t n_syms) {
    symbols_.InitCopy(symbols, n_syms);
  }

  /** Get a symbol string */
  SymbolType GetSymbol(index_t i) {
    DEBUG_ASSERT(i < symbols_.size());
    return symbols_[i];
  }

  /** Set a symbol string */
  void SetSymbol(index_t i, const SymbolType& sym) {
    DEBUG_ASSERT(i < symbols_.size());    
    symbols_[i] = sym;
  }

  /** Get the index of a symbol 
   *  return -1 if not found
   */
  RangeType GetVal(const SymbolType& sym) {
    for (int i = 0; i < symbols_.size(); i++)
      if (symbols_[i] == sym) return i;
    return -1; // not found;
  }

  RangeType GetRange() {
    return symbols_.size();
  }
};

class Factor {
 public:
  typedef index_t RangeType;

 private:
  /** The ranges of each variable of the factor 
   *  (length of symbols_ in each node)
   */
  ArrayList<RangeType> ranges_;

  /** Probabilities/values table (multi-dimentional) */
  Vector vals_;

  /** Temporary storage of factors' arguments */
  ArrayList<RangeType> args_;

  OT_DEF(Factor) {
    OT_MY_OBJECT(ranges_);
    OT_MY_OBJECT(vals_);
    OT_MY_OBJECT(args_);
  }

 public:
  /** Initialize the factor with ranges specified */
  void Init(const ArrayList<index_t>& ranges) {
    index_t len = 1;

    for (int i = 0; i < ranges.size(); i++)
      len *= ranges[i];

    ranges_.InitCopy(ranges);
    vals_.Init(len);
    args_.Init(ranges_.size());
  }

  /** Initialize the factor with ranges and probabilities/values specified */
  void Init(const ArrayList<RangeType>& ranges, const Vector& vals) {
    index_t len = 1;

    for (int i = 0; i < ranges.size(); i++)
      len *= ranges[i];

    DEBUG_ASSERT(len == vals.length());

    ranges_.InitCopy(ranges);
    vals_.Copy(vals);
    args_.Init(ranges_.size());
  }

  /** Initialize the factor with ranges and probabilities/values specified */
  void Init(const index_t* ranges, index_t n_nodes, const double* vals) {
    index_t len = 1;

    for (int i = 0; i < n_nodes; i++)
      len *= ranges[i];

    ranges_.InitCopy(ranges, n_nodes);
    vals_.Copy(vals, len);
    args_.Init(ranges_.size());
  }

  RangeType n_args() const {
    return ranges_.size();
  }

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
  double GetVal(const ArrayList<RangeType>& args) {
    index_t rpos = GetIndex(args);
    return vals_[rpos];
  }

  double GetVal() {
    return GetVal(args_);
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

  RangeType GetRange(index_t i_arg) {
    DEBUG_ASSERT(i_arg < n_args());
    return ranges_[i_arg];
  }

  RangeType GetArg(index_t i_arg) {
    DEBUG_ASSERT(i_arg < n_args());
    return args_[i_arg];
  }

  void SetArg(index_t i_arg, RangeType val) {
    DEBUG_ASSERT(i_arg < n_args());
    args_[i_arg] = val;
  }
};

class SumProductPassingAlgorithm {
 public:
  typedef ArrayList<double> MessageNode2FactorType;
  typedef ArrayList<double> MessageFactor2NodeType;
  typedef FactorGraph<Node, Factor, 
    MessageNode2FactorType, MessageFactor2NodeType> FactorGraphType;
  typedef FactorGraphType::BGraphType BGraphType;
 private:
  ArrayList<BGraphType::Edge> order_;
 public:
  void Init() {
    order_.Init();
  }

  void InitMessages(FactorGraphType& fg) {
    BGraphType& bgraph = fg.GetBGraph();

    // prepare spaces for messages between factor and node
    for (index_t i_factor = 0; i_factor < bgraph.n_factors(); i_factor++)
      for (index_t i_edge = 0; i_edge < bgraph.n_factornodes(i_factor); i_edge++)
	bgraph.msg_factor2node(i_factor, i_edge).Init
	  (fg.GetFactor(i_factor).GetRange(i_edge));

    for (index_t i_node = 0; i_node < bgraph.n_nodes(); i_node++) {
      for (index_t i_edge = 0; i_edge < bgraph.n_nodefactors(i_node); i_edge++)
	bgraph.msg_node2factor(i_node, i_edge).Init
	  (fg.GetNode(i_node).GetRange());
    }    
  }

  void CreateOrder(FactorGraphType& fg) {
    order_.Renew();
    fg.GetBGraph().CreateBFSOrder(order_, 0); // BFS from the first node
  }  

  inline index_t GetOrderEdgeFirst(index_t i_order) {
    DEBUG_ASSERT(i_order < n_orderedges());
    return order_[i_order].first;
  }

  inline index_t GetOrderEdgeSecond(index_t i_order) {
    DEBUG_ASSERT(i_order < n_orderedges());
    return order_[i_order].second;
  }

  inline index_t n_orderedges() {
    return order_.size();
  }

  /** Message passing from node to factor */
  void PassMessageNode2Factor(FactorGraphType& fg, index_t i_node, index_t i_edge) {
    BGraphType& bgraph = fg.GetBGraph();
    for (index_t val = 0; val < fg.GetNode(i_node).GetRange(); val++) {
      double s = 1;
      for (index_t i = 0; i < bgraph.n_nodefactors(i_node); i++)
	if (i != i_edge) {
	  index_t i_factor = bgraph.factor(i_node, i);
	  index_t c_edge = bgraph.factor_cedge(i_node, i);
	  s *= bgraph.msg_factor2node(i_factor, c_edge)[val];
	}
      bgraph.msg_node2factor(i_node, i_edge)[val] = s;
      //printf("  val = %d s = %f\n", val, s);
    }
  }

  /** Message passing from factor to node */
  void PassMessageFactor2Node(FactorGraphType& fg, index_t i_factor, index_t i_edge) {
    BGraphType& bgraph = fg.GetBGraph();
    index_t i_node = bgraph.node(i_factor, i_edge);
    for (index_t val = 0; val < fg.GetNode(i_node).GetRange(); val++) {
      fg.GetFactor(i_factor).SetArg(i_edge, val);
      double s = 0;
      VisitFactorArg(fg, i_factor, i_edge, 0, 1, s);
      bgraph.msg_factor2node(i_factor, i_edge)[val] = s;
      //printf("  val = %d s = %f\n", val, s);
    }
  }

  /** Recursive procedure to calculate the message from
   *  factor to node
   */
  void VisitFactorArg(FactorGraphType& fg, index_t i_factor, index_t i_edge, 
		      index_t i, double term, double& sum) {
    BGraphType& bgraph = fg.GetBGraph();
    if (i >= fg.GetFactor(i_factor).n_args()) {
      sum += term*fg.GetFactorVal(i_factor);
      return;
    }
    if (i_edge != i) {
      index_t i_node = bgraph.node(i_factor, i);
      index_t c_edge = bgraph.node_cedge(i_factor, i);
      for (index_t val = 0; val < fg.GetNode(i_node).GetRange(); val++) {
	fg.GetFactor(i_factor).SetArg(i, val);
	VisitFactorArg(fg, i_factor, i_edge, i+1, 
		       term*bgraph.msg_node2factor(i_node, c_edge)[val], sum);
      }
    }
    else
      VisitFactorArg(fg, i_factor, i_edge, i+1, term, sum);
  }

  /** Pass messages through entire graph using order */
  void PassMessages(FactorGraphType& fg, bool reverse) {
    BGraphType& bgraph = fg.GetBGraph();

    if (reverse) {
      for (index_t i = 0; i < n_orderedges(); i++) {
	index_t x = GetOrderEdgeFirst(i);
	index_t y = GetOrderEdgeSecond(i);
	//printf("x = %d y = %d\n", x, y);
	if (x < bgraph.n_nodes())
	  PassMessageNode2Factor(fg, x, y);
	else 
	  PassMessageFactor2Node(fg, x-bgraph.n_nodes(), y);
      }
    }
    else {
      for (index_t i = n_orderedges()-1; i >= 0; i--) {
	index_t x = GetOrderEdgeFirst(i);
	index_t y = GetOrderEdgeSecond(i);
	//printf("x = %d y = %d\n", x, y);
	if (x < bgraph.n_nodes()) {
	  index_t i_factor = bgraph.factor(x, y);
	  index_t c_edge = bgraph.factor_cedge(x, y);
	  //printf("i_factor = %d c_edge = %d\n", i_factor, c_edge);
	  PassMessageFactor2Node(fg, i_factor, c_edge);
	}
	else {
	  index_t i_node = bgraph.node(x-bgraph.n_nodes(), y);
	  index_t c_edge = bgraph.node_cedge(x-bgraph.n_nodes(), y);
	  //printf("i_node = %d c_edge = %d\n", i_node, c_edge);
	  PassMessageNode2Factor(fg, i_node, c_edge);
	}
      }
    }
  }

  /** Calculate marginal sum at each node
   *  assuming that messages are passed through the graph
   *  on every edge in both direction
   *  return constant Z
   */

  double NodeMarginalSum(FactorGraphType& fg, index_t i_node, ArrayList<double>* sum){
    BGraphType& bgraph = fg.GetBGraph();
    double Z = 0;
    sum->Init(fg.GetNode(i_node).GetRange());
    for (index_t val = 0; val < fg.GetNode(i_node).GetRange(); val++) {
      double s = 1;
      for (index_t i_edge = 0; 
	   i_edge < bgraph.n_nodefactors(i_node); i_edge++){
	index_t i_factor = bgraph.factor(i_node, i_edge);
	index_t c_edge = bgraph.factor_cedge(i_node, i_edge);
	s *= bgraph.msg_factor2node(i_factor, c_edge)[val];
      }
      (*sum)[val] = s;
      Z += s;
    }
    fg.SetZ(Z);
    return Z;
  } 

};

#endif
