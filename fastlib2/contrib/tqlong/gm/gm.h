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
  void Init(const ArrayList<RangeType>& ranges);

  /** Initialize the factor with ranges and probabilities/values specified */
  void Init(const ArrayList<RangeType>& ranges, const Vector& vals);

  /** Initialize the factor with ranges and probabilities/values specified */
  void Init(const index_t* ranges, index_t n_nodes, const double* vals);

  /** Get the number of arguments */
  RangeType n_args() const {
    return ranges_.size();
  }

  /** Calculate index from position which is stored 
   *  in column-wise manner
   */
  index_t GetIndex(const ArrayList<index_t>& pos);

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

  /** Get the range of an argument which will be compared to the 
   *  corresponding node's range
   */
  RangeType GetRange(index_t i_arg) {
    DEBUG_ASSERT(i_arg < n_args());
    return ranges_[i_arg];
  }

  /** Get an argument */
  RangeType GetArg(index_t i_arg) {
    DEBUG_ASSERT(i_arg < n_args());
    return args_[i_arg];
  }

  /** Set an argument */
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
  /** Any message passing algorithm should maintain an order
   *  in which the messages are passed throughout the graph
   */
  ArrayList<BGraphType::Edge> order_;

 private:
  /** Get the first end of an edge in the list */
  inline index_t GetOrderEdgeFirst(index_t i_order) {
    DEBUG_ASSERT(i_order < n_orderedges());
    return order_[i_order].first;
  }

  /** Get the second end of an edge in the list */
  inline index_t GetOrderEdgeSecond(index_t i_order) {
    DEBUG_ASSERT(i_order < n_orderedges());
    return order_[i_order].second;
  }

  /** Get the number of edges in the list */
  inline index_t n_orderedges() {
    return order_.size();
  }

  /** Message passing from node to factor */
  void PassMessageNode2Factor(FactorGraphType& fg, index_t i_node, index_t i_edge);

  /** Message passing from factor to node */
  void PassMessageFactor2Node(FactorGraphType& fg, index_t i_factor, index_t i_edge);

  /** Recursive procedure to calculate the message from
   *  factor to node
   */
  void VisitFactorArg(FactorGraphType& fg, index_t i_factor, index_t i_edge, 
		      index_t i, double term, double& sum);

 public:
  void Init() {
    order_.Init();
  }

  /** Prepare the space for the messages */
  void InitMessages(FactorGraphType& fg);

  /** Use the graph helper function to create BFS edge order */
  void CreateOrder(FactorGraphType& fg) {
    order_.Renew();
    fg.GetBGraph().CreateBFSOrder(order_, 0); // BFS from the first node
  }  

  /** Pass messages through entire graph using order */
  void PassMessages(FactorGraphType& fg, bool reverse);

  /** Calculate marginal sum at each node
   *  assuming that messages are already passed through the graph
   *  on every edge in both direction
   *  return constant Z
   */

  double NodeMarginalSum(FactorGraphType& fg, index_t i_node, ArrayList<double>* sum);
};

#endif
