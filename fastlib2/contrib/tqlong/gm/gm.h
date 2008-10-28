/**
 * @file gm.h
 *
 * This file contains definition of a factor graph
 * 
 */

#ifndef GM_FACTOR_GRAPH_H
#define GM_FACTOR_GRAPH_H

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

  /** Connection between nodes and factors */
  ArrayList<ArrayList<index_t> > node2factor_;

  /** Connection between factors and nodes for convenient */
  ArrayList<ArrayList<index_t> > factor2node_;


  /** Define an edge as a pair of intergers
   *  (x,y) = (node -> factor) if x < nodes_.size()
              and factor index = y - nodes_size()
   *  (x,y) = (factor -> node) if x >= nodes_.size()
   *          and factor index = x - nodes_size()
   */
  typedef std::pair<index_t, index_t> Edge;

  /** Edge order in which message is passed in first step,
   *  the edge order for second step is in reverse
   */
  ArrayList<Edge> order_;

  OT_DEF(FactorGraph) {
    OT_MY_OBJECT(nodes_);
    OT_MY_OBJECT(observeds_);
    OT_MY_OBJECT(factors_);
    OT_MY_OBJECT(args_);
    OT_MY_OBJECT(node2factor_);
    OT_MY_OBJECT(factor2node_);
    OT_MY_OBJECT(order_);
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

    factor2node_.InitCopy(factor2node);

    node2factor_.Init(nodes_.size());
    for (index_t i_node = 0; i_node < node2factor_.size(); i_node++)
      node2factor_[i_node].Init();

    index_t n_edge = 0;
    for (index_t i_factor = 0; i_factor < factor2node_.size(); i_factor++) {
      for (index_t i_edge = 0; i_edge < factor2node_[i_factor].size(); i_edge++) {
	index_t i_node = factor2node[i_factor][i_edge];
	n_edge++;
	DEBUG_ASSERT(i_node < nodes_.size());
	node2factor_[i_node].PushBackCopy(i_factor);
      }
    }
    order_.Init(n_edge);
    CalculateOrder();
  }

  /** Calculate factor value using arguments stored in args_ */
  double GetFactor(index_t i_factor) {
    return factors_[i_factor].GetVal(args_[i_factor]);
  }

  /** Set arguments */
  void SetArgs(index_t i_factor, index_t i_edge, index_t val) {
    args_[i_factor][i_edge] = val;
  }

  /** Set arguments from observed values */
  void SetArgsFromObserved(index_t i_factor) {
    for (index_t i_edge = 0; i_edge < factor2node_[i_factor].size(); i_edge++) {
      index_t i_node = factor2node_[i_factor][i_edge];
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

  /** Calculate edge order for message passing */
  void CalculateOrder() {
    // a breath first search procedure to find order of edges
    ArrayList<bool> not_visited;

    not_visited.InitRepeat(true, nodes_.size()+factors_.size());

    std::queue<index_t> q_visit;
    q_visit.push(0);        // visit the first node as root
    not_visited[0] = false; // and make it visited
    index_t n = order_.size()-1;

    while (!q_visit.empty()) {
      index_t x = q_visit.front(); q_visit.pop();
      if (x < nodes_.size()) { // (node --> factor)
	for (index_t i_edge = 0; i_edge < node2factor_[x].size(); i_edge++) {
	  index_t y = node2factor_[x][i_edge]+nodes_.size();
	  if (not_visited[y]) {
	    order_[n--] = Edge(x, y);
	    q_visit.push(y);
	    not_visited[y] = false;
	  }
	}
      }
      else { // (factor --> node)
	index_t xx = x-nodes_.size();
	for (index_t i_edge = 0; i_edge < factor2node_[xx].size(); i_edge++) {
	  index_t y = factor2node_[xx][i_edge];
	  if (not_visited[y]) {
	    order_[n--] = Edge(x, y);
	    q_visit.push(y);
	    not_visited[y] = false;
	  }
	}
      }
    }
    DEBUG_ASSERT(n == -1); // for a tree graph
    //ot::Print(order_, "order", stdout);
  }
};

#endif
