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

  /** Connection between nodes and factors */
  ArrayList<ArrayList<index_t> > node2factor_;

  /** Connection between factors and nodes for convenient */
  ArrayList<ArrayList<index_t> > factor2node_;


  /** Define an edge as a pair of intergers
   *  (x,edge) = (node -> factor) if x < nodes_.size()
   *  (x,edge) = (factor -> node) if x >= nodes_.size()
   *         
   */
  typedef std::pair<index_t, index_t> Edge;

  /** Edge order in which message are passed in first step,
   *  the edge order for second step is in reverse
   */
  ArrayList<Edge> order_;

  /** Message from node to factor */
  ArrayList<ArrayList<ArrayList<double> > > msg_node2factor_;

  /** Message from factor to node */
  ArrayList<ArrayList<ArrayList<double> > > msg_factor2node_;

  /** The Z constant - common denominator - normalization constant */
  double Z;

  OT_DEF(FactorGraph) {
    OT_MY_OBJECT(nodes_);
    OT_MY_OBJECT(observeds_);
    OT_MY_OBJECT(factors_);
    OT_MY_OBJECT(args_);
    OT_MY_OBJECT(node2factor_);
    OT_MY_OBJECT(factor2node_);
    OT_MY_OBJECT(order_);
    OT_MY_OBJECT(msg_node2factor_);
    OT_MY_OBJECT(msg_factor2node_);
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

    factor2node_.InitCopy(factor2node);

    // initialize connection from node to factor
    // based on connection from factor to node
    node2factor_.Init(nodes_.size());
    for (index_t i_node = 0; i_node < node2factor_.size(); i_node++)
      node2factor_[i_node].Init();

    index_t n_edge = 0;
    for (index_t i_factor = 0; i_factor < factor2node_.size(); i_factor++) {
      DEBUG_ASSERT(factors_[i_factor].ranges_.size() ==
		   factor2node_[i_factor].size());
      for (index_t i_edge = 0; i_edge < factor2node_[i_factor].size(); i_edge++) {
	index_t i_node = factor2node[i_factor][i_edge];
	DEBUG_ASSERT(factors_[i_factor].ranges_[i_edge] ==
		     nodes_[i_node].symbols_.size());
	n_edge++;
	DEBUG_ASSERT(i_node < nodes_.size());
	node2factor_[i_node].PushBackCopy(i_factor);
      }
    }

    // prepare spaces for messages between factor and node and vice versa
    msg_factor2node_.Init(factor2node_.size());
    for (index_t i_factor = 0; i_factor < factor2node_.size(); i_factor++) {
      msg_factor2node_[i_factor].Init(factor2node[i_factor].size());
      for (index_t i_edge = 0; 
	   i_edge < factor2node_[i_factor].size(); i_edge++)
	msg_factor2node_[i_factor][i_edge].Init
	  (factors_[i_factor].ranges_[i_edge]);
    }

    msg_node2factor_.Init(node2factor_.size());
    for (index_t i_node = 0; i_node < node2factor_.size(); i_node++) {
      msg_node2factor_[i_node].Init(node2factor_[i_node].size());
      for (index_t i_edge = 0; 
	   i_edge < node2factor_[i_node].size(); i_edge++)
	msg_node2factor_[i_node][i_edge].Init
	  (nodes_[i_node].symbols_.size());
    }
    
    // choose an order in which messages are passed inside the graph
    order_.Init(n_edge);
    CalculateOrder();

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
    index_t n = 0;

    while (!q_visit.empty()) {
      index_t x = q_visit.front(); q_visit.pop();
      if (x < nodes_.size()) { // (node --> factor)
	for (index_t i_edge = 0; i_edge < node2factor_[x].size(); i_edge++) {
	  index_t y = node2factor_[x][i_edge]+nodes_.size();
	  if (not_visited[y]) {
	    order_[n++] = Edge(x, i_edge);
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
	    order_[n++] = Edge(x, i_edge);
	    q_visit.push(y);
	    not_visited[y] = false;
	  }
	}
      }
    }
    DEBUG_ASSERT(n == order_.size()); // for a tree graph
    //ot::Print(order_, "order", stdout);
  }

  /** Find a node in a factor */
  index_t FindNode(index_t i_factor, index_t i_node) {
    for (index_t i_edge = 0; i_edge < factor2node_[i_factor].size(); i_edge++)
      if (factor2node_[i_factor][i_edge] == i_node) return i_edge;
    return -1;
  }

  /** Find a factor of a node */
  index_t FindFactor(index_t i_node, index_t i_factor) {
    for (index_t i_edge = 0; i_edge < node2factor_[i_node].size(); i_edge++)
      if (node2factor_[i_node][i_edge] == i_factor) return i_edge;
    return -1;
  }

  /** Message passing from node to factor */
  void PassMessageNode2Factor(index_t i_node, index_t i_edge) {
    for (index_t val = 0; val < nodes_[i_node].GetRange(); val++) {
      double s = 1;
      for (index_t i = 0; i < node2factor_[i_node].size(); i++)
	if (i != i_edge) {
	  index_t i_factor = node2factor_[i_node][i];
	  index_t c_edge = FindNode(i_factor, i_node);
	  s *= msg_factor2node_[i_factor][c_edge][val];
	}
      msg_node2factor_[i_node][i_edge][val] = s;
      //printf("  val = %d s = %f\n", val, s);
    }
  }

  /** Message passing from factor to node */
  void PassMessageFactor2Node(index_t i_factor, index_t i_edge) {
    index_t i_node = factor2node_[i_factor][i_edge];
    for (index_t val = 0; val < nodes_[i_node].GetRange(); val++) {
      args_[i_factor][i_edge] = val;
      double s = 0;
      VisitFactorArg(i_factor, i_edge, 0, 1, s);
      msg_factor2node_[i_factor][i_edge][val] = s;
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
      index_t i_node = factor2node_[i_factor][i];
      index_t c_edge = FindFactor(i_node, i_factor);
      for (index_t val = 0; val < nodes_[i_node].GetRange(); val++) {
	args_[i_factor][i] = val;
	VisitFactorArg(i_factor, i_edge, i+1, 
		       term*msg_node2factor_[i_node][c_edge][val], sum);
      }
    }
    else
      VisitFactorArg(i_factor, i_edge, i+1, term, sum);
  }

  /** Pass messages through entire graph using order */
  void PassMessages(bool reverse) {
    if (reverse) {
      for (index_t i = 0; i < order_.size(); i++) {
	index_t x = order_[i].first;
	index_t y = order_[i].second;
	//printf("x = %d y = %d\n", x, y);
	if (x < nodes_.size())
	  PassMessageNode2Factor(x, y);
	else 
	  PassMessageFactor2Node(x-nodes_.size(), y);
      }
    }
    else {
      for (index_t i = order_.size()-1; i >= 0; i--) {
	index_t x = order_[i].first;
	index_t y = order_[i].second;
	//printf("x = %d y = %d\n", x, y);
	if (x < nodes_.size()) {
	  index_t i_factor = node2factor_[x][y];
	  index_t c_edge = FindNode(i_factor, x);
	  //printf("i_factor = %d c_edge = %d\n", i_factor, c_edge);
	  PassMessageFactor2Node(i_factor, c_edge);
	}
	else {
	  index_t i_node = factor2node_[x-nodes_.size()][y];
	  index_t c_edge = FindFactor(i_node, x-nodes_.size());
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
      for (index_t i_edge = 0; i_edge < node2factor_[i_node].size(); i_edge++){
	index_t i_factor = node2factor_[i_node][i_edge];
	index_t c_edge = FindNode(i_factor, i_node);
	s *= msg_factor2node_[i_factor][c_edge][val];
      }
      (*sum)[val] = s;
      Z += s;
    }
    return Z;
  } 
  
};

#endif
