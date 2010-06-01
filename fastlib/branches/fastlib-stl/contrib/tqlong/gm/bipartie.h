/**
 * @file bipartie.h
 *
 * This file contains definition of a bipartie graph
 * 
 */

#ifndef GM_BIPARTIE_GRAPH_H
#define GM_BIPARTIE_GRAPH_H

#include <fastlib/fastlib.h>
#include <queue>

template <class MsgNode2Factor, class MsgFactor2Node>
class BipartieGraph {
 public:
  /** Define the message type */
  typedef MsgNode2Factor MessageNode2FactorType;
  typedef MsgFactor2Node MessageFactor2NodeType;

  /** Define an edge as a pair of intergers
   *  (x,edge) = (node -> factor) if x < nodes_.size()
   *  (x,edge) = (factor -> node) if x >= nodes_.size()
   */
  typedef std::pair<index_t, index_t> Edge;

 private:
  /** Number of edges */
  index_t n_edges_;

  /** Connection between nodes and factors */
  ArrayList<ArrayList<index_t> > node2factor_;
  ArrayList<ArrayList<index_t> > node2factor_cedge_;

  /** Connection between factors and nodes for convenient */
  ArrayList<ArrayList<index_t> > factor2node_;
  ArrayList<ArrayList<index_t> > factor2node_cedge_;
  
  /** Message from node to factor */
  ArrayList<ArrayList<MessageNode2FactorType> > msg_node2factor_;

  /** Message from factor to node */
  ArrayList<ArrayList<MessageFactor2NodeType> > msg_factor2node_;

  OT_DEF(BipartieGraph) {
    OT_MY_OBJECT(n_edges_);
    OT_MY_OBJECT(node2factor_);
    OT_MY_OBJECT(node2factor_cedge_);
    OT_MY_OBJECT(factor2node_);
    OT_MY_OBJECT(factor2node_cedge_);
    OT_MY_OBJECT(msg_node2factor_);
    OT_MY_OBJECT(msg_factor2node_);
  }
  
 public:
  void Init(index_t n_nodes_, 
	    const ArrayList<ArrayList<index_t> >& factor2node) {
    factor2node_.InitCopy(factor2node);
    factor2node_cedge_.InitCopy(factor2node);

    // initialize connection from node to factor
    // based on connection from factor to node
    node2factor_.Init(n_nodes_);
    node2factor_cedge_.Init(n_nodes_);

    for (index_t i_node = 0; i_node < n_nodes(); i_node++) {
      node2factor_[i_node].Init();
      node2factor_cedge_[i_node].Init();
    }

    n_edges_ = 0;
    for (index_t i_factor = 0; i_factor < factor2node_.size(); i_factor++) {
      for (index_t i_edge = 0; 
	   i_edge < n_factornodes(i_factor); i_edge++) {
	index_t i_node = factor2node_[i_factor][i_edge];
	n_edges_++;
	DEBUG_ASSERT(i_node < n_nodes());
	factor2node_cedge_[i_factor][i_edge] = n_nodefactors(i_node);
	node2factor_[i_node].PushBackCopy(i_factor);
	node2factor_cedge_[i_node].PushBackCopy(i_edge);
      }
    }

    msg_node2factor_.Init(n_nodes());
    for (index_t i_node = 0; i_node < n_nodes(); i_node++)
      msg_node2factor_[i_node].Init(n_nodefactors(i_node));

    msg_factor2node_.Init(n_factors());
    for (index_t i_factor = 0; i_factor < n_factors(); i_factor++)
      msg_factor2node_[i_factor].Init(n_factornodes(i_factor));
  }

  inline index_t n_nodes() {
    return node2factor_.size();
  }

  inline index_t n_nodefactors(index_t i_node) {
    return node2factor_[i_node].size();
  }

  inline index_t n_factors() {
    return factor2node_.size();
  }

  inline index_t n_factornodes(index_t i_factor) {
    return factor2node_[i_factor].size();
  }

  inline index_t n_edges() {
    return n_edges_;
  }

  inline index_t factor(index_t i_node, index_t i_edge) {
    return node2factor_[i_node][i_edge];
  }

  inline index_t factor_cedge(index_t i_node, index_t i_edge) {
    return node2factor_cedge_[i_node][i_edge];
  }

  inline index_t node(index_t i_factor, index_t i_edge) {
    return factor2node_[i_factor][i_edge];
  }

  inline index_t node_cedge(index_t i_factor, index_t i_edge) {
    return factor2node_cedge_[i_factor][i_edge];
  }

  MessageNode2FactorType& msg_node2factor(index_t i_node, index_t i_edge) {
    return msg_node2factor_[i_node][i_edge];
  }

  MessageFactor2NodeType& msg_factor2node(index_t i_factor, index_t i_edge) {
    return msg_factor2node_[i_factor][i_edge];
  }

  /**
   *  A breath first search procedure to find order of edges
   */

  void CreateBFSOrder(ArrayList<Edge>& order, index_t root) {
    ArrayList<bool> not_visited;

    not_visited.InitRepeat(true, n_nodes()+n_factors());

    std::queue<index_t> q_visit;
    q_visit.push(root);        // visit the first node as root
    not_visited[root] = false; // and make it visited
    index_t n = 0;

    order.Init(n_edges_);

    while (!q_visit.empty()) {
      index_t x = q_visit.front(); q_visit.pop();
      if (x < n_nodes()) { // (node --> factor)
	for (index_t i_edge = 0; i_edge < n_nodefactors(x); i_edge++) {
	  index_t y = node2factor_[x][i_edge]+n_nodes();
	  if (not_visited[y]) {
	    order[n++] = Edge(x, i_edge);
	    q_visit.push(y);
	    not_visited[y] = false;
	  }
	}
      }
      else { // (factor --> node)
	index_t xx = x-n_nodes();
	for (index_t i_edge = 0; i_edge < n_factornodes(xx); i_edge++) {
	  index_t y = factor2node_[xx][i_edge];
	  if (not_visited[y]) {
	    order[n++] = Edge(x, i_edge);
	    q_visit.push(y);
	    not_visited[y] = false;
	  }
	}
      }
    }
    DEBUG_ASSERT(n == n_edges()); // for a tree graph
    //ot::Print(order, "order", stdout);
  }

};

#endif
