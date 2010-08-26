#ifndef __MESSAGE_PENDING_INFERENCE_H
#define __MESSAGE_PENDING_INFERENCE_H

#include "gm.h"

BEGIN_GRAPHICAL_MODEL_NAMESPACE;

/** The sum-product algorithm with message priority queue:
  * + Messages are picked to update according to theirs changes in previous iteration.
  * + A changed message will make other related messages pending to update
  * + Messages are propagated along all edges for a certain number of iterations
  *   while the total change of messages is bigger than change tolerance.
  * + Messages are used to calculate beliefs of variable-vertex and average of factor-vertex.
  */
template <typename _F>
    class MessagePendingInference : public MessagePriorityInference<_F>
{
public:
  typedef MessagePriorityInference<_F>                       _Base;
  typedef typename _Base::factor_type                        factor_type;         // map from Assignment --> factor_value_type
  typedef typename _Base::factor_value_type                  factor_value_type;   // a numeric type
  typedef typename _Base::graph_type                         graph_type;          // FactorGraph<factor_type>
  typedef typename _Base::vertex_type                        vertex_type;         // Vertex* (see factor_graph.h)
  typedef typename _Base::vertex_vector_type                 vertex_vector_type;  // Vector<vertex_type>
  typedef typename _Base::belief_type                        belief_type;         // Map<Value, factor_value_type>
  typedef typename _Base::belief_map_type                    belief_map_type;     // Map<vertex_type, belief_type>
  typedef typename _Base::message_type                       message_type;        // Map<Value, factor_value_type>
  typedef typename _Base::messages_map_type                  messages_map_type;   // Map<vertex_type, Map<vertex_type, message_type> >
                                                                                  // usage: message[u][v] where u,v are vertex_type
public:
  /** Preparing inference on a graph */
  MessagePendingInference(const graph_type& graph, ConvergenceMeasure cvm = ConvergenceMeasure())
    : _Base(graph, cvm) {}

  /** The inference algorithm */
  void run();
protected:
  typedef typename _Base::msg_double_queue_type              msg_double_queue_type;
  typedef typename _Base::msg_double_type                    msg_double_type;
  typedef typename _Base::vertex_pair_type                   vertex_pair_type;
};

template <typename _F> void MessagePendingInference<_F>::run()
{
  cout << "----------------------- Message Pending Sum-Product Inference -----------------" << endl;
  this->initBeliefs();
  this->initMessages();
  cout << "---------------------- iter = " << (this->curCvm_.iter_=0) << " Initializiing ----------------" << endl;
  this->initMessageQueue();

  msg_double_queue_type message_queue_next_round_;
  msg_double_queue_type &current_round = this->message_queue_, &next_round = message_queue_next_round_;
  for (this->curCvm_.iter_ = 1, this->curCvm_.changeTolerance_ = this->cvm_.changeTolerance_+1; ; this->curCvm_.iter_++)
  {
    cout << "---------------------- iter = " << this->curCvm_.iter_ << " ------------------------------" << endl;
    this->change_sum = factor_value_type(0.0);
    unsigned int n_msg = current_round.size();
    Set<vertex_pair_type> edge_added;
    while (!current_round.empty())
    {
      // get the most changed (u, v) from the priority queue
      msg_double_type top = current_round.top(); current_round.pop();
      vertex_type u = top.first.first, v = top.first.second;

      message_type oldMsg;
      updateMessage(u, v, &oldMsg);

      factor_value_type new_difference = difference(oldMsg, this->messages_[u][v]);
      this->change_sum += new_difference;

      if ( new_difference < factor_value_type(1e-16) ) continue;
      // push (v,t) to next round queue for all t != u
      BOOST_FOREACH(const vertex_type& t, this->graph_.neighbors(v))
      {
        if (t != u && !edge_added.contains(vertex_pair_type(v, t)))
        {
          next_round << msg_double_type( vertex_pair_type(v, t), factor_value_type(1e20) );
          edge_added << vertex_pair_type(v, t);
        }
      }

      if (!edge_added.contains(vertex_pair_type(u, v)))
      {
        // push (u,v) to next round queue with difference of old and new message as priority
        next_round << msg_double_type( vertex_pair_type(u, v), new_difference);
        edge_added << vertex_pair_type(u, v);
      }
//      cout << "(" << this->graph_.toString(u) << ", " << this->graph_.toString(v)
//           << ") " << top.second << " --> " << new_difference << endl;
    }
    cout << "update n_msg = " << n_msg << " total change = " << this->change_sum << endl;
    this->curCvm_.changeTolerance_ = toDouble(this->change_sum);
    if (this->terminateCondition()) break;
    // if termination condition is not met, exchange current_round and next_round
    msg_double_queue_type &tmp = current_round; current_round = next_round; next_round = tmp;
  }
  this->calculateBeliefs();
  this->normalizeBeliefs();
}

END_GRAPHICAL_MODEL_NAMESPACE;

#endif
