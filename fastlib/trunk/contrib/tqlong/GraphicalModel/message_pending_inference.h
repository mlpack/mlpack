#ifndef __MESSAGE_PENDING_INFERENCE_H
#define __MESSAGE_PENDING_INFERENCE_H

#include "gm.h"

BEGIN_GRAPHICAL_MODEL_NAMESPACE;

/** The sum-product algorithm with message priority queue:
  * + Messages are picked to update according to theirs changes in previous iteration.
  * + Messages are propagated along all edges for a certain number of iterations
  *   while the total change of messages is bigger than change tolerance.
  * + Messages are used to calculate beliefs of variable-vertex and average of factor-vertex.
  */
template <typename _F>
    class MessagePriorityInference : public SumProductInference<_F>
{
public:
  typedef SumProductInference<_F>                            _Base;
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
  MessagePriorityInference(const graph_type& graph, ConvergenceMeasure cvm = ConvergenceMeasure())
    : _Base(graph, cvm) {}

  /** The inference algorithm */
  void run();
protected:
  typedef std::pair<vertex_type, vertex_type>                                vertex_pair_type;
  typedef std::pair< vertex_pair_type, factor_value_type>                    msg_double_type;
  typedef Vector<msg_double_type>                                            msg_double_vector_type;
  struct MsgCompare
  {
    bool operator() (const msg_double_type& lhs, const msg_double_type& rhs) const
    {
      return lhs.second < rhs.second;
    }
  };
  typedef PriorityQueue<msg_double_type, msg_double_vector_type, MsgCompare> msg_double_queue_type;

  /** Priority queue of messages */
  msg_double_queue_type message_queue_;
  /** Total change of messages in an iteration */
  factor_value_type change_sum;

  /** Init the message queue by updating all messages once */
  void initMessageQueue();

  /** The L1 difference between two messages */
  factor_value_type difference(const message_type& oldMsg, const message_type& newMsg);
};

END_GRAPHICAL_MODEL_NAMESPACE;

#endif
