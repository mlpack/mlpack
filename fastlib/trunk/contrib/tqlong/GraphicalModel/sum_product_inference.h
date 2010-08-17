#ifndef SUM_PRODUCT_INFERENCE_H
#define SUM_PRODUCT_INFERENCE_H

#include "gm.h"

BEGIN_GRAPHICAL_MODEL_NAMESPACE;

/** The sum-product algorithm:
  * + Propagating the messages along all edges for a certain number of iterations
  * + Use messages to calculate beliefs of variable-vertex and average of factor-vertex
  */
template <typename _F>
    class SumProductInference : public NaiveInference<_F>
{
public:
  typedef NaiveInference<_F>                                 _Base;
  typedef typename _Base::factor_type                        factor_type;         // map from Assignment --> factor_value_type
  typedef typename _Base::factor_value_type                  factor_value_type;   // a numeric type
  typedef typename _Base::graph_type                         graph_type;          // FactorGraph<factor_type>
  typedef typename _Base::vertex_type                        vertex_type;         // Vertex* (see factor_graph.h)
  typedef typename _Base::vertex_vector_type                 vertex_vector_type;  // Vector<vertex_type>
  typedef typename _Base::belief_type                        belief_type;         // Map<Value, factor_value_type>
  typedef typename _Base::belief_map_type                    belief_map_type;     // Map<vertex_type, belief_type>
  class message_type : public belief_type
  {
  public:
    message_type(int cardinality = 0, const factor_value_type& value = factor_value_type(1.0))
    {
      for (int val = 0; val < cardinality; val++)
        this->operator << (typename message_type::value_type(val, value));
    }
    std::string toString(const graph_type& graph, const vertex_type& u, const vertex_type& v) const
    {
      std::ostringstream cout;
      cout << "msg ";
      if (u->isVariable())
        cout << u->variable()->toString() << " --> " << graph.factor(v).toString();
      else
        cout << graph.factor(u).toString() << " --> " << v->variable()->toString();
      cout << " : ";
      BOOST_FOREACH(const typename message_type::value_type& p, *this)
      {
        cout << p.first << " --> " << p.second << " ";
      }
      return cout.str();
    }
  };
  typedef Map<vertex_type, Map<vertex_type, message_type> >  messages_map_type;   // Map<Value, factor_value_type>
                                                                                  // usage: message[u][v] where u,v are vertex_type
public:
  /** Preparing inference on a graph */
  SumProductInference(const graph_type& graph, ConvergenceMeasure cvm = ConvergenceMeasure()) : _Base(graph), cvm_(cvm), curCvm_(cvm) {}
 
  /** The inference algorithm */
  void run();
protected:
  ConvergenceMeasure cvm_, curCvm_;
  messages_map_type messages_;

  /** Check terminating condition */
  bool terminateCondition();

  /** Create messages for every edge */
  void initMessages();

  /** Calculate beliefs of every variable vertex and average value of every factor */
  void calculateBeliefs();

  /** Update the message one each edge using its neighbor-messages */
  void updateMessage(const vertex_type& u, const vertex_type& v, message_type* oldMessage = NULL);
};

// Calculate the messages (for a certain number of iteration) and beliefs
template <typename _F> void SumProductInference<_F>::run()
{
  cout << "----------------------- Sum-Product Inference -----------------" << endl;
  this->initBeliefs();
  initMessages();
  for (curCvm_.iter_ = 0; ; curCvm_.iter_++)
  {
    cout << "---------------------- iter = " << curCvm_.iter_ << " ------------------------------" << endl;
    BOOST_FOREACH(const vertex_type& u, this->graph_.vertices())
      BOOST_FOREACH(const vertex_type& v, this->graph_.neighbors(u))
        updateMessage(u, v);
    if (terminateCondition()) break;
  }
  calculateBeliefs();
  this->normalizeBeliefs();
}

template <typename _F> void SumProductInference<_F>::initMessages()
{
  BOOST_FOREACH(const vertex_type& u, this->graph_.vertices())
  {
    BOOST_FOREACH(const vertex_type& v, this->graph_.neighbors(u))
    {
      int cardinality = u->isVariable() ? u->variable()->cardinality() : v->variable()->cardinality();
      messages_[u][v] = message_type(cardinality);
    }
  }
}

template <typename _F> bool SumProductInference<_F>::terminateCondition()
{
  return curCvm_ > cvm_;
}

template <typename _F> void SumProductInference<_F>::updateMessage(const vertex_type& u, const vertex_type& v, message_type* oldMessage)
{
  message_type& msg = messages_[u][v];
  if (oldMessage) (*oldMessage) = msg;
//  cout << "BEFORE " << messages_[u][v].toString(this->graph_, u, v) << endl;
  if (u->isVariable())  // msg[u][v] = product of msg[t][u] \forall t \in nb(u) \neq v
  {
    BOOST_FOREACH (const typename message_type::value_type& p, msg)
    {
      factor_value_type prod(1.0);
      BOOST_FOREACH (const vertex_type& t, this->graph_.neighbors(u))
        if (t != v) prod *= messages_[t][u][p.first];
      msg[p.first] = prod;
    }
  }
  else // msg[u][v] = sum over all assignments of factor(u) * product of msg[t][u]  \forall t \in nb(u) \neq v
  {
    msg = message_type(v->variable()->cardinality(), factor_value_type(0.0));
    const factor_type& f = this->graph_.factor(u);
    BOOST_FOREACH(const Assignment& a, f.assignments())
    {
      factor_value_type f_val = f.get(a);
      Value v_val;
      BOOST_FOREACH(const Assignment::value_type& p, a)
      {
        const Variable* t_var = p.first;
        const Value& t_val = p.second;
        const vertex_type& t = this->graph_.dataVertexMap().get((void*) t_var);
        if (t == v)
          v_val = t_val;
        else
          f_val *= messages_[t][u].get(t_val);
      }
      msg[v_val] += f_val;
    }
  }
//  cout << "AFTER  " << messages_[u][v].toString(this->graph_, u, v) << endl;
}

template <typename _F> void SumProductInference<_F>::calculateBeliefs()
{
  BOOST_FOREACH(const vertex_type& u, this->graph_.vertices())
  {
    if (u->isVariable())
    {
      int cardinality = u->variable()->cardinality();
      for (int val = 0; val < cardinality; val++)  // beliefs_[u][val] = product of msg[v][u][val]
      {
        this->beliefs_[u][val] = factor_value_type(1.0);
        BOOST_FOREACH(const vertex_type& v, this->graph_.neighbors(u))
          this->beliefs_[u][val] *= messages_[v][u][val];
      }
    }
    else
    {
      const factor_type& f = this->graph_.factor(u);
      BOOST_FOREACH(const Assignment& a, f.assignments())
      {
        factor_value_type f_val = f.get(a);
        factor_value_type msg_prod = factor_value_type(1.0);
        BOOST_FOREACH(const Assignment::value_type& p, a)
        {
          const Variable* t_var = p.first;
          const Value& t_val = p.second;
          const vertex_type& t = this->graph_.dataVertexMap().get((void*) t_var);
          msg_prod *= messages_[t][u][t_val];
        }
        this->beliefs_[u][0] += f_val * (f_val*msg_prod);    // f_val*msg_prod = p(a) where a is the current assignment
        this->beliefs_[u][1] += f_val*msg_prod;
      }
    }
  }
}

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

template <typename _F> void MessagePriorityInference<_F>::run()
{
  cout << "----------------------- Message Priority Sum-Product Inference -----------------" << endl;
  this->initBeliefs();
  this->initMessages();
  cout << "---------------------- iter = " << (this->curCvm_.iter_=0) << " Initializiing ----------------" << endl;
  initMessageQueue();
  for (this->curCvm_.iter_ = 1, this->curCvm_.changeTolerance_ = this->cvm_.changeTolerance_+1;
       ;
       this->curCvm_.iter_++)
  {
    cout << "---------------------- iter = " << this->curCvm_.iter_ << " ------------------------------" << endl;
    int n_msg = message_queue_.size();
    change_sum = factor_value_type(0.0);
    for (int i = 0; i < n_msg; i++)
    {
      // get the most changed (u, v) from the priority queue
      msg_double_type top = message_queue_.top();
      vertex_type u = top.first.first, v = top.first.second;

      message_type oldMsg;
      updateMessage(u, v, &oldMsg);

      // pop (u,v) out of queue
      message_queue_.pop();
      // push (u,v) back to queue with difference of old and new message as priority
      factor_value_type new_difference = difference(oldMsg, this->messages_[u][v]);
      message_queue_ << msg_double_type( vertex_pair_type(u, v), new_difference);
//      cout << "(" << this->graph_.toString(u) << ", " << this->graph_.toString(v)
//           << ") " << top.second << " --> " << new_difference << endl;
      change_sum += new_difference;
    }
    cout << "Total message change = " << change_sum << endl;
    this->curCvm_.changeTolerance_ = toDouble(change_sum);
    if (this->terminateCondition()) break;
  }
  this->calculateBeliefs();
  this->normalizeBeliefs();
}

template <typename _F> void MessagePriorityInference<_F>::initMessageQueue()
{
  BOOST_FOREACH(const vertex_type& u, this->graph_.vertices())
    BOOST_FOREACH(const vertex_type& v, this->graph_.neighbors(u))
    {
      message_type oldMsg;
      updateMessage(u, v, &oldMsg);
      factor_value_type initialValue = difference(oldMsg, this->messages_[u][v]);
      message_queue_ << msg_double_type( vertex_pair_type(u, v), initialValue);
    }
}

template <typename _F>
typename MessagePriorityInference<_F>::factor_value_type MessagePriorityInference<_F>::difference(const message_type& oldMsg, const message_type& newMsg)
{
  // calculate the l1 norm of the difference
  factor_value_type l1(0.0);
  BOOST_FOREACH(const typename message_type::value_type& p, oldMsg)
  {
    Value val = p.first;
    factor_value_type oldVal = p.second;
    factor_value_type newVal = newMsg.get(val);
    l1 += (oldVal > newVal) ? (oldVal-newVal) : (newVal-oldVal);
  }
  return l1;
}

END_GRAPHICAL_MODEL_NAMESPACE;

#endif // SUM_PRODUCT_INFERENCE_H
