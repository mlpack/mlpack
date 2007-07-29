#ifndef COL_QUEUE_H
#define COL_QUEUE_H

template<typename T>
class Queue {
 private:
  struct Node {
    T data;
    Node *next;

    Node() : next(NULL) { }
    Node(const T& value) : data(value), next(NULL) {}
  };

 private:
  Node *head_;
  Node **tailp_;

 public:
  Queue() { DEBUG_POISON_PTR(tailp_); }
  ~Queue() {
    Clear();
  }

  void Init() {
    head_ = NULL;
    tailp_ = &head_;
  }

  T *Add() {
    Node *node = new Node();
    *tailp_ = node;
    tailp_ = &node->next;
    return &node->data;
  }

  T *Add(const T& value) {
    Node *node = new Node(value);
    node->next = NULL;
    *tailp_ = node;
    tailp_ = &node->next;
    return &node->data;
  }

  void PopOnly() {
    Node *tmp = head_;
    head_ = head_->next;
    delete tmp;
    if (unlikely(head_ == NULL)) {
      tailp_ = &head_;
    }
  }

  T Pop() {
    T val(head_->data);
    PopOnly();
    return val;
  }

  bool is_empty() const {
    return head_ == NULL;
  }

  const T& top() const {
    return head_->data;
  }

  T& top() {
    return head_->data;
  }

  void Clear() {
    Node *cur;
    Node *next;

    for (cur = head_; cur; cur = next) {
      next = cur->next;
      delete cur;
    }
  }
};

#endif
