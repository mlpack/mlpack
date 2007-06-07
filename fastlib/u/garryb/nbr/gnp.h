#ifndef NBR_GNP_H
#define NBR_GNP_H

#include "kdtree.h"
#include "spnode.h"

#include "fastlib/fastlib.h"

struct BlankDelta {
 public:
  OT_DEF(BlankDelta) {}
 public:
  template<typename Param>
  void Init(const Param& param) {}
  template<typename Param>
  void ApplyDelta(const Param& param, const BlankDelta& other) {}
};

struct BlankQPostponed {
 public:
  OT_DEF(BlankQPostponed) {}
 public:
  template<typename Param>
  void Init(const Param& param) {}
  template<typename Param>
  void Reset(const Param& param) {}
  template<typename Param>
  void ApplyPostponed(const Param& param, const BlankQPostponed& other) {}
};

class BlankStat {
 public:
  OT_DEF(BlankStat) {}
 public:
  template<typename Param>
  void Init(const Param& param) {}
  template<typename Param, typename Point>
  void Accumulate(const Param& param, const Point& point) {}
  template<typename Param, typename Bound>
  void Accumulate(const Param& param,
      const BlankStat& stat, const Bound& bound, index_t n) {}
  template<typename Param, typename Bound>
  void Postprocess(const Param& param, const Bound& bound, index_t n) {}
};

struct BlankQResult {
 public:
  OT_DEF(BlankQResult) {}
 public:
  template<typename Param>
  void Init(const Param& param) {}
  template<typename Param, typename Point, typename RNode>
  void Postprocess(const Param& param,
      const Point& q_point, index_t q_index,
      const RNode& r_root) {}
  template<typename Param, typename QPostponed, typename Point>
  void ApplyPostponed(const Param& param,
      const QPostponed& postponed,
      const Point& q_point, index_t q_index) {}
};

class BlankGlobalResult {
 public:
  OT_DEF(BlankGlobalResult) {}
 public:
  template<typename Param>
  void Init(const Param& param) {}
  template<typename Param>
  void Accumulate(const Param& param, const BlankGlobalResult& other) {}
  template<typename Param, typename Delta>
  void ApplyDelta(const Param& param, const Delta& delta) {}
  template<typename Param, typename Delta>
  void UndoDelta(const Param& param, const Delta& delta) {}
  template<typename Param>
  void Postprocess(const Param& param) {}
  void Report(datanode *datanode) {}
};

struct BlankQMassResult {
 public:
  OT_DEF(BlankQMassResult) {}
 public:
  template<typename Param>
  void Init(const Param& param) {}
  template<typename Param, typename Delta>
  void ApplyDelta(const Param& param, const Delta& delta) {}
  template<typename Param, typename QPostponed, typename QNode>
  void ApplyPostponed(const Param& param,
      const QPostponed& postponed, const QNode& q_node) {}
  template<typename Param>
  void ApplyMassResult(const Param& param, const BlankQMassResult& other) {}
  template<typename Param, typename QNode>
  void StartReaccumulate(const Param& param, const QNode& q_node) {}
  template<typename Param, typename QResult>
  void Accumulate(const Param& param, const QResult& result) {}
  template<typename Param>
  void Accumulate(const Param& param, const BlankQMassResult& result,
      index_t n_points) {}
  template<typename Param, typename QNode>
  void FinishReaccumulate(const Param& param, const QNode& q_node) {}
};

class BlankAlgorithm {
 public:
  template<typename Param, typename QNode, typename RNode,
      typename Delta, typename QPostponed, typename GlobalResult>
  static bool ConsiderPairIntrinsic(
      const Param& param,
      const QNode& q_node,
      const RNode& r_node,
      Delta* delta,
      GlobalResult* global_result,
      QPostponed* q_postponed) {
    return true;
  }

  template<typename Param, typename QNode, typename RNode,
      typename Delta, typename QMassResult, typename QPostponed,
      typename GlobalResult>
  static bool ConsiderPairExtrinsic(
      const Param& param,
      const QNode& q_node,
      const RNode& r_node,
      const Delta& delta,
      const QMassResult& q_mass_result,
      const GlobalResult& global_result,
      QPostponed* q_postponed) {
    return true;
  }

  template<typename Param, typename QNode,
      typename QMassResult, typename QPostponed,
      typename GlobalResult>
  static bool ConsiderQueryTermination(
      const Param& param,
      const QNode& q_node,
      const QMassResult& q_mass_result,
      const GlobalResult& global_result,
      QPostponed* q_postponed) {
    return true;
  }
  
  template<typename Param, typename QNode, typename RNode>
  static double Heuristic(
      const Param& param,
      const QNode& q_node,
      const RNode& r_node) {
    return 0;
  }
};

#endif
