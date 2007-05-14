#ifndef NBR_GNP_H
#define NBR_GNP_H

#include "kdtree.h"
#include "spnode.h"

#include "fastlib/fastlib.h"

template<
  typename TParam, typename TAlgorithm,
  typename TPoint, typename TBound,
  typename TQPointInfo, typename TQStat,
  typename TRPointInfo, typename TRStat,
  typename TPairVisitor, typename TDelta,
  typename TQResult, typename TQMassResult, typename TQPostponed,
  typename TGlobalResult>
class DualTreeGNP {
 public:
  typedef TParam Param;
  typedef TAlgorithm Algorithm;

  typedef TPoint Point;
  typedef TBound Bound;

  typedef TQPointInfo QPointInfo;
  typedef TQStat QStat;

  typedef TRPointInfo RPointInfo;
  typedef TRStat RStat;

  typedef TPairVisitor PairVisitor;
  typedef TDelta Delta;

  typedef TQResult QResult;
  typedef TQMassResult QMassResult;
  typedef TQPostponed QPostponed;

  typedef TGlobalResult GlobalResult;

  typedef SpNode<Bound, QStat> QNode;
  typedef SpNode<Bound, RStat> RNode;
};

struct BlankPointInfo {
  OT_DEF(BlankPointInfo) {}
};

template<typename Param>
struct BlankDelta {
 public:
  OT_DEF(BlankDelta) {}
 public:
  void Init(const Param& param) {}
  void ApplyDelta(const Param& param, const BlankDelta& other) {}
};

template <typename Param>
struct BlankQPostponed {
 public:
  OT_DEF(BlankQPostponed) {}
 public:
  void Init(const Param& param) {}
  void Reset(const Param& param) {}
  void ApplyPostponed(const Param& param, const BlankQPostponed& other) {}
};

template<typename Param, typename Point, typename Bound, typename PointInfo>
class BlankStat {
 public:
  OT_DEF(BlankStat) {}
 public:
  void Init(const Param& param) {}
  void Accumulate(const Param& param, const Point& point,
      const PointInfo& r_info) {}
  void Accumulate(const Param& param,
      const BlankStat& stat, const Bound& bound, index_t n) {}
  void Postprocess(const Param& param, const Bound& bound, index_t n) {}
};

template <typename Param, typename Point, typename Bound,
    typename QPointInfo, typename QPostponed, typename RNode>
struct BlankQResult {
 public:
  OT_DEF(BlankQResult) {}
 public:
  void Init(const Param& param,
      const Point& q_point, const QPointInfo& q_info,
      const RNode& r_root) {}
  void Postprocess(const Param& param,
      const Point& q_point, const QPointInfo& q_info,
      const RNode& r_root) {}
  void ApplyPostponed(const Param& param,
      const QPostponed& postponed,
      const Point& q_point) {}
};

template <typename Param, typename Delta>
class BlankGlobalResult {
 public:
  OT_DEF(BlankGlobalResult) {}
 public:
  void Init(const Param& param) {}
  void Accumulate(const Param& param, const BlankGlobalResult& other) {}
  void ApplyDelta(const Param& param, const Delta& delta) {}
  void UndoDelta(const Param& param, const Delta& delta) {}
  void Postprocess(const Param& param) {}
};

template<typename Param, typename QNode,
    typename Delta, typename QResult, typename QPostponed>
struct BlankQMassResult {
 public:
  OT_DEF(BlankQMassResult) {}
 public:
  void Init(const Param& param) {}
  void ApplyDelta(const Param& param, const Delta& delta) {}
  void ApplyPostponed(const Param& param,
      const QPostponed& postponed, const QNode& q_node) {}
  void ApplyMassResult(const Param& param, const BlankQMassResult& other) {}
  void StartReaccumulate(const Param& param, const QNode& q_node) {}
  void Accumulate(const Param& param, const QResult& result) {}
  void Accumulate(const Param& param, const BlankQMassResult& result,
      index_t n_points) {}
  void FinishReaccumulate(const Param& param, const QNode& q_node) {}
};

template<typename Param, typename QNode, typename RNode,
    typename Delta, typename QMassResult, typename QPostponed,
    typename GlobalResult>
class BlankAlgorithm {
 public:
  static bool ConsiderPairIntrinsic(
      const Param& param,
      const QNode& q_node,
      const RNode& r_node,
      Delta* delta,
      GlobalResult* global_result,
      QPostponed* q_postponed) {
    return true;
  }

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

  static bool ConsiderQueryTermination(
      const Param& param,
      const QNode& q_node,
      const QMassResult& q_mass_result,
      const GlobalResult& global_result,
      QPostponed* q_postponed) {
    return true;
  }
  
  static double Heuristic(
      const Param& param,
      const QNode& q_node,
      const RNode& r_node) {
    return 0;
  }
};

#endif
