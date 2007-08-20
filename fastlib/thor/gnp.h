/**
 * @file gnp.h
 *
 * Abstract concepts used by THOR generalized N-body problems.
 *
 * This mostly contains blank elements that you can substitute if certain
 * parts of the problem don't make sense for you.
 */

#ifndef THOR_GNP_H
#define THOR_GNP_H

#include "la/matrix.h"
#include "data/dataset.h"
#include "base/otrav.h"
#include "fx/fx.h"

/**
 * A point in vector-space.
 *
 * This is a wrapper around Vector which implements the vec() method, because
 * that is what the THOR tree-builders expect.
 *
 * A side note, it's possible to have a THOR problem that doesn't use vectors
 * at all.  Instead, define your own point, and you have to make your own
 * tree-builder.
 */
class ThorVectorPoint {
 private:
  Vector vec_;

  OT_DEF_BASIC(ThorVectorPoint) {
    OT_MY_OBJECT(vec_);
  }

 public:
  /**
   * Gets the vector.
   */
  const Vector& vec() const {
    return vec_;
  }
  /**
   * Gets the vector.
   */
  Vector& vec() {
    return vec_;
  }
  /**
   * Initializes a "default element" from a dataset schema.
   *
   * This is the only function that allows allocation.
   */
  template<typename Param>
  void Init(const Param& param, const DatasetInfo& schema) {
    vec_.Init(schema.n_features());
    vec_.SetZero();
  }
  /**
   * Sets the values of this object, not allocating any memory.
   *
   * If memory needs to be allocated it must be allocated at the beginning
   * with Init.
   *
   * @param param ignored
   * @param index the index of the point, ignored
   * @param data the vector data read from file
   */
  template<typename Param>
  void Set(const Param& param, index_t index, const Vector& data) {
    vec_.CopyValues(data);
  }
};

struct BlankDelta {
 public:
  OT_DEF_BASIC(BlankDelta) {}
 public:
  template<typename Param>
  void Init(const Param& param) {}
};

struct BlankQPostponed {
 public:
  OT_DEF_BASIC(BlankQPostponed) {}
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
  OT_DEF_BASIC(BlankStat) {}
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
  OT_DEF_BASIC(BlankQResult) {}
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
  OT_DEF_BASIC(BlankGlobalResult) {}
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
  template<typename Param>
  void Report(const Param& param, datanode *datanode) const {}
  template<typename Param, typename QPoint, typename QResult>
  void ApplyResult(const Param& param,
      const QPoint& q_point, index_t q_index, const QResult& q_result) {}
};

struct BlankQSummaryResult {
 public:
  OT_DEF_BASIC(BlankQSummaryResult) {}
 public:
  template<typename Param>
  void Init(const Param& param) {}
  template<typename Param, typename Delta>
  void ApplyDelta(const Param& param, const Delta& delta) {}
  template<typename Param, typename QPostponed, typename QNode>
  void ApplyPostponed(const Param& param,
      const QPostponed& postponed, const QNode& q_node) {}
  template<typename Param>
  void ApplySummaryResult(const Param& param, const BlankQSummaryResult& other) {}
  template<typename Param, typename QNode>
  void StartReaccumulate(const Param& param, const QNode& q_node) {}
  template<typename Param, typename QResult>
  void Accumulate(const Param& param, const QResult& result) {}
  template<typename Param>
  void Accumulate(const Param& param, const BlankQSummaryResult& result,
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
      typename Delta, typename QSummaryResult, typename QPostponed,
      typename GlobalResult>
  static bool ConsiderPairExtrinsic(
      const Param& param,
      const QNode& q_node,
      const RNode& r_node,
      const Delta& delta,
      const QSummaryResult& q_summary_result,
      const GlobalResult& global_result,
      QPostponed* q_postponed) {
    return true;
  }

  template<typename Param, typename QNode,
      typename QSummaryResult, typename QPostponed,
      typename GlobalResult>
  static bool ConsiderQueryTermination(
      const Param& param,
      const QNode& q_node,
      const QSummaryResult& q_summary_result,
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
