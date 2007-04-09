#ifndef SUPERPAR_GNP_H
#define SUPERPAR_GNP_H

template<
    typename TArray,
    typename TBound,
    typename TPoint,
    typename TQStat,
    typename TRStat,
    typename TQMutStat,
    typename TQResult
    typename TGlobalStat,
    typename TGlobalMutStat,
    >
class GnpRunnerQr {
 public:
  /** The array type used to store data. */
  typedef TArray Array;
  /** The bounding type. */
  typedef TBound Bound;
  /** Statistic that is pre-computed for each query node. */
  typedef TQStat QStat;
  /** Statistic that is pre-computed for each reference node. */
  typedef TRStat RStat;
  /** Statistic updated for each query node as part of the GNP computation. */
  typedef TQMutStat QMutStat;
  /** A desired result for each query point. */
  typedef TQResult QResult;
  
  /** Statistic computed for the entire computation. */
  typedef TGlobalStat GlobalStat;
  /** Statistic that is continually updated and assists in pruning. */
  typedef TGlobalMutStat GlobalMutStat;
  
 private:
  Array<Bound> qbound_;
  Array<Bound> rbound_;
  Array<QStat> qstat_;
  Array<RStat> rstat_;
  Array<QMutStat> qmutstat_;
  GlobalStat globalstat_;
  GlobalMutStat globalmutstat_;
};

#endif
