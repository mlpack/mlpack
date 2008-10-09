#ifndef NWRCDE_STAT_H
#define NWRCDE_STAT_H

class NWRCdeStat {
 public:
    
  void Init(const Matrix& dataset, index_t &start, index_t &count) {
  }
    
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const NWRCdeStat& left_stat, const NWRCdeStat& right_stat) {
  }
    
  NWRCdeStat() { }
    
  ~NWRCdeStat() { }
    
};

#endif
