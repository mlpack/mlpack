/** @file scoped_omp_lock.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_SCOPED_OMP_LOCK_H
#define CORE_PARALLEL_SCOPED_OMP_LOCK_H

#include <omp.h>

namespace core {
namespace parallel {

class scoped_omp_lock {
  private:
    omp_lock_t *lock_;

  public:

    scoped_omp_lock(omp_lock_t *lock_in) {
      lock_ = lock_in;
      omp_set_lock(lock_);
    }

    ~scoped_omp_lock() {
      omp_unset_lock(lock_);
    }
};

class scoped_omp_nest_lock {
  private:
    omp_nest_lock_t *lock_;

  public:

    scoped_omp_nest_lock(omp_nest_lock_t *lock_in) {
      lock_ = lock_in;
      omp_set_nest_lock(lock_);
    }

    ~scoped_omp_nest_lock() {
      omp_unset_nest_lock(lock_);
    }
};
}
}

#endif
