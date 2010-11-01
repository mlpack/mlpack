/** @file timer.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_UTIL_TIMER_H
#define CORE_UTIL_TIMER_H

#include <sys/time.h>
#include <string>
#include <sstream>

namespace core {
namespace util {
class Timer {

  public:

    void Start() {
      gettimeofday(&start_, NULL);
    }

    void End() {
      gettimeofday(&end_, NULL);
    }

    void Reset() {
      checkpoints_.clear();
    }

    int CheckPoint() {
      timeval curr_time;
      gettimeofday(&curr_time, NULL);
      checkpoints_.push_back(curr_time);
      return (checkpoints_.size() - 1);
    }

    std::string GetTotalElapsedTimeString() {
      timeval result;
      timersub(&end_, &start_, &result);
      std::stringstream str;
      str << (result.tv_sec + static_cast<double>(result.tv_usec) / 1000000.0);
      return str.str();
    }

    double GetTotalElapsedTime() {
      timeval result;
      timersub(&end_, &start_, &result);
      return (result.tv_sec + static_cast<double>(result.tv_usec) / 1000000.0);
    }

    double GetElapsedTime(int checkpoint_id) {
      timeval result;
      timersub(&checkpoints_[checkpoint_id], &start_, &result);
      return (result.tv_sec + static_cast<double>(result.tv_usec) / 1000000.0);
    }

  private:

    timeval start_;

    timeval end_;

    std::vector<timeval> checkpoints_;
};
};
};

#endif
