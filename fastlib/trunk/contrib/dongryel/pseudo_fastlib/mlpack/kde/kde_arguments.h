#ifndef MLPACK_KDE_KDE_ARGUMENTS_H
#define MLPACK_KDE_KDE_ARGUMENTS_H

#include "core/table/table.h"

namespace ml {
class KdeArguments {
  public:
    core::table::Table *reference_table_;

    core::table::Table *query_table_;

    double bandwidth_;

    double relative_error_;

    double probability_;

  public:
    KdeArguments() {
      reference_table_ = NULL;
      query_table_ = NULL;
      bandwidth_ = 0.0;
      relative_error_ = 0.0;
      probability_ = 0.0;
    }

    ~KdeArguments() {
      if (reference_table_ == query_table_) {
        delete reference_table_;
      }
      else {
        delete reference_table_;
        delete query_table_;
      }
      reference_table_ = NULL;
      query_table_ = NULL;
    }
};
};

#endif
