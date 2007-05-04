/**
 * @file otrav.cpp
 *
 * Definitions for object-traversal.
 */

#include "otrav.h"

namespace ot_private {
  void OTPrinter::ShowIndents() {
    for (int i = 0; i < indent_amount_; i++) {
      putc(' ', stream_);
    }
  }
  void OTPrinter::Write(const char *format, ...) {
    va_list vl;
    ShowIndents();
    va_start(vl, format);
    vfprintf(stream_, format, vl);
    va_end(vl);
    putc('\n', stream_);
  }
};
