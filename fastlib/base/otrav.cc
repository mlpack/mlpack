/**
 * @file otrav.cc
 *
 * Non-templated implementations of object-traversal.
 *
 * Very little is defined here since pretty much every part of OT is
 * templated.
 */

#include "otrav.h"

void ot__private::ZOTPrinter::ShowIndents() {
  for (int i = 0; i < indent_amount_; i++) {
    putc(' ', stream_);
  }
}
void ot__private::ZOTPrinter::Write(const char *format, ...) {
  va_list vl;
  ShowIndents();
  va_start(vl, format);
  vfprintf(stream_, format, vl);
  va_end(vl);
  putc('\n', stream_);
}
