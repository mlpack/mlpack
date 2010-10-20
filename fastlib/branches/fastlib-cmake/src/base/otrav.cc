/* MLPACK 0.1
 *
 * Copyright (c) 2008 Alexander Gray,
 *                    Garry Boyer,
 *                    Ryan Riegel,
 *                    Nikolaos Vasiloglou,
 *                    Dongryeol Lee,
 *                    Chip Mappus, 
 *                    Nishant Mehta,
 *                    Hua Ouyang,
 *                    Parikshit Ram,
 *                    Long Tran,
 *                    Wee Chin Wong
 *
 * Copyright (c) 2008 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
// Copyright 2007 Georgia Institute of Technology. All rights reserved.
/**
 * otrav.cc (this file not included by Doxygen)
 *
 * Non-templated implementations of object traversal, which at present
 * is constrained to printer formats.
 */

#include "fastlib/base/otrav.h"
//#include "otrav.h"

void ot::StandardFormat::PrintIndent_() {
  for (int i = 0; i < indent_; ++i) {
    putc(' ', stream_);
  }
}

void ot::StandardFormat::PrintHeader_(const char *name, index_t index,
				      const char *type, index_t len) {
  if (index >= 0) {
    fprintf(stream_, "[%"LI"d]", index);
  } else {
    fprintf(stream_, "%s", name);
  }
  fprintf(stream_, " : ");
  if (len >= 0) {
    fprintf(stream_, "len %"LI"d ", len);
  }
  fprintf(stream_, "%s = ", type);
}

void ot::StandardFormat::Untraversed(const unsigned char *obj_cp,
				     size_t bytes) {
  while (bytes > 8) {
    PrintIndent_();
    for (size_t i = 0; i < 8; ++i) {
      fprintf(stream_, "%02X ", obj_cp[i]);
    }
    putc('\n', stream_);
    bytes -= 8;
    obj_cp += 8;
  }
  PrintIndent_();
  for (size_t i = 0; i < bytes; ++i) {
    fprintf(stream_, "%02X ", obj_cp[i]);
  }
  putc('\n', stream_);
}

#define STANDARD_FORMAT__PRIMITIVE(T, TF) \
    void ot::StandardFormat::Primitive(const char *name, index_t index, \
		                       const char *type, T val) { \
      PrintIndent_(); \
      PrintHeader_(name, index, type, -1); \
      fprintf(stream_, TF, val); \
      putc('\n', stream_); \
    }

FOR_ALL_PRIMITIVES_DO(STANDARD_FORMAT__PRIMITIVE)

void ot::StandardFormat::Primitive(const char *name, index_t index,
				   const char *type, bool val) {
  PrintIndent_();
  PrintHeader_(name, index, type, -1);
  fprintf(stream_, val ? "true" : "false");
  putc('\n', stream_);
}

#undef STANDARD_FORMAT__PRIMITIVE

void ot::StandardFormat::Str(const char *name, index_t index,
			     const char *type, const char *str) {
  PrintIndent_();
  PrintHeader_(name, index, type, -1);
  if (!str) {
    fprintf(stream_, "0x0");
  } else {
    char c;

    putc('"', stream_);
    while ((c = *str++)) {
      if (c == '"') {
	fprintf(stream_, "\\\"");
      } else if (c == '\a') {
	fprintf(stream_, "\\a");
      } else if (c == '\b') {
	fprintf(stream_, "\\b");
      } else if (c == '\f') {
	fprintf(stream_, "\\f");
      } else if (c == '\n') {
	fprintf(stream_, "\\n");
      } else if (c == '\t') {
	fprintf(stream_, "\\t");
      } else if (c == '\v') {
	fprintf(stream_, "\\v");
      } else if (c == '\\') {
	fprintf(stream_, "\\\\");
      } else {
	putc(c, stream_);
      }
    }
    putc('"', stream_);
  }
  putc('\n', stream_);
}

void ot::StandardFormat::Ptr(const char *name, index_t index, 
			     const char *type, ptrdiff_t ptr) {
  PrintIndent_();
  PrintHeader_(name, index, type, -1);
  if (sizeof(ptrdiff_t) >= sizeof(int)) {
    fprintf(stream_, "0x%X", (int)ptr);
  } else if (sizeof(ptrdiff_t) >= sizeof(long)) {
    fprintf(stream_, "0x%lX", (long)ptr);
  } else {
    fprintf(stream_, "0x%llX", (long long)ptr);
  }
  putc('\n', stream_);
}

void ot::StandardFormat::Open(const char *name, index_t index, 
			      const char *type, index_t len) {
  PrintIndent_();
  PrintHeader_(name, index, type, len);
  putc('\n', stream_);
  indent_ += 2;
  PrintIndent_();
  putc('{', stream_);
  putc('\n', stream_);
  indent_ += 2;
}

void ot::StandardFormat::Close(const char *name, const char *type) {
  indent_ -= 2;
  PrintIndent_();
  putc('}', stream_);
  putc('\n', stream_);
  indent_ -= 2;
}



void ot::XMLFormat::PrintIndent_() {
  for (int i = 0; i < indent_; ++i) {
    putc(' ', stream_);
  }
}

void ot::XMLFormat::PrintHeader_(const char *name, index_t index,
				 const char *type, index_t len) {
  fprintf(stream_, "<_%s", type);
  if (index >= 0) {
    fprintf(stream_, " index=\"%"LI"d\"", index);
  } else {
    fprintf(stream_, " name=\"%s\"", name);
  }
  if (len >= 0) {
    fprintf(stream_, " len=\"%"LI"d\"", len);
  }
  fprintf(stream_, ">");
}

void ot::XMLFormat::PrintFooter_(const char *type) {
  fprintf(stream_, "</_%s>", type);
}

void ot::XMLFormat::Untraversed(const unsigned char *obj_cp,
				size_t bytes) {
  while (bytes > 8) {
    PrintIndent_();
    for (size_t i = 0; i < 8; ++i) {
      fprintf(stream_, "%02X ", obj_cp[i]);
    }
    putc('\n', stream_);
    bytes -= 8;
    obj_cp += 8;
  }
  PrintIndent_();
  for (size_t i = 0; i < bytes; ++i) {
    fprintf(stream_, "%02X ", obj_cp[i]);
  }
  putc('\n', stream_);
}

#define XML_FORMAT__PRIMITIVE(T, TF) \
    void ot::XMLFormat::Primitive(const char *name, index_t index, \
		                  const char *type, T val) { \
      PrintIndent_(); \
      PrintHeader_(name, index, type, -1); \
      fprintf(stream_, TF, val); \
      PrintFooter_(type); \
      putc('\n', stream_); \
    }

FOR_ALL_PRIMITIVES_DO(XML_FORMAT__PRIMITIVE)

void ot::XMLFormat::Primitive(const char *name, index_t index,
			      const char *type, bool val) {
  PrintIndent_();
  PrintHeader_(name, index, type, -1);
  fprintf(stream_, val ? "true" : "false");
  PrintFooter_(type);
  putc('\n', stream_);
}

#undef XML_FORMAT__PRIMITIVE

void ot::XMLFormat::Str(const char *name, index_t index,
			const char *type, const char *str) {
  PrintIndent_();
  PrintHeader_(name, index, type, -1);
  if (str) {
    char c;

    while ((c = *str++)) {
      if (c == '"') {
	fprintf(stream_, "&quot;");
      } else if (c == '\'') {
	fprintf(stream_, "&apos;");
      } else if (c == '&') {
	fprintf(stream_, "&amp;");
      } else if (c == '<') {
	fprintf(stream_, "&lt;");
      } else if (c == '>') {
	fprintf(stream_, "&gt;");
      } else {
	putc(c, stream_);
      }
    }
  }
  PrintFooter_(type);
  putc('\n', stream_);
}

void ot::XMLFormat::Ptr(const char *name, index_t index,
			const char *type, ptrdiff_t ptr) {
  PrintIndent_();
  PrintHeader_(name, index, type, -1);
  if (sizeof(ptrdiff_t) >= sizeof(int)) {
    fprintf(stream_, "0x%X", (int)ptr);
  } else if (sizeof(ptrdiff_t) >= sizeof(long)) {
    fprintf(stream_, "0x%lX", (long)ptr);
  } else {
    fprintf(stream_, "0x%llX", (long long)ptr);
  }
  PrintFooter_(type);
  putc('\n', stream_);
}

void ot::XMLFormat::Open(const char *name, index_t index,
			 const char *type, index_t len) {
  PrintIndent_();
  PrintHeader_(name, index, type, len);
  putc('\n', stream_);
  indent_ += 4;
}

void ot::XMLFormat::Close(const char *name, const char *type) {
  indent_ -= 4;
  PrintIndent_();
  PrintFooter_(type);
  putc('\n', stream_);
}
