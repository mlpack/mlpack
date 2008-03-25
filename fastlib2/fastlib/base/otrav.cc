// Copyright 2007 Georgia Institute of Technology. All rights reserved.
/**
 * otrav.cc (this file not included by Doxygen)
 *
 * Non-templated implementations of object traversal, which at present
 * is constrained to printer formats.
 */

#include "otrav.h"

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

#undef STANDARD_FORMAT__PRIMITIVE

  void ot::StandardFormat::Str(const char *name, index_t index,
			       const char *type, const char *str) {
  PrintIndent_();
  PrintHeader_(name, index, type, -1);
  if (!str) {
    fprintf(stream_, "0x0");
  } else {
    putc('"', stream_);
    while (*str) {
      if (*str == '"') {
	fprintf(stream_, "\\\"");
      } else if (*str == '\a') {
	fprintf(stream_, "\\a");
      } else if (*str == '\b') {
	fprintf(stream_, "\\b");
      } else if (*str == '\f') {
	fprintf(stream_, "\\f");
      } else if (*str == '\n') {
	fprintf(stream_, "\\n");
      } else if (*str == '\t') {
	fprintf(stream_, "\\t");
      } else if (*str == '\v') {
	fprintf(stream_, "\\v");
      } else if (*str == '\\') {
	fprintf(stream_, "\\\\");
      } else {
	putc(*str, stream_);
      }
      ++str;
    }
    putc('"', stream_);
  }
  putc('\n', stream_);
}

void ot::StandardFormat::Ptr(const char *name, index_t index, 
			     const char *type, ptrdiff_t ptr) {
  PrintIndent_();
  PrintHeader_(name, index, type, -1);
  fprintf(stream_, "0x%X", ptr);
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

#undef XML_FORMAT__PRIMITIVE

void ot::XMLFormat::Str(const char *name, index_t index,
			const char *type, const char *str) {
  PrintIndent_();
  PrintHeader_(name, index, type, -1);
  if (str) {
    while (*str) {
      if (*str == '"') {
	fprintf(stream_, "&quot;");
      } else if (*str == '\'') {
	fprintf(stream_, "&apos;");
      } else if (*str == '&') {
	fprintf(stream_, "&amp;");
      } else if (*str == '<') {
	fprintf(stream_, "&lt;");
      } else if (*str == '>') {
	fprintf(stream_, "&gt;");
      } else {
	putc(*str, stream_);
      }
      ++str;
    }
  }
  PrintFooter_(type);
  putc('\n', stream_);
}

void ot::XMLFormat::Ptr(const char *name, index_t index,
			const char *type, ptrdiff_t ptr) {
  PrintIndent_();
  PrintHeader_(name, index, type, -1);
  fprintf(stream_, "0x%X", ptr);
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
