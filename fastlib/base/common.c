/**
 * @file common.c
 *
 * Implementations for the base library.
 */

#include "base/common.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

double debug_verbosity = 0.0;

int abort_on_nonfatal = 0;
int pause_on_nonfatal = 0;
int print_notify_headers = 0;
#ifdef VERBOSE
int print_got_heres = 1;
int print_warnings = 1;
#else
int print_got_heres = 0;
int print_warnings = 0;
#endif

void fl_pause(void)
{
  if (isatty(0)) {
    char c;

    fprintf(stderr, "=*=*= Press Return to continue =*=*=");
    fflush(stderr);

    while ((c = getchar()) != EOF && c != '\n');
  }
}

const char *fl_filename(const char *name)
{
  const char *found = strstr(name, "/c/");

  if (found) {
    return found + 3;
  } else {
    const char *prev = 0;
    const char *last = 0;

    found = name;
    while ((found = strchr(found + 1, '/'))) {
      prev = last;
      last = found;
    }

    if (prev) {
      return prev;
    } else {
      return name;
    }
  }
}

void fl_msg_header(char type, const char *color,
    const char *file, const char *func, int line)
{
  fprintf(stderr, "%s[%c]%s %s:%s:%d: ",
      color, type, ANSI_CLEAR, fl_filename(file), func, line);
}
	
void fatal(const char *file, const char *func, int line,
	   const char *format, ...)
{
  va_list vl;

  fl_msg_header('X', ANSI_HRED, file, func, line);

  va_start(vl, format);
  vfprintf(stderr, format, vl);
  va_end(vl);

  fprintf(stderr, "\n");

  *(int*)NULL = 0;
  abort();
}

void nonfatal(const char *file, const char *func, int line,
	      const char *format, ...)
{
  va_list vl;

  fl_msg_header('!', ANSI_HYELLOW, file, func, line);

  va_start(vl, format);
  vfprintf(stderr, format, vl);
  va_end(vl);

  fprintf(stderr, "\n");

  if (abort_on_nonfatal) {
    *(int*)NULL = 0;
    abort();
  } else if (pause_on_nonfatal) {
    fl_pause();
  }
}

void notify(const char *file, const char *func, int line,
	    const char *format, ...)
{
  va_list vl;

  if (print_notify_headers) {
    fl_msg_header('.', ANSI_BLUE, file, func, line);
  }

  va_start(vl, format);
  vfprintf(stderr, format, vl);
  va_end(vl);
  
  fprintf(stderr, "\n");
}

static char tsprintf_pool[TSPRINTF_COUNT][TSPRINTF_LENGTH];
static int tsprintf_index = 0;

char *tsprintf(const char *format, ...) {
  char *s = tsprintf_pool[tsprintf_index];
  va_list vl;

  tsprintf_index = (tsprintf_index + 1) % TSPRINTF_COUNT;

  va_start(vl, format);
  vsnprintf(s, TSPRINTF_LENGTH, format, vl);
  va_end(vl);

  return s;
}

void percent_indicator(const char *what, uint64 numerator, uint64 denominator) {
  static int last_id = -1;
  char buf[80];
  int length = 50;
  int i = 0;
  int percent = (int)(numerator * 100 / denominator);
  int id = percent;

  if (unlikely(id != last_id)) {
    last_id = id;

    if (percent == 100) {
      for (; i < sizeof(buf) - 1; i++) {
        buf[i] = ' ';
      }
      buf[i] = '\0';
      fprintf(stderr, "%s\r", buf);
    } else {
      int num_ticks = percent * length / 100;

      for (i = 0; i < num_ticks; i++) {
        buf[i] = '#';
      }
      for (; i < length; i++) {
        buf[i] = '.';
      }

      buf[i] = '\0';

      fprintf(stderr, "\r"ANSI_BLUE" [%s] %02d%% %s"ANSI_CLEAR"  \r",
          buf, percent, what);
    }
  }
}
