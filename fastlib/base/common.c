/**
 * @file common.c
 *
 * Implementations for bare-necessities FASTlib programming in C.
 */

#include "base/common.h"

#include <stdarg.h>
#include <unistd.h>

int segfault_on_abort = 0;
int abort_on_nonfatal = 0;
int pause_on_nonfatal = 0;
int print_notify_locs = 0;

char fl_msg_marker[] = {'X', '!', '*', '.'};
const char * fl_msg_color[] =
  {ANSI_HRED, ANSI_HYELLOW, ANSI_HGREEN, ANSI_BLUE};

void fl_abort(void)
{
  if (segfault_on_abort) {
    fflush(NULL);
    *(int*)NULL = 0;
  }
  abort();
}

void fl_pause(void)
{
  if (isatty(0)) {
    char c;

    fprintf(stderr, ANSI_HBLACK"Press Return to continue..."ANSI_CLEAR);
    fflush(stderr);

    while ((c = getchar()) != EOF && c != '\n');
  }
}

void fl_print_msg_header(char marker, const char *color)
{
  fprintf(stderr, "%s[%c]%s ", color, marker, ANSI_CLEAR);
}

void fl_print_msg_loc(const char *file, const char *func, int line)
{
  const char *prev = file;
  const char *last = file;

  /* Finds the file and containing directory, if it exists. */
  while ((file = strchr(last, '/'))) {
    prev = last;
    last = file + 1;
  }

  fprintf(stderr, "%s:%s:%d: ", prev, func, line);
}

void fl_print_fatal_msg(const char *file, const char *func, int line,
			const char *format, ...)
{
  va_list vl;

  fl_print_msg_header(fl_msg_marker[FL_MSG_FATAL], fl_msg_color[FL_MSG_FATAL]);
  fl_print_msg_loc(file, func, line);

  va_start(vl, format);
  vfprintf(stderr, format, vl);
  va_end(vl);

  fprintf(stderr, "\n");

  fl_abort();
}

void fl_print_msg(const char *file, const char *func, int line,
		  fl_msg_t msg_type, const char *format, ...)
{
  va_list vl;

  fl_print_msg_header(fl_msg_marker[msg_type], fl_msg_color[msg_type]);
  if (msg_type < FL_MSG_NOTIFY_STAR || print_notify_locs) {
    fl_print_msg_loc(file, func, line);
  }

  va_start(vl, format);
  vfprintf(stderr, format, vl);
  va_end(vl);

  fprintf(stderr, "\n");

  if (msg_type < FL_MSG_NOTIFY_STAR) {
    if (msg_type == FL_MSG_FATAL || abort_on_nonfatal) {
      fl_abort();
    } else if (pause_on_nonfatal) {
      fl_pause();
    }
  }
}

void fl_print_progress(const char *desc, int prec)
{
  const int BAR_LEN = 50;
  static int prev_prec = -1;
  static const char *prev_desc = NULL;

  if (isatty(0)) {
    if (unlikely(prec != prev_prec || desc != prev_desc)) {
      char buf[BAR_LEN + 1];
      int pos = prec * BAR_LEN / 100;
      int i = 0;

      pos = unlikely(pos > BAR_LEN) ? BAR_LEN : unlikely(pos < 0) ? 0 : pos;

      for (; i < pos; ++i) {
	buf[i] = '#';
      }
      for (; i < BAR_LEN; ++i) {
	buf[i] = '.';
      }
      buf[i] = '\0';

      fprintf(stderr, "\r"ANSI_BLUE"[%s] %d%% %s"ANSI_CLEAR"\r",
	      buf, prec, desc);

      prev_prec = prec;
      prev_desc = desc;
    }
  }
}
