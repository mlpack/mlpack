// Copyright 2007 Georgia Institute of Technology. All rights reserved.
/**
 * @file ansi_colors.h
 *
 * Definitions for various ANSI color sequences for use in stderr.
 */

#ifndef BASE_ANSI_COLORS_H
#define BASE_ANSI_COLORS_H

/** ANSI color sequence wrapper */
#define ANSI_SEQ(str) "\033["str"m"
/** Clears ANSI colors */
#define ANSI_CLEAR ANSI_SEQ("0")
/** Begin high-intensity */
#define ANSI_BOLD ANSI_SEQ("1")

/** Color code: High-intensity Black */
#define ANSI_HBLACK ANSI_SEQ("1;30")
/** Color code: High-intensity Red */
#define ANSI_HRED ANSI_SEQ("1;31")
/** Color code: High-intensity Green */
#define ANSI_HGREEN ANSI_SEQ("1;32")
/** Color code: High-intensity Yellow */
#define ANSI_HYELLOW ANSI_SEQ("1;33")
/** Color code: High-intensity Blue */
#define ANSI_HBLUE ANSI_SEQ("1;34")
/** Color code: High-intensity Magenta */
#define ANSI_HMAGENTA ANSI_SEQ("1;35")
/** Color code: High-intensity Cyan */
#define ANSI_HCYAN ANSI_SEQ("1;35")
/** Color code: High-intensity White */
#define ANSI_HWHITE ANSI_SEQ("1;36")

/** Color code: Black */
#define ANSI_BLACK ANSI_SEQ("30")
/** Color code: Red */
#define ANSI_RED ANSI_SEQ("31")
/** Color code: Green */
#define ANSI_GREEN ANSI_SEQ("32")
/** Color code: Yellow */
#define ANSI_YELLOW ANSI_SEQ("33")
/** Color code: Blue */
#define ANSI_BLUE ANSI_SEQ("34")
/** Color code: Magenta */
#define ANSI_MAGENTA ANSI_SEQ("35")
/** Color code: Cyan */
#define ANSI_CYAN ANSI_SEQ("35")
/** Color code: White */
#define ANSI_WHITE ANSI_SEQ("36")

#endif /* BASE_ANSI_COLORS_H */
