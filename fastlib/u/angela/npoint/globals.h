/**
 * @author: Angela N Grigoroaia
 * @date: 12.06.2007
 * @file: globals.h
 *
 * @description:
 * Definitions for global variables and constants. Currently most of the stuff
 * here  is just for show.
 */


#define INCLUDE 10
#define EXCLUDE -10
#define RECURSE 0

#define LEAF_SIZE 20
#ifndef LEAF_SIZE
	extern int leaf_size;
#endif

extern int use_permutes;
extern int use_symmetry;
extern int nweights;
extern char *format;
