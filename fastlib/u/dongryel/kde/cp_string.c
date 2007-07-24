/* cp_string.c */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cp_platform.h"
#include "cp_string.h"

/* machine dependant stuff */ 
/****************************************************************************
 *   function:      strdup
 *   description:   This function determines the length of the given string. 
 *                  Then it creates an allocated string buffer of that size. 
 *                  Followed by the copying of the given string into the new
 *                  buffer.  Then it returns the newly created string.
 *   calls:         strlen
 *                  utMalloc
 *                  strcpy
 *   returns:       newly duplicated string.
 ****************************************************************************/
/* issue 2 */
#ifndef strdup
#if (CP_STRDUP_DEFINED == CP_STRDUP_IN_CP_STRING_H)
char *strdup (s)
	char *s;
	{
	char *p=NULL;
													/* +1 for '\0' */
	if ((p = (char *) malloc((unsigned)strlen(s) + 1)) != (char *)NULL)												
		strcpy(p,s);
	return((char *)p);
	} /* strdup */
#endif
#endif
/***** MACUNIXDOS...MACUNIXDOS...MACUNIXDOS *****/
