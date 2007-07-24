/* cp_string.h */

#ifndef CP_STRING_HDR
#define CP_STRING_HDR

/* constants */

/* typedefs */

/* macros */

/* function prototypes */
#ifdef __STDC__

#ifndef streq
#define streq(a, b) (strcmp((a), (b)) == 0)
#endif

/*
#if (CP_STRDUP_DEFINED == CP_STRDUP_IN_CP_STRING_H)
char *strdup(char *s);
#endif
*/

#endif

/*
#if (CP_STRDUP_DEFINED == CP_STRDUP_IN_CP_STRING_H)
char *strdup();
#endif
*/

#endif
