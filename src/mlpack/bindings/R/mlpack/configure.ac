##  mlpack configure.ac
AC_PREREQ(2.61)

## Process this file with autoconf to produce a configure script.
AC_INIT()

## Set R_HOME, respecting an environment variable if one is set 
: ${R_HOME=$(R RHOME)}
if test -z "${R_HOME}"; then
    AC_MSG_ERROR([Could not determine R_HOME.])   
fi

## Check for Solaris.
RSysinfoName=$("${R_HOME}/bin/Rscript" --vanilla -e 'cat(Sys.info()[["sysname"]])')

## Default the flag to the empty string.
extra_flag=""

if test x"${RSysinfoName}" = x"SunOS"; then
  extra_flag="-ftrack-macro-expansion=0"
fi

AC_SUBST([EXTRA_FLAG], ["${extra_flag}"])
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
