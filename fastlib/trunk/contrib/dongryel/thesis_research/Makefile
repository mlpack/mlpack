## Regular expression matching source files
SRC_PATTERN = ".*\.\(cc\|hh\|[ch]\(pp\|xx\|\+\+\)?\)"
## All source files built by this project
SOURCES     = ${shell find core mlpack physpack -iregex ${SRC_PATTERN}}

## Regular expression matching temporary files
TMP_PATTERN = ".*\(\.orig\|\.bak\|~\|/\#[^/]*\#\|/\.\#[^/]*\)"
TEMPORARIES = ${shell find . -iregex ${TMP_PATTERN}}

## Call AStyle on all sources
#
# Standard options are:
#   -s [--indent=spaces=] 2
#   -a [--brackets=attach]
#   -y [--brackets=break-closing]
#   -C [--indent-classes]
#   -S [--indent-switches]
#   -K [--indent-cases]
#   -w [--indent-preprocessor]
#   -p [--pad=oper]
#   -U [--unpad=paren]
#   -v [--convert-tabs]
style:
	astyle -s2 -ay -CSKw -pU -v ${SOURCES}

## Delete Emacs backup/autosave files thoughout the project
clean-src:
	rm -f ${TEMPORARIES}
