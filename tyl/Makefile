######################################################################
## System directories and programs.
LIB_PKGS=num unix str extlib pcre netsys netstring #findlib package name
LIB_NAMES=nums unix str extLib pcre netsys netstring
LIBS=$(patsubst %,%.cma,$(LIB_NAMES))

# Source code directories to include in ocamlbuild's search path
DIRS=base bioCaml

# Given a library package name, return the directory in which the library exists
lib_inc = $(shell ocamlfind query -format "%d" $(1))

# Given a directory, return list of cmo files in order in which they should be loaded
get_deps = $(patsubst $(1)/%,%,$(shell ocamldsort -I $(1) -byte $(1)/*.ml))

INCLUDE=$(foreach lib,$(LIB_PKGS),-I $(call lib_inc,$(lib))) $(patsubst %,-I src/_build/%,$(DIRS)) -I src/_build

.PHONY: all doc clean

all:
	make -C lib/tylesBase all
	cd src; ocamlbuild bioCaml.cma
	cd src; ocamlbuild bioCaml.cmxa

doc/html/%: src/%/*.mli
	rm -rf $@
	mkdir -p $@
	ocamldoc $(INCLUDE) -d $@ -no-stop -html $^

doc: doc/html/base doc/html/bioCaml

clean:
	make -C lib/tylesBase clean
	cd src; ocamlbuild -clean
	rm -rf doc/html
