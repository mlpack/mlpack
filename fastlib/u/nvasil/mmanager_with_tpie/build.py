librule(name="tpiemm",
		    headers=["memory_manager.h"],
				deplibs=["u/nvasil/tpie:tpie", "libsigsegv.a"]
		);
binrule(
		name="test",
		sources=["memory_manager_unit.cc"],
		headers=lglob("*.h"),
		#cflags="-I../tpie -lsigsegv",
		#linkables=["fastlib:fastlib",
		#					 "u/nvasil/tpie:tpie",
		#					 "libsigsegv.a"]
		linkables=["fastlib:fastlib", ":tpiemm"]
		);
