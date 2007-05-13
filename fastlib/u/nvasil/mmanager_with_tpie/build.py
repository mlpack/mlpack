binrule(
		name="test",
		sources=["memory_manager_unit.cc"],
		headers=lglob("*.h"),
		cflags="-I../tpie -lsigsegv",
		linkables=["fastlib:fastlib",
							 "u/nvasil/tpie:tpie",
							 "libsigsegv.a"]
		);
