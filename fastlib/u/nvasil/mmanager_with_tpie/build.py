binrule(
		name="test",
		sources=["memory_manager_unit.cc"],
		headers=lglob("*.h"),
		cflags="-I../tpie -Lsigsegv",
		linkables=["fastlib:fastlib",
							 "u/nvasil/tpie:tpie"]
		);
