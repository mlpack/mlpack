librule(name="bindataset",
		    headers=lglob("*.h"),
				deplibs=["fastlib:fastlib","u/nvasil/loki:loki" ]
		)
binrule(name="test",
		    sources=["binary_dataset_unit.cc"],
				linkables=[":bindataset"]
				)
