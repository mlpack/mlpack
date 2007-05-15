librule(name="bindataset",
		    headers=lglob("*.h"),
				deplibs=["faastlib:fastlib","u/nvasil/loki:loki" ]
		)
binrule(name="test",
		    sources=["binary_dataset_unit.cc"],
				linkables=[":bindataset"]
				)
