#to calculate DCT 


#librule(
# 	name="DCT",
#   sources=["DCT.cc"],
#	headers=["matrix.h"],
#	deplibs = ["fastlib:fastlib"],
#	tests=[]
#)


 binrule(
     name="DCT",
     sources=["DCT.cc"],
     deplibs = ["fastlib:fastlib"]
)
