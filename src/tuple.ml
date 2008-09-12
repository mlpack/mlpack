module Pr = struct
  let make a b = (a,b)
  let prj1 (a,_) = a
  let prj2 (_,b) = b
  let map fa fb (a,b) = (fa a, fb b)
  let map1 f (a,b) = (f a, b)
  let map2 f (a,b) = (a, f b)
  let curry f a b = f(a,b)
  let uncurry f (a,b) = f a b
end

module Tr = struct
  let make a b c = (a,b,c)
  let prj1 (a,_,_) = a
  let prj2 (_,b,_) = b
  let prj3 (_,_,c) = c
  let prj12 (a,b,_) = (a,b)
  let prj13 (a,_,c) = (a,c)
  let prj23 (_,b,c) = (b,c)
  let map fa fb fc (a,b,c) = (fa a, fb b, fc c)
  let map1 f (a,b,c) = (f a, b, c)
  let map2 f (a,b,c) = (a, f b, c)
  let map3 f (a,b,c) = (a, b, f c)
  let curry f a b c = f(a, b, c)
  let uncurry f (a,b,c) = f a b c
end

module Fr = struct
  let make a b c d = (a,b,c,d)
  let prj1 (a,_,_,_) = a
  let prj2 (_,b,_,_) = b
  let prj3 (_,_,c,_) = c
  let prj4 (_,_,_,d) = d
  let prj12 (a,b,_,_) = (a,b)
  let prj13 (a,_,c,_) = (a,c)
  let prj14 (a,_,_,d) = (a,d)
  let prj23 (_,b,c,_) = (b,c)
  let prj24 (_,b,_,d) = (b,d)
  let prj34 (_,_,c,d) = (c,d)
  let prj123 (a,b,c,_) = (a,b,c)
  let prj124 (a,b,_,d) = (a,b,d)
  let prj234 (_,b,c,d) = (b,c,d)
  let map fa fb fc fd (a,b,c,d) = (fa a, fb b, fc c, fd d)
  let map1 f (a,b,c,d) = (f a, b, c, d)
  let map2 f (a,b,c,d) = (a, f b, c, d)
  let map3 f (a,b,c,d) = (a, b, f c, d)
  let map4 f (a,b,c,d) = (a, b, c, f d)
  let curry f a b c d = f(a,b,c,d)
  let uncurry f (a,b,c,d) = f a b c d
end
