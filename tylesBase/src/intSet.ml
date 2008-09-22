open Set2
include Set.Make(struct type t = int let compare = Pervasives.compare end)
