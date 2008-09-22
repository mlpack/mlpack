open Map2
include Map.Make(struct type t = int let compare = Pervasives.compare end)
