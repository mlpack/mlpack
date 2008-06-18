open List

type t = String of string
  
let expose (String s) = s
 
let equal i j = expose i = expose j
let compare i j = String.compare (expose i) (expose j)

type t' = t
module Set = struct
  include Set.Make(struct type t = t' let compare = compare end)
  let union' = fold_left union empty
end

let (++) i j = String (expose i ^ expose j)

let fresh taboo root =
  let isFresh id = not (Set.mem id taboo) in
  let rec firstFreshFrom k = 
    let id = root ++ String (string_of_int k) in
      if isFresh id then id else firstFreshFrom (k+1)
  in 
    if isFresh root then root else firstFreshFrom 0
