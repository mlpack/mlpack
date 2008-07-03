open List

type t = string
  
let make s = s
let toString s = s 
let make s = s
let equal i j = i = j

module Set = struct
  include Set.Make(String)
  let union' = fold_left union empty
  let addAll xs set = fold_left (fun s x -> add x s) set xs
end

(* return a new name not in 'taboo' *)
let fresh taboo prefix =
  let isFresh id = not (Set.mem id taboo) in
  let rec firstFreshFrom k = 
    let id = prefix ^ string_of_int k in
      if isFresh id then id else firstFreshFrom (k+1)
  in 
    (* if isFresh prefix then prefix else firstFreshFrom 0 *)
    firstFreshFrom 0

(* returns a list of 'n' new names using 'prefix' *)
let rec fresh' n taboo prefix = assert (n >= 0) ;
  if n = 0 then [] else 
    let id = fresh taboo prefix in
      id :: fresh' (n-1) (Set.add id taboo) prefix

(* *)
let rec fresh'' n taboo prefixes = match prefixes with
  | [] -> []
  | prefix::rest -> 
      let ids = fresh' n taboo prefix in
        ids :: fresh'' n (Set.addAll ids taboo) rest
