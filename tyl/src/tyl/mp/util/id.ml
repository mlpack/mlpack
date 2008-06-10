type t = String of string | Marked of Pos.range * t
  
let rec expose i : string = match i with String s -> s | Marked (_,j) -> expose j
 
let equal i j = expose i = expose j

type temp = t
module IdOrd = struct 
  type t = temp
  let compare = compare 
end

(* module Map = Map.Make(IdOrd) *)
module Set = Set.Make(IdOrd)


(*
let compare i j = String.compare (expose i) (expose j)
let toString = expose
  
let shorter i j = let len = String.length in 
  if len (expose j) < len (expose i) then j else i
    
let shortest is = 
  match is with [] -> None | j::js -> Some (List.fold_left shorter j js)
       
let getMark i = match i with String _ -> None | Marked (r,_) -> Some r

let rec last xs = match xs with [] -> raise (Failure "") | x::[] -> x | _::xs' -> last xs'

let (++) i j = 
  let s = String (expose i ^ expose j) in 
  let r = Pos.uniono (getMark i) (getMark j) in
    match r with None -> s | Some r' -> Marked (r',s)

(* FIXME ??? *)
let concat is = 
  let ans = String (String.concat "" (List.map toString is)) in
  let getPos js = match js with 
      []    -> None 
    | j::js' -> Pos.uniono (getMark j) (getMark (last js'))
  in match getPos is with None -> ans | Some r -> Marked(r,ans)
        
let fresh _ _ = String ""
let freshl _ _ _ = []
let freshll _ _ _ = []
*)

(* 
let fresh s root =
  let iter k = let newId = root ++ String (string_of_int k) in
    if mem newId s then iter (k+1) else newId
  in if mem root s then iter 0 else root
        
let freshl s root k =
  List.map (fun k -> fresh s (root ++ String (string_of_int k))) (MyInt.int_range(1,k))
    
let freshll s rootl k =
  if !(Global.debug) andalso not (Listops.unique ident_compare rootl) 
  then raise Domain
  else
    let helper (root:id, (s:set, ill : id list list)) : (set * id list list) = 
      let il = freshl(s,root,k) in (union(s,fromList il), ill @ [il])
    in let (_,ans) = List.foldl helper (s,nil) rootl in ans
*)
