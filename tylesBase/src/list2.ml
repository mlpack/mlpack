include List (* ExtList.List does not include all functions from standard List *)
include ExtList.List

let zip = combine
let unzip = split

let zip3 al bl cl =
  let rec loop ans al bl cl =
    match (al,bl,cl) with
      | ([], [], []) -> ans
      | (a::al, b::bl, c::cl) -> loop ((a,b,c)::ans) al bl cl
      | _ -> raise (Different_list_size "zip3")
  in rev (loop [] al bl cl)
  
let zip4 al bl cl dl =
  let rec loop ans al bl cl dl =
    match (al,bl,cl,dl) with
      | ([], [], [], []) -> ans
      | (a::al, b::bl, c::cl, d::dl) -> loop ((a,b,c,d)::ans) al bl cl dl
      | _ -> raise (Different_list_size "zip4")
  in rev (loop [] al bl cl dl)
  
let unzip3 abcl =
  let rec loop (al,bl,cl) abcl =
    match abcl with
      | [] -> (al,bl,cl)
      | (a,b,c)::abcl -> loop (a::al, b::bl, c::cl) abcl
  in
  let (al,bl,cl) = loop ([],[],[]) abcl in
    rev al, rev bl, rev cl

let unzip4 abcdl =
  let rec loop (al,bl,cl,dl) abcdl =
    match abcdl with
      | [] -> (al,bl,cl,dl)
      | (a,b,c,d)::abcdl -> loop (a::al, b::bl, c::cl, d::dl) abcdl
  in
  let (al,bl,cl,dl) = loop ([],[],[],[]) abcdl in
    rev al, rev bl, rev cl, rev dl

let npartition eq l =
  let insertl ll a =
    let rec loop prefix ll =
      match ll with
        | [] -> rev ([a]::prefix)
        | l::ll ->
            if eq a (hd l)
            then (rev ((a::l)::prefix)) @ ll 
            else loop (l::prefix) ll
    in loop [] ll
  in map rev (fold_left insertl [] l)
    
let interleave al1 al2 =
  let rec iter ans al1 al2 =
    match (al1,al2) with
	(_,[]) -> ans @ al1
      | ([],_) -> ans @ al2
      | (a1::al1, a2::al2) -> iter (ans @ [a1;a2]) al1 al2
  in iter [] al1 al2
    
let to_string f l =
  "[" ^ (String.concat "; " (List.map f l)) ^ "]"

let elements_unique ?(cmp = (=)) l =
  length l = length (unique ~cmp l)

let first_repeat ?(cmp = (=)) l =
  let rec loop prev rest =
    match rest with
      | [] -> raise Not_found
      | r::rest -> if exists (cmp r) prev then r else loop (r::prev) rest
  in loop [] l

let set_assoc_with f a l =
  let rec loop prevl l =
    match l with
      | [] -> rev ((a, f None)::prevl)
      | (a',b')::l ->
          if a = a'
          then (rev prevl) @ ((a, f (Some b'))::l)
          else loop ((a',b')::prevl) l
  in loop [] l

let is_sorted ?(cmp = Pervasives.compare) l =
  let rec loop l =
    match l with
      | [] | _::[] -> true
      | x::y::l -> if cmp x y <= 0 then loop (y::l) else false
  in loop l
