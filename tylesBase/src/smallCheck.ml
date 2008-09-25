type 'a t = {fld : 'b . int -> ('b -> 'a -> 'b) -> 'b -> 'b}
type 'a series = unit -> 'a t
type 'a tester = int -> ('a * exn) option

let fold sa n b f = (sa()).fld n f b

let rec enumFromTo a b = if a==b then [a] else a :: enumFromTo (a+1) b

let pure a      = fun () -> {fld = fun n f b -> if n<=0 then b else f b a}
let (++) sa sa' = fun () -> {fld = fun n f b -> fold sa' n (fold sa n b f) f}
let (%%) sf sa  = fun () -> {fld = fun n g c -> fold sf n c (fun c1 f -> fold sa (n-1) c1 (fun c2 a -> g c2 (f a)))}

(* series generators for common types *)
let units       = pure ()  
let bools       = pure false ++ pure true
let pairs a b   = pure (fun x y -> (x,y)) %% a %% b
let options a   = pure None ++ pure (fun x -> Some x) %% a
let ints        = fun () -> {fld = fun n f b -> List.fold_left f b (enumFromTo (-n) n)}
let floats      = fun () -> {fld = fun n f b -> List.fold_left f b (List.map float_of_int (enumFromTo (-n) n))}
let rec lists a = fun () -> ( pure [] ++ pure (fun x xs -> x::xs) %% a %% lists a ) ()

exception Counterexample_found

let forAll sa p n = 
  let curr = ref None in
  let test () a = curr := Some a ; if not (p a) then raise Counterexample_found else () in
    try fold sa n () test ; None with e -> let Some c = !curr in Some (c,e)
