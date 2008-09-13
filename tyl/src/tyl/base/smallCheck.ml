open Util

(* possible TODOs: 
    make it a lazy list instead to allow let recs (?)
    make 'a series ordered strictly by depth values
*)

type 'a series = {fold : 'b . int -> ('b -> 'a -> 'b) -> 'b -> 'b}

let pure a      = {fold = fun n f b -> if n<=0 then b else f b a}
let (++) sa sa' = {fold = fun n f b -> sa'.fold n f (sa.fold n f b)}
let (%%) sf sa  = {fold = fun n g c -> sf.fold n (fun c1 f -> sa.fold (n-1) (fun c2 a -> (g c2 $ f a)) c1) c}

(* series generators for common types *)
let units       = pure ()  
let bools       = pure false ++ pure true
let pairs a b   = pure pair %% a %% b
let options a   = pure None ++ pure some %% a  
let ints        = {fold = fun n f b -> List.fold_left f b $ enumFromTo (-n) n}
let floats      = {fold = fun n f b -> List.fold_left f b $ List.map float_of_int (enumFromTo (-n) n)}
let rec lists a = {fold = fun n f b -> (pure [] ++ pure cons %% a %% lists a).fold n f b}

exception Counterexample_found

(* given a series, a property, and a depth parameter, attempts to find a counterexample *) 
let forAll sa p n = 
  let curr = ref None in
  let test () a = curr := Some a ; if not (p a) then raise Counterexample_found else () in
    try sa.fold n test () ; None with e -> let Some c = !curr in Some (c,e)

open Show 

;; (lists bools).fold 4 (fun _ -> Printf.printf "%s\n" % show_list show_bool) ()
