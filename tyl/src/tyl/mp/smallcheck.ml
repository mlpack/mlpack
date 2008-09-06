open Util

type 'a series = int -> 'a list 

let return x = fun d -> if d<=0 then [] else [x]
let lift f a = map f % a
let (++) a b d = a d @ b d
let (><) a b d = if d<=0 then [] else foreach (a $ d-1) $ fun x -> foreach (b $ d-1) $ fun y -> [(x,y)]
let (>>) x f = f x

(* constructors as functions *)
let some x = Some x
let cons (x,xs) = x::xs

(* series generators for common types *)
let ints        d = enumFromTo (-d) d
let floats      d = assert false
let units       d = d >> return () 
let bools       d = d >> return true ++ return false
let options a   d = d >> return None ++ lift some a 
let rec lists a d = d >> return [] ++ lift cons (a >< lists a)

(* printing *)
let show_pair a b (x,y) = "(" ^ a x ^ "," ^ b y ^ ")"
let show_option a x = match x with Some y -> "(Some " ^ a y ^")" | None -> "None" 
let show_list a xs = "[" ^ String.concat "," (List.map a xs) ^ "]"
let show_int = string_of_int
let show_unit = const "()"

(* evaluate the proposition up to the specifies depth parameter *)
let evaluate d prop = prop d

(* given a series, a property, and a depth parameter, attempts to find a counterexample *) 
let forAll a f = fun d -> try Some (List.find (not % f) (a d)) with Not_found -> None

(* testing *)
let rev = List.rev

 ;; Printf.printf "Counterexample: %s\n\n" % show_option (show_list show_int) % evaluate 5 $ forAll (lists ints) (fun xs -> (rev % rev) xs = xs)  
 ;; Printf.printf "%s\n\n" % show_list (show_option (show_option (show_unit))) $ (options (options units)) 3 
 ;; Printf.printf "%s\n\n" % show_list (            (show_option (show_unit))) $ (        (options units)) 2 
 ;; Printf.printf "%s\n\n" % show_list (show_list   (show_list   (show_unit))) $ (lists   (lists   units)) 3
 ;; Printf.printf "%s\n\n" % show_list (            (show_list   (show_unit))) $ (        (lists   units)) 3  
 ;; Printf.printf "%s\n\n" % show_list (            (show_list   (show_int ))) $ (        (lists    ints)) 3  


