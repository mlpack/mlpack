type t = Pos of string * int * int | Eof of string
type range = t * t

let toString p = 
  match p with
    | Pos (f,l,c) ->
        let s = string_of_int l ^ "." ^ string_of_int c
        in if f = "" then s else f ^ ":" ^ s
    | Eof f ->
        let fstr = if f = "" then "file" else f
        in "end of " ^ fstr
             
let toStringR (p,q) =
  let ans = toString p 
  in match p,q with
    | Pos (pf,pl,pc), Pos (qf,ql,qc) ->
        if pf <> qf then
          ans ^ "-" ^ toString q
        else if pl <> ql || pc <> qc then
          ans ^ "-" ^ string_of_int ql ^ "." ^ string_of_int qc
        else
          ans
    | Eof pf, Eof qf -> if pf = qf then ans else ans ^ "-" ^ toString q
    | _ -> ans ^ "-" ^ toString q
        
let pos f l c = Pos (f,l,c)
let eof f = Eof f

(* Post: Return true if p1 comes before p2.
 *       Always true if positions in different files.
 * NOTE: Not a true order relation. It is possible that both
 * less(p1,p2) and less(p2,p1) are true. *)
let less p q : bool =
  match p,q with
    | Pos (pf,pl,pc), Pos (qf,ql,qc) ->
        if pf <> qf then           true
        else if pl < ql then       true
        else if pl > ql then       false
        else if pc <= qc then      true
        else                       false
    |  Pos _  , Eof _           -> true
    |  Eof pf , Pos (qf,_,_)    -> pf <> qf
    |  Eof _  , Eof _           -> true

let range p q =
  if less p q then (p,q)
  else raise (Failure ("Invalid range: " ^ toString p ^ " comes after " ^ toString q))

let left = fst
let right = snd

let union (p,q) (p',q') =
  match less p p', less q q' with
    | true  , true   -> p  , q'
    | true  , false  -> p  , q 
    | false , true   -> p' , q'
    | false , false  -> p' , q 

let uniono r s =
  match r,s with
    | Some r' , Some s' -> Some (union r' s')
    | Some _  , None    -> r
    | None    , Some _  -> s
    | None    , None    -> None
