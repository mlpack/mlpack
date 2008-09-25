open TylesBase
open Ast
open Vars

let rec containsQuantifier c = match c with
  | CQuant _ -> true
  | CPropOp (_,cs) -> List.exists containsQuantifier cs
  | _ -> false
      
let rec isPNF c = match c with
  | CIsTrue _ -> true
  | CBoolVal _ -> true
  | CNumRel _ -> true
  | CQuant (_,_,_,c') -> isPNF c'
  | CPropOp (_,cs) -> not (List.exists containsQuantifier cs)

let rec toPNF c = match c with 
  | CIsTrue _ -> assert false
  | CBoolVal _ -> c
  | CNumRel _ -> c
  | CQuant (Exists,x,t,c') -> CQuant (Exists,x,t,toPNF c')
  | CPropOp (op,cs) -> 
      let rec mergePNF op c1 c2 = (* pre: c1 and c2 are PNF *)
        match c1,c2 with
          | CQuant(Exists,x,t,c1'), _ -> CQuant(Exists,x,t,mergePNF op c1' c2)
          | _, CQuant(q,x,t,c2') -> 
              let CQuant(_,x',_,_) as c2'' = alphaConvert c2' (freeVarsc c1) 
              in CQuant(q,x',t,mergePNF op c1 c2'')
          | _,_ -> CPropOp(op,[c1;c2])
      in 
      let fold_left' f (x::xs) = List.fold_left f x xs in
        fold_left' (mergePNF op) (List.map toPNF cs)

open Tests
open PPrint
open TylesBase.SmallCheck
module X = Printexc

;; fold boolExprs 3 () (fun _ -> Printf.printf "%s\n" <<- ppexpr)

;; let result = forAll props (isPNF <<- toPNF) 4 in 
  match result with 
    | None -> Printf.printf "None\n"
    | Some (x,e) -> Printf.printf "%s , %s\n" (ppprop x) (X.to_string e)
