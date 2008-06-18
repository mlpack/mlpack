open Ast
open List

module Id = Util.Id
module S = Util.Id.Set
let single = S.singleton
let (--) = S.diff

let set_of_list xs = fold_left (fun s x -> S.add x s) S.empty xs
let namesIn xts = set_of_list (fst (split xts))

let rec freeVarse e = match e with 
  | EVar x              -> single x
  | EConst _            -> S.empty
  | EUnaryOp (_,e')     -> freeVarse e'
  | EBinaryOp (_,e1,e2) -> S.union' [freeVarse e1; freeVarse e2]

let rec freeVarsc c = match c with 
  | CBoolVal _        -> S.empty
  | CIsTrue e         -> freeVarse e
  | CNumRel (_,e1,e2) -> S.union' [freeVarse e1; freeVarse e2]
  | CPropOp (_,cs)    -> S.union' (map freeVarsc cs)
  | CQuant (_,x,_,c') -> freeVarsc c' -- single x

let freeVarsp p = match p with 
  | PMain (_,xts,e,c) -> S.union' [freeVarse e; freeVarsc c] -- namesIn xts

let isClosede e = S.is_empty (freeVarse e)
let isClosedc c = S.is_empty (freeVarsc c)
let isClosedp p = S.is_empty (freeVarsp p)

let rec subee e x e' = match e' with 
  | EVar x'              -> if Id.equal x x' then e else e'
  | EConst _             -> e'
  | EUnaryOp (op,e'')    -> EUnaryOp (op, subee e x e'')
  | EBinaryOp (op,e1,e2) -> EBinaryOp (op, subee e x e1, subee e x e2)

let rec subec e x c = match c with 
  | CBoolVal _               -> c
  | CIsTrue e'               -> CIsTrue (subee e x e')
  | CNumRel (op,e1,e2)       -> CNumRel (op, subee e x e1, subee e x e2)
  | CPropOp (op,cs)          -> CPropOp (op, map (subec e x) cs)
  | CQuant (q,x',t,c')       -> 
      if not (S.mem x (freeVarsc c'))       then c 
      else if not (S.mem x' (freeVarse e))  then CQuant (q,x',t,subec e x c')
      else let x'' = Id.fresh (S.union' [freeVarse e; freeVarsc c']) x' in 
        subec e x (CQuant (q, x'', t, subec (EVar x'') x' c'))
