(*
i - index
e - expr
t - type
p - prog
s - syntax
c - prop
z - prop type
x - variable
xx,cc - plural (sets/lists of things)
*)
open Ast

module Id = Util.Id
module S = Util.Id.Set
let map = List.map
let fold = List.fold_left
let empty = S.empty
let single = S.singleton
let union = fold S.union S.empty
let fail s = raise (Failure s)
let (--) = S.diff

let rec unzip = function
  | [] -> ([],[])
  | (a,b)::xx -> let (aa,bb) = unzip xx in (a::aa,b::bb)

let fromList xx = fold (fun s x -> S.add x s) empty xx
let namesIn xx = fromList (fst (unzip xx))

let rec freeVarse = function
  | EVar x              -> single x
  | EConst _            -> empty
  | EUnaryOp (_,e)      -> freeVarse e
  | EBinaryOp (_,e1,e2) -> union [freeVarse e1; freeVarse e2]
  | ELambdai (x,e)      -> freeVarse e -- single x
  | ELambda (x,e)       -> freeVarse e -- single x
  | EApply (e1,e2)      -> union [freeVarse e1; freeVarse e2]
  | ETuple ee           -> union (map freeVarse ee)
  | ELet (xx,e)         -> freeVarse e -- namesIn xx
  | EMarked (_,e)       -> freeVarse e
  | _                   -> fail "FIXME"

let rec freeVarsc = function
  | CVar x            -> single x
  | CBoolVal _        -> empty
  | CIsTrue e         -> freeVarse e
  | CNumRel (_,e1,e2) -> union [freeVarse e1; freeVarse e2]
  | CPropOp (_,cc)    -> union (map freeVarsc cc)
  | CQuant (_,x,t,c)  -> freeVarsc c -- single x
  | CLambdai (x,c)    -> freeVarsc c -- single x
  | CLambda (x,c)     -> freeVarsc c -- single x
  | CLet (xx,c)       -> freeVarsc c -- namesIn xx
  | CMarked (_,c)     -> freeVarsc c
  | _                 -> fail "FIXME"

let rec freeVarsp = function
  | PMain (_,xx,e,c) -> union [freeVarse e; freeVarsc c] -- namesIn xx
  | PLet (xx,p)      -> freeVarsp p -- namesIn xx
  | PMarked (_,p)    -> freeVarsp p

let rec freeVarsz = function
  | ZProp         -> empty
  | ZMarked (_,z) -> freeVarsz z
  | _             -> fail "FIXME"

let rec freeVarst t = empty

let rec freeVars = function
  | STyp t     -> freeVarst t
  | SExpr e    -> freeVarse e
  | SProp c    -> freeVarsc c
  | SPropTyp z -> freeVarsz z
  | SProg p    -> freeVarsp p

let isClosed s = S.is_empty (freeVars s)

let rec subee e x = function 
  | EVar x' as e'        -> if Id.equal x x' then e else e'
  | EConst _ as e'       -> e'
  | EUnaryOp (op,e')     -> EUnaryOp (op, subee e x e')
  | EBinaryOp (op,e1,e2) -> EBinaryOp (op, subee e x e1, subee e x e2)
  | EMarked (range,e')   -> EMarked (range, subee e x e')
  | _ -> fail "FIXME"

let rec subec e x = function 
  | CBoolVal _ as c          -> c
  | CIsTrue e'               -> CIsTrue (subee e x e')
  | CNumRel (op,e1,e2)       -> CNumRel (op, subee e x e1, subee e x e2)
  | CPropOp (op,cc)          -> CPropOp (op, map (subec e x) cc)
  | CQuant (q,x',t,c) as c'  -> 
      if not (S.mem x (freeVarsc c'))       then c' 
      else if not (S.mem x' (freeVarse e))  then CQuant (q,x',t,subec e x c)
      else let x'' = Id.fresh (union [freeVarse e; freeVarsc c]) (Id.make "x") in 
        subec e x (CQuant (q, x'', t, subec (EVar x'') x' c))
  | _                        -> fail "FIXME"
