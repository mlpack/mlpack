open Ast 
open Vars
open List

module C = Ctxt
module S = Util.Id.Set

let isType t = true

let rec isOfType ctxt e t = match e,t with 
  | EVar x                  , _     -> C.contains ctxt x t
  | EConst (Bool _)         , TBool -> true
  | EConst (Real _)         , TReal -> true
  | EUnaryOp (Neg,e')       , TReal -> isOfType ctxt e' TReal
  | EUnaryOp (Not,e')       , TBool -> isOfType ctxt e' TBool
  | EBinaryOp (Plus,e1,e2)  , TReal -> isOfType ctxt e1 TReal && isOfType ctxt e2 TReal
  | EBinaryOp (Minus,e1,e2) , TReal -> isOfType ctxt e1 TReal && isOfType ctxt e2 TReal
  | EBinaryOp (Mult,e1,e2)  , TReal -> isOfType ctxt e1 TReal && isOfType ctxt e2 TReal
  | EBinaryOp (Or,e1,e2)    , TBool -> isOfType ctxt e1 TBool && isOfType ctxt e2 TBool
  | EBinaryOp (And,e1,e2)   , TBool -> isOfType ctxt e1 TBool && isOfType ctxt e2 TBool
  | _ -> false

let rec isProp ctxt c = match c with 
  | CBoolVal _        -> true
  | CIsTrue e         -> isOfType ctxt e TBool
  | CNumRel (_,e1,e2) -> isOfType ctxt e1 TReal && isOfType ctxt e2 TReal
  | CPropOp (_,cs)    -> for_all (isProp ctxt) cs
  | CQuant (_,x,t,c') -> isType t && isProp (C.add ctxt x t) c'

let isMP p = match p with 
  | PMain (_,xts,e,c) -> 
      let ctxt = C.fromList xts in 
        for_all (fun (_,t) -> isType t) xts && isOfType ctxt e TReal && isProp ctxt c

(* The following functions assume 'pre: e is boolean' *)

let rec isLiteral e = match e with
  | EVar _            -> true
  | EConst (Bool _)   -> true
  | EUnaryOp (Not,e') -> isLiteral e'
  | _ -> false

let rec isDLF e = isLiteral e || match e with 
  | EBinaryOp (Or,e1,e2) -> isDLF e1 && isDLF e2
  | _ -> false

let rec isCNF e = isDLF e || match e with 
  | EBinaryOp (And,e1,e2) -> isCNF e1 && isCNF e2
  | _ -> false

let isConj e = isCNF e && not (isDLF e)

let rec toCNF e = 
  let rec toCNF' e' = 
    let (&.) e1 e2 = EBinaryOp(And,e1,e2) in
    let (|.) e1 e2 = EBinaryOp(Or,e1,e2) in
    let nt e1 = EUnaryOp(Not,e1) in
      match e' with 
        | EUnaryOp (Not,EUnaryOp(Not,e1))          -> toCNF e1
        | EUnaryOp (Not,EBinaryOp(Or,e1,e2))       -> toCNF (nt e1 &. nt e2)
        | EUnaryOp (Not,EBinaryOp(And,e1,e2))      -> toCNF (nt e1 |. nt e2)
        | EBinaryOp (Or,EBinaryOp(And,e11,e12),e2) -> toCNF ((e11 |. e2) &. (e12 |. e2))
        | EBinaryOp (Or,e1,EBinaryOp(And,e21,e22)) -> toCNF ((e21 |. e1) &. (e22 |. e1))
        | EBinaryOp (Or,e1,e2)                     -> toCNF (toCNF e1 |. toCNF e2)
        | EBinaryOp (And,e1,e2)                    -> toCNF (toCNF e1 &. toCNF e2)
        | _ -> failwith "should be boolean expressions only"
  in
    if isCNF e then e else toCNF' e

(* *)

let bounded t = true

let rec existVarsBounded c = match c with 
  | CBoolVal _ 
  | CIsTrue _ 
  | CNumRel _              -> true
  | CPropOp (_,cs)         -> for_all existVarsBounded cs
  | CQuant (Exists,_,t,c') -> bounded t && existVarsBounded c'

let rec disjVarsBounded ctxt c = match c with
  | CBoolVal _
  | CIsTrue _
  | CNumRel _              -> true
  | CPropOp (Disj,cs)      -> 
      let freeVars = S.union' (map freeVarsc cs) in
      let ts = map (C.lookup ctxt) (S.elements freeVars) in
        for_all bounded ts && for_all existVarsBounded cs 
  | CPropOp (Conj,cs)      -> for_all (disjVarsBounded ctxt) cs 
  | CQuant (Exists,x,t,c') -> disjVarsBounded (C.add ctxt x t) c'

