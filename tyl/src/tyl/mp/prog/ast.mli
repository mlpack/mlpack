open Util

type nullOp = Rational of float | Int of int | Bool of bool
type unaryOp = Neg | Not | Proj of int
type binaryOp = Plus | Minus | Mult | Or | And
    (*    type indexOP = SUM *)
type numRel = Equal | Lte | Gte
type propOp = Disj | Conj
    (*    type propOP = DISJ | CONJ *)
type quant = Exists | Forall
type direction = Min | Max
    
type typ = 
  | TVar of Id.t
  | TReal
      (*| TIntervalr of XRat.interval *)
      (*| TIntervali of XInt.interval *)
  | TBool
  | TBoolVal of bool
      (*		 | TArrowi of Id.t * I.typ * typ *)
  | TArrow of typ * typ
  | TProduct of typ list
  | TLet of (Id.t * syntax) list * typ
  | TMarked of Pos.range * typ
      
and expr = 
  | EVar of Id.t
  | EConst of nullOp
  | EUnaryOp of unaryOp * expr
  | EBinaryOp of binaryOp * expr * expr
      (* 	   | EIndexOP of indexOP * Id.t * I.typ * expr *)
      (*		  | ECase of (Id.t * typ) * I.expr * expr AbsMap.map *)
  | ELambdai of Id.t * expr
      (*		  | EApplyi of expr * I.expr *)
  | ELambda of Id.t * expr
  | EApply of expr * expr
  | ETuple of expr list
  | EAscription of expr * typ
  | ELet of (Id.t * syntax) list * expr
  | EMarked of Pos.range * expr
      
and prop_typ = 
  | ZProp
      (*		      | ZArrowi of Id.t * I.typ * prop_typ *)
  | ZArrow of typ * prop_typ
  | ZMarked of Pos.range * prop_typ

and prop = 
  | CVar of Id.t
  | CBoolVal of bool
  | CIsTrue of expr
  | CNumRel of numRel * expr * expr
  | CPropOp of propOp * prop list
  | CQuant of quant * Id.t * typ * prop
      (*		  | CPropOP of propOP * Id.t * I.typ * prop *)
      (*		  | CCase of (Id.t * prop_typ) * I.expr * prop AbsMap.map *)
  | CLambdai of Id.t * prop
      (*		  | CApplyi of prop * I.expr *)
  | CLambda of Id.t * prop
  | CApply of prop * expr
  | CAscription of prop * prop_typ
  | CLet of (Id.t * syntax) list * prop
  | CMarked of Pos.range * prop
      
and prog = 
  | PMain of direction * ((Id.t * typ) list) * expr * prop
  | PLet of (Id.t * syntax) list * prog | PMarked of Pos.range * prog
      
and syntax = 
  | STyp of typ | SExpr of expr | SProp of prop | SPropTyp of prop_typ 
  | SProg of prog (* | SInd of I.syntax *)
      
type context = string
type category = Typ | Expr | Prop | PropTyp | Prog (* | ICat of I.category *)
