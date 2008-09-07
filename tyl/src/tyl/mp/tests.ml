open Smallcheck
open Ast
open Wf
open Show 
open Util

let bool x = Bool x
let int x = Int x
let real x = Real x

let nullOps = pure bool %% bools ++ pure int %% ints ++ pure real %% floats
let unaryOps = pure Neg ++ pure Not
let binaryOps = pure Plus ++ pure Minus ++ pure Mult ++ pure Or ++ pure And
let numRels = pure Equal ++ pure Lte ++ pure Gte
let propOps = pure Disj ++ pure Conj
let quants = pure Exists
let directions = pure Min (* ++ pure Max *)

let discrete x = Discrete x
let continuous x = Continuous x

let intervals a = pairs (options a) (options a)
let refinedReals = pure discrete %% intervals ints ++ pure continuous %% intervals floats
let refinedBools = options bools

let treal x = TReal x
let tbool x = TBool x

let evar x = EVar x
let econst x = EConst x
let eunaryop x y = EUnaryOp (x,y)
let ebinaryop x y z = EBinaryOp (x,y,z)

let cboolval x = CBoolVal x
let cistrue x = CIsTrue x
let cnumrel x y z = CNumRel (x,y,z)
let cpropop x y = CPropOp (x,y)
let cquant w x y z = CQuant (w,x,y,z)
let pmain w x y z = PMain (w,x,y,z)

let ids = pure (Id.make "a") ++ pure (Id.make "b") ++ pure (Id.make "c") 

let (>-) x f = f x

let typs = pure treal %% refinedReals ++ pure tbool %% refinedBools

let rec exprs = fun n -> n >-
     pure evar %% ids 
  ++ pure econst %% nullOps 
  ++ pure eunaryop %% unaryOps %% exprs
  ++ pure ebinaryop %% binaryOps %% exprs %% exprs

let rec props = fun n -> n >-
     pure cboolval %% bools 
  ++ pure cistrue %% exprs 
  ++ pure cnumrel %% numRels %% exprs %% exprs
  ++ pure cpropop %% propOps %% lists props 
  ++ pure cquant %% quants %% ids %% typs %% props

let progs = pure pmain %% directions %% lists (pairs ids typs) %% exprs %% props

let contexts = lists (pairs ids typs)

(* printing *)
let show_unit x = "()"
let show_bool x = if x then "true" else "false"
let show_pair a b (x,y) = "(" ^ a x ^ "," ^ b y ^ ")"
let show_option a x = match x with Some y -> "(Some " ^ a y ^")" | None -> "None" 
let show_int x = string_of_int x
let show_float x = string_of_float x
let show_list a xs = "[" ^ String.concat "," (List.map a xs) ^ "]" 

;; Printf.printf "%s" % show_option showt $ evaluate 5 (forAll typs isType)
