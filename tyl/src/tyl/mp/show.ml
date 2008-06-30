open Ast
open Printf
open List

module Id = Util.Id

let showt t = match t with
  | TBool None -> "bool"
  | TBool (Some b) -> sprintf "{%b}" b
  | TReal (Discrete (None,None)) -> "int"
  | TReal (Continuous (None,None)) -> "real"
  | TReal (Discrete (lo,hi)) ->
      let lower = match lo with None -> "-inf" | Some x -> sprintf "%d" x in
      let upper = match hi with None -> "inf" | Some x -> sprintf "%d" x in
        sprintf "[%s,%s]" lower upper
  | TReal (Continuous (lo,hi)) ->
      let lower = match lo with None -> "-inf" | Some x -> sprintf "%f" x in
      let upper = match hi with None -> "inf" | Some x -> sprintf "%f" x in
        sprintf "<%s,%s>" lower upper

let rec showe e = match e with
  | EVar x -> Id.toString x
  | EConst (Bool b) -> sprintf "%b" b
  | EConst (Real r) -> sprintf "%f" r
  | EConst (Int n) -> sprintf "%d" n
  | EUnaryOp (op,e') -> 
      sprintf "(%s%s)" 
        (match op with Neg -> "-" | Not -> "not ") 
        (showe e')
  | EBinaryOp (op,e1,e2) ->
      sprintf "(%s %s %s)" 
        (showe e1) 
        (match op with Plus -> "+" | Minus -> "-" | Mult -> "*" | Or -> "||" | And -> "&&")
        (showe e2)

let showxt (x,t) = sprintf "%s:%s" (Id.toString x) (showt t)

let rec showc c = match c with
  | CBoolVal b -> if b then "T" else "F"
  | CIsTrue e -> sprintf "(isTrue %s)" (showe e)
  | CNumRel (op,e1,e2) ->
      sprintf "(%s %s %s)" 
        (showe e1) 
        (match op with Lte -> "<=" | Equal -> "==" | Gte -> ">=")
        (showe e2)
  | CPropOp (op,cs) -> 
      let opstr = match op with Disj -> " disj " | Conj -> " conj " in
        "(" ^ String.concat opstr (map showc cs) ^ ")"
  | CQuant (Exists,x,t,c) -> sprintf "(exists %s . %s)" (showxt (x,t)) (showc c)

let showp p = match p with
  | PMain (d,xts,e,c) -> 
      let vars = String.concat "\n" (map showxt xts) in 
      let dirxn = (match d with Min -> "min" | Max -> "max") in
        sprintf "%s\n\n%s %s\nsubject_to %s" vars dirxn (showe e) (showc c)
