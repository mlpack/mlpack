open Ast
open Format
open Util

(* naming convention: pp<typename> *)

let pptyp t = match t with
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

let rec ppexpr e = match e with
  | EVar x -> Id.toString x
  | EConst (Bool b) -> sprintf "%b" b
  | EConst (Real r) -> sprintf "%f" r
  | EConst (Int n) -> sprintf "%d" n
  | EUnaryOp (op,e') -> 
      sprintf "%s%s" (match op with Neg -> "-" | Not -> "not ") (ppexpr e')
  | EBinaryOp (op,e1,e2) ->
      sprintf "(@[%s %s@ %s@])"
        (ppexpr e1) 
        (match op with Plus -> "+" | Minus -> "-" | Mult -> "*" | Or -> "||" | And -> "&&")
        (ppexpr e2)

let ppcontext = String.concat "\n" % map (fun (x,t) -> sprintf "%s:%s" (Id.toString x) (pptyp t)) 

let rec ppprop c = match c with
  | CBoolVal b -> if b then "T" else "F"
  | CIsTrue e -> sprintf "(@[isTrue %s@])" (ppexpr e)
  | CNumRel (op,e1,e2) ->
      sprintf "(@[%s %s %s@])" 
        (ppexpr e1) 
        (match op with Lte -> "<=" | Equal -> "==" | Gte -> ">=")
        (ppexpr e2)
  | CPropOp (op,cs) -> 
      let opstr = match op with Disj -> " disj " | Conj -> "\n conj " in
        sprintf "(@[%s@])" (String.concat opstr (map ppprop cs))
  | CQuant (Exists,x,t,c) -> sprintf "(@[exists %s:%s .@ %s@])" (Id.toString x) (pptyp t) (ppprop c)

let ppprog p = match p with
  | PMain (d,ctxt,e,c) -> 
      let vars = ppcontext ctxt in 
      let dirxn = (match d with Min -> "min" | Max -> "max") in
        sprintf "%s\n\n%s %s subject_to\n\n%s" vars dirxn (ppexpr e) (ppprop c)
