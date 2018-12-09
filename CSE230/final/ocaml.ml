(**********CATEGORY 1 complicated data structure question*******)

(* Fall 13, expressions *)
type expr =
  | Var of string
  | Const of int
  | Plus of expr * expr
let rec simpl e =
  (* case for plus: *)
  match e with
  | Plus(e1,e2) -> (let e1' = simpl e1 in let e2' = simpl e2 in 
    match (e1',e2') with 
    | (Const a,Const b) -> Const (a+b)
    | _ -> Plus(e1',e2'))
  (* all other cases: *)
  |_  -> e

(*Winter 12, dictionaries*)
let rec find d k =
  match d with
  | [] -> raise Not_found
  | (k',v') :: t ->  if k' = k then v' else if k'<k then find t k else raise Not_found;;

let rec add d k v =
  match d with
  | [] -> [(k,v)]
  | (k',v') :: t -> if k'>k then (k,v)::d else if k'=k then (k,v)::t else (k',v')::(add t k v)

let keys d = List.map (fun x-> let (k,v) = x in k) d;;
let values d = List.map (fun x-> let (k,v) = x in v) d;;

let key_of_max_val d =
(* definition of fold_fn *)
  let fold_fn (maxK,maxV) (elemK,elemV) = if elemV>maxV then (elemK,elemV) else (maxK,maxV) in
  match d with
  | [] -> raise Not_found
  (* the call to fold should be: fold_left fold_fn base t *)
  | base::t -> let (res,_) = List.fold_left fold_fn base t in res;;

(* Winter 11, more dictionaries *)
type 'a dict = Empty | Node of string * 'a * 'a dict * 'a dict

let fruitd =
Node ("grape", 2.65,
Node ("banana", 1.50,
Node ("apple", 2.25, Empty, Empty),
Node ("cherry", 2.75, Empty, Empty)),
Node ("orange", 0.75,
Node ("kiwi", 3.99, Empty, Empty),
Node ("peach", 1.99, Empty, Empty)))

let rec find d k =
  match d with
  | Empty -> raise Not_found
  | Node (k', v', l, r) ->
    if k = k' then v' else
    if k < k' then find l k else find r k;;

let rec add d k v =
  match d with
  | Empty -> Node(k,v,Empty,Empty)
  | Node (k', v', l, r) ->
      if k = k' then Node(k',v,l,r) else
      if k < k' then let newL = (add l k v) in Node(k',v',newL,r) else 
      let newR = (add r k v) in Node(k',v',l,newR);;

let d0 = fruitd in
let d1 = add d0 "banana" 5.0 in d1
let d2 = add d1 "mango" 10.25 in
(find d2 "banana");;


let rec fold f b d =
  match d with
  | Empty -> b
  | Node (k, v, l, r) -> let newBase = (fold f b l) in let newB = (f k v newBase) in (fold f newB r);;

fold (fun k v b -> b^","^k) "" fruitd
fold (fun k v b -> b +. v) 0.0 fruitd

(* Winter 11, python namespaces *)
type name_space = EmptyNameSpace
|Info of (string * value) list * name_space
and value = Int of int
| Method of (name_space -> int -> int)

(* Spring 13, trees *)
type 'a tree =
| Empty
| Node of 'a * 'a tree list;;

let rec zip l1 l2 =
  match (l1,l2) with
  | ([],[]) -> []
  | (h1::t1, h2::t2) -> (h1,h2)::(zip t1 t2)
  | _ -> raise Mismatch;;


let rec tree_zip t1 t2 =
  match (t1,t2) with
  | (Empty,Empty) -> Empty
  | (Node(r1,l1),Node(r2,l2)) -> Node((r1,r2),(List.map (fun (t1,t2) -> tree_zip t1 t2) (zip l1 l2)))
  | _ -> raise Mismatch

(********** CATEGORY 2 map/fold_left **********)
(* Spring 13, count *)
let count f l =
  let fold_fn acc elem = 
  if (f elem) then acc+1 else acc in 
  List.fold_left fold_fn 0 l

(* Spring 13, stretch. *)
let stretch l =
  let fold_fn acc elem = acc@[elem;elem]in
  List.fold_left fold_fn [] l;;  

(* Winter 13, sum_matrix*)
let sum_matrix m = 
  let fold_fn acc elem = 
    (List.fold_left (+) acc elem) in 
  List.fold_left fold_fn 0 m;;

(********** CATEGORY 3 recursion **********)
(* Fall 13, insertion sort with duplicates *)
let rec insert l i =
  match l with
  | [] -> [i]
  | h::t -> if i<= h then i::l else h::(insert t i)

let insertion_sort = 
 let fold_fn acc elem = 
  insert acc elem in 
 List.fold_left fold_fn []
insertion_sort [1;5;2;5;7;0]








