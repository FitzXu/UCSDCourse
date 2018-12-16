envtype(Bindings,Var,Ty) :- Bindings = [[Var,Ty]|_].
envtype(Bindings,Var,Ty) :- Bindings = [[V1,_]|Tail], not(V1=Var), envtype(Tail,Var,Ty).

int_op(plus).
int_op(minus).
int_op(mul).
int_op(div).

comp_op(eq).
comp_op(neq).
comp_op(lt).
comp_op(leq).

bool_op(and).
bool_op(or).

typeof(_,const(_),X) :- X = int.
typeof(_,boolean(_),X) :- X = bool.
typeof(_,nil,X) :- X = list(_).
typeof(Env,var(X),T) :- envtype(Env,X,T).

typeof(Env,bin(E1,Bop,E2),int) :- int_op(Bop), typeof(Env,E1,int),typeof(Env,E2,int).
typeof(Env,bin(E1,Bop,E2),bool) :- comp_op(Bop), typeof(Env,E1,T),typeof(Env,E2,T).
typeof(Env,bin(E1,Bop,E2),bool) :- bool_op(Bop), typeof(Env,E1,bool),typeof(Env,E2,bool).


typeof(Env,bin(E1,cons,nil),list(T)) :- typeof(Env,E1,T).
typeof(Env,bin(E1,cons,E2),list(T))  :- typeof(Env,E1,T),typeof(Env,E2,list(T)).

typeof(Env, ite(E1,E2,E3),T)  :- typeof(Env,E1,bool), typeof(Env,E2,T), typeof(Env,E3,T).
typeof(Env, let(X,E1,E2), T)  :- typeof(Env,E1,T1),  typeof([[X,T1]|Env],E2,T).
typeof(Env, letrec(X,E1,E2),T) :- typeof([[X,T2]|Env],E1,T2),  typeof([[X,T2]|Env],E2,T).
typeof(Env, fun(X,E), arrow(T1,T2)) :- typeof([[X,T1]|Env], E, T2).
typeof(Env, app(E1,E2),T2) :- typeof(Env,E1,arrow(T1,T2)), typeof(Env,E2,T1).