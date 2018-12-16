
% spring 13, 3) zip
zip([],[],[]).
zip([H1|T1],[H2|T2],R):- zip(T1,T2,RT),R = [[H1,H2]|RT].

% spring 13, 4) quicksort
part([], _, [], []).
part([H1|T1], N, [H1|L1], L2) :- H1=<N, part(T1,N,L1,L2).
part([H1|T1], N, L1, [H1|L2]) :- H1>N, part(T1,N,L1,L2).

% winter 13, 2) list functions
remove_all(_, [], []).
remove_all(K, [H|T], R1) :- K=H, remove_all(K,T,R1).
remove_all(K, [H|T], [H|R1]) :- K\=H, remove_all(K,T,R1).

remove_first(_,[],[]).
remove_first(H, [H|T], T).
remove_first(K, [H|T], [H|R1]) :- K\=H, remove_first(K,T,R1).

prefix( [], _).
prefix( [H|T1],[H|T2]) :- prefix(T1,T2).

segment(A,B) :- prefix(A,B).
segment(A,[_|T]) :- segment(A,T).

% Winter 12, 4) mergesort
sorted([]).
sorted([_]).
sorted([A,B|T]) :- A=<B, sorted(T).


%sort(L1,L2) :- permutation(L1,L2), sorted(L2).

%sort(L1,L2) :- permutation(L1,P), sorted(P),L2 = P,!.

split([],  [],  []).
split([X], [X], []).
split([X|T],[X|R],L) :- split(T,L,R).

merge([],L,L).
merge(L,[],L).
merge([H1|T1],[H2|T2],[H1|R1]) :- H1=<H2, merge(T1,[H2|T2],R1).
merge([H1|T1],[H2|T2],[H2|R2]) :- H1>H2, merge([H1|T1],T2,R2).

merge_sort([],[]).
merge_sort([A],[A]).
merge_sort(L,S) :- split(L,L1,L2),merge_sort(L1,R1),merge_sort(L2,R2),merge(R1,R2,S).

% winter 11, 5) SAT solving
var(0).
var(1).
sat(T):- T = var(1).
sat(T):- not(T = var(0)).

sat(and([H|T])):- sat(H), sat(and(T)).
sat(or([H|T])):-  not(sat(H)),.











