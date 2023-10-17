move(1,X,Y,_):-
    write('Move top diskr from'),
    write(X),
    write('to'),
    write(Y),
    nl.a

move(N,X,Y,Z):-
    N>1,
    M is N-1,
    move(M,X,Y,Z),
    move(1,X,Y,_),
    move(M,Z,Y,X).
