function [Pw1, Pw2, tx_erro] = parzen(r11, r12, r2, teste, treino1, treino2)

r = [r11; r12; r2];
n = 300;
harr = [0.5 0.5];

outpt1 = ones(1,n);
for k = 1:n
    sz = size(treino1(:,(1:2),:));
    outpt1(1,k) = bivar(r(k,:), treino1(:,(1:2),:), harr, sz(1));
end

outpt2 = ones(1,n);
for k = 1:n
    sz = size(treino2(:,(1:2),:));
    outpt2(1,k) = bivar(r(k,:), treino2(:,(1:2),:), harr,sz(1));
end

%Computando P(wi | xk)
Pw1 = ones(1,n);
Pw2 = ones(1,n);
for k = 1:n
    Pw1(1,k) = (outpt1(1,k)*(2/3))/(outpt1(1,k)*(2/3) + outpt2(1,k)*(1/3));
    Pw2(1,k) = (outpt2(1,k)*(1/3))/(outpt1(1,k)*(2/3) + outpt2(1,k)*(1/3));
end

%test = [test1; test2];
%ixtest = [ix11(34:end) ix12(34:end) ix2(34:end)];
%sizeTest = size(ixtest);
%sizeTest = sizeTest(:,2);

sizeTest = size(teste);
sizeTest = sizeTest(1);
decida = ones(1,sizeTest);
% Classificando...
nerro = 0;
for k = 1:sizeTest
    if(Pw1(1,teste(k,3)) == 1)
        if(teste(k,3) > 200)
            nerro = nerro + 1;
        end
        decida(1,k) = 1;
    else
        if(teste(k,3) < 200)
            nerro = nerro + 1;
        end
        decida(1,k) = 0;
    end
end

% Taxa de erro
tx_erro = nerro / sizeTest;

