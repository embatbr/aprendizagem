% Validacao Cruzada Estratificada
% Divide em k conjuntos
% k = 10
% Todos os conjuntos tem a mesma cardinalidade
% Cada subconjunto é usado como teste, o restante como treino

[r11, r12, r2] = initialize();

r = [r11; r12; r2];
k = 10;
setsize = 27;
sets = ones(k, setsize, 3);


for i = 1:k
    ix1 = randperm(200);
    ix2 = randperm(100);
    ix2 = ix2 + 200;
    % cj global
    ix = [ix1(1:((2/3)*setsize)) ix2(1:((1/3)*setsize))];
    % subconjunto de uma classe
    ixsubs1 = ix1;
    ixsubs2 = ix2;
    for j = 1:setsize
        sets(i, j, 1) = r(ix(j),1);
        sets(i, j, 2) = r(ix(j),2);
        sets(i, j, 3) = ix(j);
    end
end

% Definicao de testes e treinamento
error_parzen = 0;
error_mle = 0;
errplot_parzen = ones(1,k);
errplot_mle = ones(1,k);
for j = 1:k
    teste = ones(setsize, 3);
    for l = 1:setsize
        teste(l,1) = sets(j, l, 1);
        teste(l,2) = sets(j, l, 2);
        teste(l,3) = sets(j, l, 3);
    end

    % Cj de treinamento para classe 1
    treinoaux = ones(setsize, 3);
    treino1 = 0*ones(setsize, 3);
    for i = 1:k
        if(i ~= j)
            for a = 1:setsize
                if(sets(i, a, 3) < 200)
                    treinoaux(a, 1) = sets(i, a, 1);
                    treinoaux(a, 2) = sets(i, a, 2);
                    treinoaux(a, 3) = sets(i, a, 3);
                else
                    a = a - 1;
                end
            end
            treino1 = union(treinoaux, treino1, 'rows');
        end
    end

    % Cj de treinamento para classe 11
    treinoaux = ones(setsize, 3);
    treino11 = 0*ones(setsize, 3);
    for i = 1:k
        if(i ~= j)
            for a = 1:setsize
                if(sets(i, a, 3) < 100)
                    treinoaux(a, 1) = sets(i, a, 1);
                    treinoaux(a, 2) = sets(i, a, 2);
                    treinoaux(a, 3) = sets(i, a, 3);
                else
                    a = a - 1;
                end
            end
            treino11 = union(treinoaux, treino11, 'rows');
        end
    end

    % Cj de treinamento para classe 12
    treinoaux = ones(setsize, 3);
    treino12 = 0*ones(setsize, 3);
    for i = 1:k
        if(i ~= j)
            for a = 1:setsize
                if((sets(i, a, 3) < 200) && (sets(i, a, 3) > 100))
                    treinoaux(a, 1) = sets(i, a, 1);
                    treinoaux(a, 2) = sets(i, a, 2);
                    treinoaux(a, 3) = sets(i, a, 3);
                else
                    a = a - 1;
                end
            end
            treino12 = union(treinoaux, treino12, 'rows');
        end
    end

    % Cj de treinamento para classe 2:
    treinoaux = ones(setsize, 3);
    treino2 = 0*ones(setsize, 3);
    for i = 1:k
        if(i ~= j)
            for a = 1:setsize
                if(sets(i, a, 3) > 200)
                    treinoaux(a, 1) = sets(i, a, 1);
                    treinoaux(a, 2) = sets(i, a, 2);
                    treinoaux(a, 3) = sets(i, a, 3);
                else
                    a = a - 1;
                end
            end
            treino2 = union(treinoaux, treino2, 'rows');
        end
    end

    treino1 = treino1(2:end,:,:);
    treino2 = treino2(2:end,:,:);

    %Cj de treinamento total:
    treinoaux = ones(setsize, 3);
    treino = 0*ones(setsize, 3);
    for i = 1:k
        if(i ~= j)
            for a = 1:setsize
                treinoaux(a, 1) = sets(i, a, 1);
                treinoaux(a, 2) = sets(i, a, 2);
                treinoaux(a, 3) = sets(i, a, 3);
            end
            treino = union(treinoaux, treino, 'rows');
        end
    end

    treino1 = treino1(2:end,:,:);
    treino2 = treino2(2:end,:,:);
    treino = treino(2:end,:,:);

    % escolha o classificador. Exemplo: Parzen
    [Pw1_parzen, Pw2_parzen, err_parzen] = parzen(r11, r12, r2, teste, treino1, treino2);
    [Pw1_mle, Pw2_mle, err_mle] = maxle(r11, r12, r2, teste, treino1, treino2, treino11, treino12);

    error_parzen = error_parzen + err_parzen;
    errplot(1,j) = err_parzen;

    error_mle = error_mle + err_mle;
    errplot(1,j) = err_mle;
end

%error = error/k;
%plot(err);