function [Pw1_mle, Pw2_mle, err_mle] = maxle(r11, r12, r2, teste, treino1, treino2, treino11, treino12)

    treino = [treino1; treino2];
    streino = size(treino);
    ix = treino(:,3,:);
    treino = treino(:,sort(ix),:);
    mu2mle = (1/100)*sum((treino2(:,1:2,:)));
    sizet2 = size((treino(:,1:2,:)));
    sizet2 = sizet2(1);
    mu2tot = repmat(mu2mle, sizet2, 1);
    sigma = diag((1/100)*sum((((treino(:,1:2,:)) - mu2tot)')*((treino(:,1:2,:)) - mu2tot)));

    pw2_mle = mvnpdf((treino(:,1:2,:)), mu2tot, sigma);

    [W1,M,V,L] = EM_GM(treino1(:,1:2,:), 2, 1, 1000, 0, []);
    pw11_mle = mvnpdf(treino(:,1:2,:), M(:,1)', V(:,:,1));
    pw12_mle = mvnpdf(treino(:,1:2,:), M(:,2)', V(:,:,2));
    pw1_mle = pw11_mle*(W1(1)) + pw12_mle*(W1(2)); % realizando a mistura atraves dos pesos

    %Pw1_mle = (pw1_mle*(2/3))/((pw1_mle*(2/3)) + (pw2_mle*(1/3)));
    %Pw2_mle = (pw2_mle*(1/3))/((pw1_mle*(2/3)) + (pw2_mle*(1/3)));

    Pw1_mle = pw1_mle/(pw1_mle + pw2_mle);
    Pw2_mle = pw2_mle/(pw1_mle + pw2_mle);

    plot(Pw1_mle)
   % plot(Pw2_mle)

    err_mle = 0.5;