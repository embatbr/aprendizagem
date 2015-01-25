mu11 = [60 30];
sigma11 = [9 0; 0 144];
rng default;
r11 = mvnrnd(mu11, sigma11, 100);

mu12 = [52 30];
sigma12 = [9 0; 0 9];
rng default;
r12 = mvnrnd(mu12, sigma12, 100);

mu2 = [45 22];a
sigma2 = [100 0; 0 9];
r2 = mvnrnd(mu2, sigma2, 100);


n = 300;

r = [r11; r12; r2];

mu2mle = (1/100)*sum(r2);

mu2tot = repmat(mu2mle, 100, 1);

sigma = diag((1/100)*sum(((r2 - mu2tot)')*(r2 - mu2tot)));


% EM ALGORITMO

r1 = [r11 r12];
EM_GM(r1, 2, 1, 1000, 0, [])
