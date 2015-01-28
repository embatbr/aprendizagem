mu11 = [60 30];
sigma11 = [9 0; 0 144];
rng default;
r11 = mvnrnd(mu11, sigma11, 100);

mu12 = [52 30];
sigma12 = [9 0; 0 9];
rng default;
r12 = mvnrnd(mu12, sigma12, 100);

mu2 = [45 22];
sigma2 = [100 0; 0 9];
r2 = mvnrnd(mu2, sigma2, 100);


n = 300;

r = [r11; r12; r2];

mu2mle = (1/100)*sum(r2);

mu2tot = repmat(mu2mle, 100, 1);

sigma = diag((1/100)*sum(((r2 - mu2tot)')*(r2 - mu2tot)));

pdf2 = mvnpdf([r11; r12; r2], mu2mle, sigma);

% EM ALGORITMO

r1 = [r11; r12];
[W,M,V,L] = EM_GM(r1, 2, 1, 1000, 0, []);

pdfem11 = mvnpdf([r11; r12; r2], M(:,1)', V(:,:,1));
pdfem12 = mvnpdf([r11; r12; r2], M(:,2)', V(:,:,2));

pdf1em = pdfem11*W(1) + pdfem12*W(2);

pfinal1 = (pdf1em*(2/3))/((pdf1em*(2/3)) + (pdf2*(1/3)));

pfinal2 = (pdf2*(1/3))/((pdf1em*(2/3)) + (pdf2*(1/3)));
