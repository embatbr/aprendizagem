function [r11, r12, r2] = initialize()

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