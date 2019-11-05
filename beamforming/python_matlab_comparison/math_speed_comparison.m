%% Test the speed of basic complex math operations to compare to python
m = 10000; n=10000;
vals_a = rand(m,n)+1i.*rand(m,n);
vals_b = rand(m,n)+1i.*rand(m,n);
tic;
%for i=1:5
%co = vals_a+vals_a.*vals_b.*exp(vals_a);
co = vals_a*vals_b;
%end
mult_time = toc;

% mathfun = @(x,y) x+x.*y.*exp(x);
% tic;
% coaf = arrayfun(mathfun,vals_a,vals_b);
% mult_time_afun = toc;

