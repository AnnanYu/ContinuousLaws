% S4/S6 continuity experiment under temporal refinement.
%
% This script:
% 1) Generates random SISO S4 and S6 systems that share the same A, B, C, D.
% 2) Builds a smooth random continuous input u(t) on [0,1] from Chebyshev polynomials.
% 3) Computes a high-resolution reference trajectory with RK4 using dt_ref = 2^-14.
% 4) For tau = 2.^(-(10:-1:2)), evaluates ZOH and bilinear discretizations.
% 5) Compares coarse-grid outputs against the reference sampled at the same times.
% 6) Repeats over multiple random systems and saves results to a .mat file.
%
% The S6 continuous-time model used here is the scalar-input version of
% x'(t) = delta(u(t)) * (A x(t) + B * u(t)^2),
% y(t) = u(t) * (C' x(t)) + D u(t),
% with delta(u) = softplus(w_delta * u + b_delta).

rng(37);

%% User settings
cfg.numSystems = 20;
cfg.N = 8; % state dimension
cfg.t0 = 0.0;
cfg.t1 = 1.0;
cfg.dtRef = 2^(-14);
cfg.taus = 2.^(-(10:-1:2));
cfg.numCheb = 20; % number of nonconstant Chebyshev modes
cfg.inputCoeffScale = 0.35;
cfg.inputDCScale = 0.25;
cfg.scale_grid = 2.^(0:0.5:5);
cfg.saveFile = 'discretize_err.mat';

numTaus = numel(cfg.taus);
numSystems = cfg.numSystems;

% Error containers: max abs error and RMSE over sample times.
methods = {'S4_ZOH', 'S4_BIL', 'S6_ZOH', 'S6_BIL'};
for m = 1:numel(methods)
    results.errors.(methods{m}).linf = zeros(numSystems, numTaus, size(cfg.scale_grid,1));
    results.errors.(methods{m}).rmse = zeros(numSystems, numTaus, size(cfg.scale_grid,1));
end

results.cfg = cfg;
results.systems = repmat(struct(), numSystems, 1);

%% Main loop over random systems
for s = 1:numSystems
    fprintf('Running system %d / %d ...\n', s, numSystems);

    % Shared S4/S6 system parameters.
    A = hippo_legS_matrix(cfg.N);
    B = randn(cfg.N, 1) / sqrt(cfg.N);
    C = randn(cfg.N, 1) / sqrt(cfg.N);
    D = 0;

    % S6 selectivity parameters.
    w_delta = randn();
    base_delta = 0.01;
    b_delta = log(exp(base_delta) - 1);

    % Smooth random input via Chebyshev basis on [0,1].
    inputSpec = make_random_chebyshev_input(cfg.numCheb, cfg.inputCoeffScale, cfg.inputDCScale);

    for scale_ind = 1:length(cfg.scale_grid)
        ufun = @(t) 4 * cheb_input_eval(t, inputSpec) * cfg.scale_grid(scale_ind);

        % High-resolution continuous reference via RK4.
        [tRef, yRefS4, yRefS6] = compute_reference_outputs(A, B, C, D, w_delta, b_delta, ufun, cfg);

        % Save system spec.
        results.systems(s).A = A;
        results.systems(s).B = B;
        results.systems(s).C = C;
        results.systems(s).D = D;
        results.systems(s).w_delta = w_delta;
        results.systems(s).b_delta = b_delta;
        results.systems(s).inputSpec = inputSpec;

        % Coarse discretization experiments.
        for j = 1:numTaus
            tau = cfg.taus(j);
            stride = round(tau / cfg.dtRef);
            if abs(stride * cfg.dtRef - tau) > 1e-15
                error('Each tau must be an integer multiple of dtRef.');
            end

            tCoarse = tRef(1:stride:end);
            yRefS4Coarse = yRefS4(1:stride:end);
            yRefS6Coarse = yRefS6(1:stride:end);

            yS4Z = simulate_discrete_s4(A, B, C, D, ufun, tau, cfg.t0, cfg.t1, 'zoh');
            yS4B = simulate_discrete_s4(A, B, C, D, ufun, tau, cfg.t0, cfg.t1, 'bilinear');
            yS6Z = simulate_discrete_s6(A, B, C, D, w_delta, b_delta, ufun, tau, cfg.t0, cfg.t1, 'zoh');
            yS6B = simulate_discrete_s6(A, B, C, D, w_delta, b_delta, ufun, tau, cfg.t0, cfg.t1, 'bilinear');

            % Sanity check lengths.
            assert(numel(yS4Z) == numel(tCoarse), 'Grid mismatch for S4 ZOH');
            assert(numel(yS4B) == numel(tCoarse), 'Grid mismatch for S4 bilinear');
            assert(numel(yS6Z) == numel(tCoarse), 'Grid mismatch for S6 ZOH');
            assert(numel(yS6B) == numel(tCoarse), 'Grid mismatch for S6 bilinear');

            % Errors.
            results.errors.S4_ZOH.linf(s,j,scale_ind) = max(abs(yS4Z - yRefS4Coarse)) / max(abs(yRefS4Coarse));
            results.errors.S4_ZOH.rmse(s,j,scale_ind) = sqrt(mean((yS4Z - yRefS4Coarse).^2)) / sqrt(mean(yRefS4Coarse.^2));

            results.errors.S4_BIL.linf(s,j,scale_ind) = max(abs(yS4B - yRefS4Coarse)) / max(abs(yRefS4Coarse));
            results.errors.S4_BIL.rmse(s,j,scale_ind) = sqrt(mean((yS4B - yRefS4Coarse).^2)) / sqrt(mean(yRefS4Coarse.^2));

            results.errors.S6_ZOH.linf(s,j,scale_ind) = max(abs(yS6Z - yRefS6Coarse)) / max(abs(yRefS6Coarse));
            results.errors.S6_ZOH.rmse(s,j,scale_ind) = sqrt(mean((yS6Z - yRefS6Coarse).^2)) / sqrt(mean(yRefS6Coarse.^2));

            results.errors.S6_BIL.linf(s,j,scale_ind) = max(abs(yS6B - yRefS6Coarse)) / max(abs(yRefS6Coarse));
            results.errors.S6_BIL.rmse(s,j,scale_ind) = sqrt(mean((yS6B - yRefS6Coarse).^2)) / sqrt(mean(yRefS6Coarse.^2));
        end
    end
end

%% Aggregate summaries
results.summary = summarize_results(results.errors, cfg.taus);

save(cfg.saveFile, 'results');
fprintf('Saved results to %s\n', cfg.saveFile);

% Optional quick plot.
make_summary_plot(results);

function A = hippo_legS_matrix(N)
% A simple HiPPO-LegS-style lower-triangular stable matrix.
% This is a common continuous-time HiPPO construction used in S4-style models.
idx = (0:N-1)';
r = sqrt(2*idx + 1);
A = -diag(idx + 1) - tril(r * r.', -1);
end

function spec = make_random_chebyshev_input(K, coeffScale, dcScale)
spec.c0 = dcScale * randn();
spec.coeffs = coeffScale * randn(K,1) ./ (1:K)';
end

function u = cheb_input_eval(t, spec)
% Evaluate a smooth random input on [0,1] using Chebyshev basis on [-1,1].
% T_k(x) = cos(k arccos(x)), x = 2t - 1.
x = 2*t - 1;
x = min(max(x, -1), 1);
theta = acos(x);
u = spec.c0 * ones(size(t));
K = numel(spec.coeffs);
for k = 1:K
    u = u + spec.coeffs(k) * cos(k * theta);
end
u = u;
end

function s = softplus(x)
s = log1p(exp(-abs(x))) + max(x, 0);
end

function [tGrid, yS4, yS6] = compute_reference_outputs(A, B, C, D, w_delta, b_delta, ufun, cfg)
% High-resolution continuous reference via RK4.
dt = cfg.dtRef;
tGrid = cfg.t0:dt:cfg.t1;
numSteps = numel(tGrid);

xS4 = zeros(size(A,1), 1);
xS6 = zeros(size(A,1), 1);
yS4 = zeros(1, numSteps);
yS6 = zeros(1, numSteps);

for i = 1:numSteps
    t = tGrid(i);
    u = ufun(t);
    yS4(i) = C.' * xS4 + D * u;
    yS6(i) = u * (C.' * xS6) + D * u;

    if i == numSteps
        break;
    end

    xS4 = rk4_step(@(tt,xx) s4_rhs(tt,xx,A,B,ufun), t, xS4, dt);
    xS6 = rk4_step(@(tt,xx) s6_rhs(tt,xx,A,B,w_delta,b_delta,ufun), t, xS6, dt);
end
end

function xNext = rk4_step(rhs, t, x, h)
k1 = rhs(t, x);
k2 = rhs(t + 0.5*h, x + 0.5*h*k1);
k3 = rhs(t + 0.5*h, x + 0.5*h*k2);
k4 = rhs(t + h, x + h*k3);
xNext = x + (h/6) * (k1 + 2*k2 + 2*k3 + k4);
end

function dx = s4_rhs(t, x, A, B, ufun)
base_delta = 0.01;
u = ufun(t);
dx = base_delta * (A * x + B * u);
end

function dx = s6_rhs(t, x, A, B, w_delta, b_delta, ufun)
u = ufun(t);
delta = softplus(w_delta * u + b_delta);
dx = delta * (A * x + B * (u^2));
end

function y = simulate_discrete_s4(A, B, C, D, ufun, tau, t0, t1, method)
base_delta = 0.01;
tGrid = t0:tau:t1;
N = size(A,1);
x = zeros(N,1);
y = zeros(1, numel(tGrid));

switch lower(method)
    case 'zoh'
        Abar = expm(tau * base_delta * A);
        Bbar = A \ ((Abar - eye(N)) * B);
        for k = 1:numel(tGrid)
            u = ufun(tGrid(k));
            y(k) = C.' * x + D * u;
            if k < numel(tGrid)
                x = Abar * x + Bbar * u;
            end
        end

    case 'bilinear'
        M1 = eye(N) - 0.5 * base_delta * tau * A;
        M2 = eye(N) + 0.5 * base_delta * tau * A;
        T = M1 \ M2;
        G = M1 \ (tau * base_delta * B);
        for k = 1:numel(tGrid)
            u = ufun(tGrid(k));
            y(k) = C.' * x + D * u;
            if k < numel(tGrid)
                x = T * x + G * u;
            end
        end

    otherwise
        error('Unknown method: %s', method);
end
end

function y = simulate_discrete_s6(A, B, C, D, w_delta, b_delta, ufun, tau, t0, t1, method)
tGrid = t0:tau:t1;
N = size(A,1);
x = zeros(N,1);
y = zeros(1, numel(tGrid));
I = eye(N);

for k = 1:numel(tGrid)
    u = ufun(tGrid(k));
    delta = softplus(w_delta * u + b_delta);
    y(k) = u * (C.' * x) + D * u;

    if k < numel(tGrid)
        switch lower(method)
            case 'zoh'
                Abar = expm((tau * delta) * A);
                Bbar = A \ ((Abar - I) * (B * u));
                x = Abar * x + Bbar * u; % total forcing = (B*u)*u = B*u^2

            case 'bilinear'
                M1 = I - 0.5 * tau * delta * A;
                M2 = I + 0.5 * tau * delta * A;
                T = M1 \ M2;
                G = M1 \ (tau * delta * (B * u));
                x = T * x + G * u; % total forcing = tau*delta*B*u^2

            otherwise
                error('Unknown method: %s', method);
        end
    end
end
end

function summary = summarize_results(errors, taus)
methodNames = fieldnames(errors);
summary.taus = taus;
for i = 1:numel(methodNames)
    m = methodNames{i};
    summary.(m).linf_mean = mean(errors.(m).linf, 1);
    summary.(m).linf_median = median(errors.(m).linf, 1);
    summary.(m).rmse_mean = mean(errors.(m).rmse, 1);
    summary.(m).rmse_median = median(errors.(m).rmse, 1);
end
end

function make_summary_plot(results)
taus = results.cfg.taus;
S = results.summary;

figure('Name', 'S4/S6 temporal refinement errors');
loglog(taus, S.S4_ZOH.linf_mean, '-o', 'DisplayName', 'S4 ZOH'); hold on;
loglog(taus, S.S4_BIL.linf_mean, '-s', 'DisplayName', 'S4 bilinear');
loglog(taus, S.S6_ZOH.linf_mean, '-^', 'DisplayName', 'S6 ZOH');
loglog(taus, S.S6_BIL.linf_mean, '-d', 'DisplayName', 'S6 bilinear');
xlabel('\tau');
ylabel('mean max-abs error');
title('Temporal refinement error vs \tau');
legend('Location', 'best');
grid on;
end