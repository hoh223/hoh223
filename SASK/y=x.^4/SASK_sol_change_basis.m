clear; close all; clc;

% Define fixed dimension
dim = 1000;

% Define range of basis sizes as percentages of the dimension
basis_percentages = 10:10:100; % from 1% to 100%
basis_sizes = floor((basis_percentages / 100) * dim);

% Preallocate arrays for storing computation times
sask_times = zeros(size(basis_sizes));
grad_desc_times = zeros(size(basis_sizes));
tol = [1e-3, 1e-6, 1e-9];

% Total number of iterations for the progress bar
total_iterations = length(tol) * length(basis_sizes);
current_iteration = 0;

% Initialize progress bar
h = waitbar(0, 'Please wait...');

for tol_idx = 1:length(tol)
    tols = tol(tol_idx);
    
    for basis_idx = 1:length(basis_sizes)
        num_basis = basis_sizes(basis_idx);
        basis_label = [num2str(basis_percentages(basis_idx)), '% of dim'];
        
        % Seed for reproducibility
        rand('seed',1); % Note: rng is preferred over rand for seed control
        tmp = randperm(dim, floor(dim / 2));
        indx_array = [1 tmp + 1 setdiff(1:dim, tmp) + 1 + dim];
        
        % Initial guess
        y0 = ones(dim, 1);
        r_min = 1e-2;
        odefun = @dsum4;
        obj = @sum4;
        
        % Neighborhood bounds
        r = r_min;
        low = y0 - r;
        upp = y0 + r;
        
        tic;
        [x_ref, meas_mat] = gen_sparse_lvl1_simple(dim);
        xx = (x_ref + 1) * diag(upp - low) / 2 + low';
        F = -odefun(xx);
        
        K = zeros(2 * dim + 1);
        for k = 1:dim
            K(:, k + 1) = F(:, k) * 2 / (upp(k) - low(k));
            K(k + 1, k + 1 + dim) = -4 * F(k + 1, k) * 2 / (upp(k) - low(k));
            K(k + 1 + dim, k + 1 + dim) = 4 * F(k + 1 + dim, k) * 2 / (upp(k) - low(k));
        end
        
        [eig_vec, lambda] = eigs(K, meas_mat, num_basis);
        lambda = diag(lambda);
        eig_fun = meas_mat * eig_vec;
        coef = eig_fun(indx_array, :) \ xx(indx_array, :);
        
        % Adaptive Spectral Koopman
        n_check = 8;
        sol = y0;
        T = 100000;
        t = linspace(0, T, n_check + 2);
        t_ind = 1;
        decomp_flag = false;
        
        % Preallocate solution array
        solution = zeros(dim, 1);
        
        % Timing for SASK
        for j = 2:n_check + 2
            for k = 1:dim
                solution(k) = real(eig_fun(1, :) * (coef(:, k) .* exp(lambda * (t(j) - t(t_ind)))));
            end
            
            if norm(odefun(solution)) < tols
                break;
            end
            
            if any(solution - y0 > r) || any(solution - y0 < -r)
                decomp_flag = true;
            end
            
            if decomp_flag && (j < n_check + 1)
                y0 = solution;
                low = y0 - r;
                upp = y0 + r;
                xx = (x_ref + 1) * diag(upp - low) / 2 + low';
                F = -odefun(xx);
                
                for k = 1:dim
                    K(:, k + 1) = F(:, k) * 2 / (upp(k) - low(k));
                    K(k + 1, k + 1 + dim) = -4 * F(k + 1, k) * 2 / (upp(k) - low(k));
                    K(k + 1 + dim, k + 1 + dim) = 4 * F(k + 1 + dim, k) * 2 / (upp(k) - low(k));
                end
                
                [eig_vec, lambda] = eigs(K, meas_mat, num_basis);
                lambda = diag(lambda);
                eig_fun = meas_mat * eig_vec;
                coef = eig_fun(indx_array, :) \ xx(indx_array, :);
                t_ind = j;
            end
            decomp_flag = false;
        end
        sask_times(basis_idx) = toc;
        
        % Gradient Descent Method
        tic;
        [x_ref, history] = graddescent(@(x) odefun(x), y0, 1e6, 0.01, tols);
        grad_desc_times(basis_idx) = toc;
        
        % Update progress bar
        current_iteration = current_iteration + 1;
        waitbar(current_iteration / total_iterations, h);
    end
    
    % Plotting results for the current tolerance
    figure;
    semilogy(basis_percentages, sask_times, 'b', 'DisplayName', 'SASK');
    hold on;
    semilogy(basis_percentages, grad_desc_times, 'r', 'DisplayName', 'Gradient Descent');
    xlabel('Basis Size (% of dim)');
    ylabel('Time (log scale)');
    title(['Comparison of Computation Time for SASK and Gradient Descent (tol = ', num2str(tols), ')']);
    legend show;
    grid on;
end

% Close the progress bar
close(h);

% Objective and derivative functions
function val = sum4(y)
    val = sum(y .^ 4);
end

function val = dsum4(y)
    val = 4 * y .^ 3;
end

% Gradient Descent Method
function [x, history] = graddescent(f, x0, max_iter, alpha, tol)
    if nargin < 3, max_iter = 1e6; end
    if nargin < 4, alpha = 0.01; end
    
    x = x0;
    history = zeros(max_iter, 1);
    for k = 1:max_iter
        grad = f(x);
        x = x - alpha * grad;
        history(k) = norm(grad);
        if norm(grad) < tol, break; end
    end
end

% Sparse level 1 points and measurement matrix generation
function [quad_pnt, meas_mat] = gen_sparse_lvl1_simple(dim)
    quad_pnt = zeros(2 * dim + 1, dim);
    quad_pnt(2:dim + 1, :) = -eye(dim);
    quad_pnt(dim + 2:end, :) = eye(dim);
    
    num_quad = 2 * dim + 1;
    meas_mat = zeros(num_quad);
    
    meas_mat(:, 1) = 1;
    meas_mat(2:dim + 1, 2:dim + 1) = -eye(dim);
    meas_mat(dim + 2:end, 2:dim + 1) = eye(dim);
    meas_mat(:, dim + 2:end) = -1;
    meas_mat(2:dim + 1, dim + 2:end) = meas_mat(2:dim + 1, dim + 2:end) + 2 * eye(dim);
    meas_mat(dim + 2:end, dim + 2:end) = meas_mat(dim + 2:end, dim + 2:end) + 2 * eye(dim);
end