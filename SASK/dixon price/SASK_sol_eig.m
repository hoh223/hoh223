clear; close all; clc;

% Define range of dimensions
dim_range = 50:50:1000;

% Preallocate arrays for storing computation times
sask_times = zeros(size(dim_range));
grad_desc_times = zeros(size(dim_range));
tols = [1e-3, 1e-6, 1e-9];

% Total number of iterations for the progress bar
total_iterations = length(tols) * length(dim_range);
current_iteration = 0;

% Initialize progress bar
h = waitbar(0, 'Please wait...');

for tol_idx = 1:length(tols)
    tol = tols(tol_idx);
    
    for dim_idx = 1:length(dim_range)
        dim = dim_range(dim_idx);
        
        % Seed for reproducibility
        rand('seed',1); %
        tmp = randperm(dim, floor(dim / 2));
        indx_array = [1 tmp + 1 setdiff(1:dim, tmp) + 1 + dim];
        
        % Initial guess
        y0 = ones(dim, 1);
        r_min = 1e-2;
        odefun = @dedixpr;
        obj = @dixpr;
        
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
        
        [eig_vec, lambda] = eig(K, meas_mat);
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
            
            if norm(odefun(solution)) < tol
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
                
                [eig_vec, lambda] = eig(K, meas_mat);
                lambda = diag(lambda);
                eig_fun = meas_mat * eig_vec;
                coef = eig_fun(indx_array, :) \ xx(indx_array, :);
                t_ind = j;
            end
            decomp_flag = false;
        end
        sask_times(dim_idx) = toc;
        
        % Gradient Descent Method
        tic;
        [x_ref, history] = graddescent(@(x) odefun(x), y0, 1e6, 0.01, tol);
        grad_desc_times(dim_idx) = toc;
        
        % Update progress bar
        current_iteration = current_iteration + 1;
        waitbar(current_iteration / total_iterations, h);
    end
    
    % Plotting results for the current tolerance and basis case
    figure;
    semilogy(dim_range, sask_times, 'b', 'DisplayName', 'SASK');
    hold on;
    semilogy(dim_range, grad_desc_times, 'r', 'DisplayName', 'Gradient Descent');
    xlabel('Dimension');
    ylabel('Time (log scale)');
    title(['Comparison of Computation Time for SASK and Gradient Descent (tol = ', num2str(tol), ')']);
    legend show;
    grid on;
end

% Close the progress bar
close(h);

% Objective and derivative functions
function val = dixpr(xx)
    x1 = xx(1);
    d = length(xx);
    term1 = (x1 - 1)^2;

    % Using vectorized operations for the summation
    xi = xx(2:d);
    xold = xx(1:d-1);
    new_terms = (2 * xi.^2 - xold).^2;
    sum_value = sum((2:d)' .* new_terms);

    val = term1 + sum_value;
end

function grad = dedixpr(xx)
    d = length(xx);
    grad = zeros(size(xx));
    
    % First term derivative
    grad(1) = 2 * (xx(1) - 1);
    
    % Middle terms derivatives
    for ii = 2:d
        if ii > 1
            grad(ii) = grad(ii) + 8 * (ii * xx(ii) .* (2 * xx(ii).^2 - xx(ii - 1)));
        end
        if ii < d
            grad(ii) = grad(ii) - (2 * (ii + 1) * xx(ii + 1) .* (2 * xx(ii + 1).^2 - xx(ii)));
        end
    end
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