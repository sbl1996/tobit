#include <torch/script.h>
#include <ATen/ATen.h>

#define PI 3.14159265358979323846

std::tuple<torch::Tensor, torch::Tensor> predict(
        const torch::Tensor& x, const torch::Tensor& P,
        const torch::Tensor& A, const torch::Tensor& Q) {
    auto x1 = torch::matmul(A, x);
    auto P1 = torch::chain_matmul({A, P, A.t()}) + Q;
    return {x1, P1};
}

std::tuple<torch::Tensor, torch::Tensor> update(
        const torch::Tensor& y, const torch::Tensor& x,
        const torch::Tensor& P, const torch::Tensor& H,
        const torch::Tensor& R) {
    auto K = torch::chain_matmul({P, H.t(), torch::inverse(torch::chain_matmul({H, P, H.t()}) + R)});
    auto x1 = x + torch::matmul(K, y - torch::matmul(H ,x));
    auto P1 = P - torch::chain_matmul({K, H, P});
    return {x1, P1};
}


torch::Tensor gaussian_cdf(const torch::Tensor& value) {
    return torch::erf(value / sqrt(2.)).add_(1).mul_(0.5);
}


torch::Tensor gaussian_pdf(const torch::Tensor& value) {
    return (value * value).mul_(-0.5).exp_().div_(sqrt(2 * PI));
}


std::tuple<torch::Tensor, torch::Tensor> tobit_update(
        const torch::Tensor& y, const torch::Tensor& x,
        const torch::Tensor& P, const torch::Tensor& H,
        const torch::Tensor& R, const torch::Tensor& Tl, const torch::Tensor& Tu) {
    auto r = torch::sqrt(torch::diagonal(R)) + 1e-4;
    auto z = torch::matmul(H, x);
    auto zl = (Tl - z).div_(r);
    auto zu = (Tu - z).div_(r);
    auto cpl = gaussian_cdf(zl);
    auto cpu = gaussian_cdf(zu);
    auto ppl = gaussian_pdf(zl);
    auto ppu = gaussian_pdf(zu);
    auto p = cpu - cpl + 1e-4;
    auto Pun = torch::diag(p);
    auto l = (ppu - ppl).div_(p);
    auto Ey = z.addcmul(r, l, -1).mul_(p).addcmul_(cpl, Tl).addcmul_(1 - cpu, Tu);
    auto R_ = torch::matmul(R, torch::diag((zl * ppl).sub_(zu * ppu).sub_(p).add_(1).sub_(l * l)));

    auto R1 = torch::chain_matmul({P, H.t(), Pun});
    auto R2 = torch::chain_matmul({Pun, H, R1}) + R_;
    auto K = torch::matmul(R1, torch::inverse(R2));
    auto x1 = x.addmv_(K, y - Ey);
    auto P1 = P - torch::chain_matmul({K, Pun, H, P});
    return {x1, P1};
}


std::tuple<torch::Tensor, torch::Tensor> filter(
        const torch::Tensor& x0, const torch::Tensor& P0,
        const torch::Tensor& A, const torch::Tensor& H,
        const torch::Tensor& Q, const torch::Tensor& R,
        const torch::Tensor& m) {
    auto n_dim_state = H.size(1);
    auto n_timesteps = m.size(0);
    auto option = x0.options();
    auto state_means = torch::empty({n_timesteps + 1, n_dim_state}, option);
    auto state_covariances = torch::empty({n_timesteps + 1, n_dim_state, n_dim_state}, option);
    state_means[0] = x0;
    state_covariances[0] = P0;
    for (auto t = 0; t < n_timesteps; t++) {
        const auto& [x1, P1] = predict(state_means[t], state_covariances[t], A, Q);
        const auto& [x, P] = update(m[t], x1, P1, H, R);
        state_means[t + 1] = x;
        state_covariances[t + 1] = P;
    }

    return {state_means.narrow(0, 1, n_timesteps), state_covariances.narrow(0, 1, n_timesteps)};
}

std::tuple<torch::Tensor, torch::Tensor> tobit_filter(
        const torch::Tensor& Tl, const torch::Tensor& Tu,
        const torch::Tensor& x0, const torch::Tensor& P0,
        const torch::Tensor& A, const torch::Tensor& H,
        const torch::Tensor& Q, const torch::Tensor& R,
        const torch::Tensor& m) {
    auto n_dim_state = H.size(1);
    auto n_timesteps = m.size(0);
    auto option = x0.options();
    auto state_means = torch::empty({n_timesteps + 1, n_dim_state}, option);
    auto state_covariances = torch::empty({n_timesteps + 1, n_dim_state, n_dim_state}, option);
    state_means[0] = x0;
    state_covariances[0] = P0;
    for (auto t = 0; t < n_timesteps; t++) {
        const auto& [x1, P1] = predict(state_means[t], state_covariances[t], A, Q);
        const auto& [x, P] = tobit_update(m[t], x1, P1, H, R, Tl, Tu);
        state_means[t + 1] = x;
        state_covariances[t + 1] = P;
    }

    return {state_means.narrow(0, 1, n_timesteps), state_covariances.narrow(0, 1, n_timesteps)};
}

static auto registry =
        torch::RegisterOperators("tkf::kalman_filter", &filter)
        .op("tkf::tobit_kalman_filter", &tobit_filter);
