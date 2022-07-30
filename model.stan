functions {
  vector sir(real t, vector y, real p_beta, real p_gamma) {
    real dSdt = - p_beta * y[1] * y[2];
    real dIdt = p_beta * y[1] * y[2] - p_gamma * y[2];
    real dRdt = p_gamma * y[2];
    return to_vector([dSdt, dIdt, dRdt]);
  }
}

data {
  int<lower=1> n;
  int<lower=1> n_obs;
  int<lower=1> n_sample;
  matrix[n, n] distances;
  real t_obs[n, n_obs];
  int infected_obs[n, n_obs];
}

transformed data {
  matrix[n, n] distances_sq = distances .* distances;
  matrix[n, n] jitter = diag_matrix(rep_vector(0.001, n));
  vector[n] zeros = rep_vector(0, n);

  int n_eq = 3;
  real t0 = 0.;
}

parameters {

  vector<lower=0>[n] betas;
  vector<lower=0,upper=1>[n] gammas;
  vector<lower=0,upper=1>[n] initial_infections;

  real<lower=0> eta_sq;
  real<lower=0> rho_sq;

  vector[n] k;
}

transformed parameters {
  vector<lower=0,upper=1>[n] initial_susceptible = 1. - initial_infections;
  matrix[n, n] K;
  vector[n_eq] y_pred[n, n_obs];

  profile("kernel") {
    K = eta_sq * exp(-rho_sq * distances_sq) + jitter;
  }

  profile("ode") {
    for (i in 1:n) {
      y_pred[i] = ode_rk45(sir, [initial_susceptible[i], initial_infections[i], 0.]', t0, t_obs[i], betas[i], gammas[i]);
      print(y_pred[i,,2]);
    }
  }
}

model {

  profile("priors") {

    eta_sq ~ exponential(0.5);
    rho_sq ~ exponential(2);

    betas ~ exponential(0.75);
    gammas ~ beta(1., 20.);

    k ~ multi_normal(zeros, K);

    initial_infections ~ beta(1, 10);
  }

  profile("likelihood") {
    for (i in 1:n) {
      infected_obs[i] ~ binomial(n_sample, to_vector(y_pred[i,,2]));
    }
  }

}
