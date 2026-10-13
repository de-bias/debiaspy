data {
  int<lower=1> N_obs;
  int<lower=1> N_pred;
  int<lower=1> K_true;
  int<lower=1> K_bias;
  matrix[N_obs, K_true] X_true_obs;
  matrix[N_pred, K_true] X_true_pred;
  matrix[N_obs, K_bias] X_bias_obs;
  matrix[N_pred, K_bias] X_bias_pred;
  int<lower=1> L;
  int<lower=1> L_basis;
  matrix[L, L_basis] latent_basis;
  int<lower=1, upper=L> latent_id_obs[N_obs];
  int<lower=1, upper=L> latent_id_pred[N_pred];
  int<lower=0> y[N_obs];
  vector[N_obs] log_q_obs;
  vector[N_pred] log_q_pred;
  int<lower=1> S;
  int<lower=1> S_basis;
  matrix[S, S_basis] source_basis;
  int<lower=1, upper=S> source_id_obs[N_obs];
  int<lower=1, upper=S> source_id_pred[N_pred];
  int<lower=1> T;
  int<lower=1> T_basis;
  matrix[T, T_basis] time_basis;
  int<lower=1, upper=T> time_id_obs[N_obs];
  int<lower=1, upper=T> time_id_pred[N_pred];
  int<lower=0, upper=1> use_time_effect;
  int<lower=0, upper=1> use_negbin;
  int<lower=0, upper=K_true> true_intercept_col;
  real intercept_loc;
  real<lower=0> intercept_scale;
  real<lower=0> prior_coef_scale;
  real<lower=0> prior_bias_scale;
  real<lower=0> prior_latent_state_scale;
  real<lower=0> prior_source_scale;
  real<lower=0> prior_time_scale;
  real<lower=0> phi_prior_rate;
  real<lower=0> max_rng_eta;
}
parameters {
  vector[K_true] beta_true_raw;
  vector[K_bias] beta_bias;
  vector[L_basis] latent_effect_raw;
  vector[S_basis] source_effect_raw;
  vector[T_basis] time_effect_raw;
  real<lower=0> phi;
}
transformed parameters {
  vector[K_true] beta_true = beta_true_raw;
  vector[L] latent_effect = prior_latent_state_scale * (latent_basis * latent_effect_raw);
  vector[S] source_effect = prior_source_scale * (source_basis * source_effect_raw);
  vector[T] time_effect = prior_time_scale * (time_basis * time_effect_raw);

  if (true_intercept_col > 0) {
    beta_true[true_intercept_col] = intercept_loc +
      intercept_scale * beta_true_raw[true_intercept_col];
  }
}
model {
  for (k in 1:K_true) {
    if (k == true_intercept_col) {
      beta_true_raw[k] ~ normal(0, 1);
    } else {
      beta_true_raw[k] ~ normal(0, prior_coef_scale);
    }
  }
  beta_bias ~ normal(0, prior_bias_scale);
  latent_effect_raw ~ normal(0, 1);
  source_effect_raw ~ normal(0, 1);
  time_effect_raw ~ normal(0, 1);
  phi ~ exponential(phi_prior_rate);

  for (n in 1:N_obs) {
    real eta_true = X_true_obs[n] * beta_true +
      latent_effect[latent_id_obs[n]];
    real eta_obs = eta_true +
      log_q_obs[n] +
      X_bias_obs[n] * beta_bias +
      source_effect[source_id_obs[n]] +
      use_time_effect * time_effect[time_id_obs[n]];

    if (use_negbin == 1) {
      y[n] ~ neg_binomial_2_log(eta_obs, phi);
    } else {
      y[n] ~ poisson_log(eta_obs);
    }
  }
}
generated quantities {
  vector[N_pred] flow_true_pred;
  vector[N_pred] mu_mpd_pred;
  vector[N_obs] log_lik;
  int y_rep_obs[N_obs];

  for (n in 1:N_pred) {
    real eta_true = X_true_pred[n] * beta_true +
      latent_effect[latent_id_pred[n]];
    real eta_obs = eta_true +
      log_q_pred[n] +
      X_bias_pred[n] * beta_bias +
      source_effect[source_id_pred[n]] +
      use_time_effect * time_effect[time_id_pred[n]];
    flow_true_pred[n] = exp(fmin(eta_true, max_rng_eta));
    mu_mpd_pred[n] = exp(fmin(eta_obs, max_rng_eta));
  }

  for (n in 1:N_obs) {
    real eta_true = X_true_obs[n] * beta_true +
      latent_effect[latent_id_obs[n]];
    real eta_obs = eta_true +
      log_q_obs[n] +
      X_bias_obs[n] * beta_bias +
      source_effect[source_id_obs[n]] +
      use_time_effect * time_effect[time_id_obs[n]];
    if (use_negbin == 1) {
      log_lik[n] = neg_binomial_2_log_lpmf(y[n] | eta_obs, phi);
      y_rep_obs[n] = neg_binomial_2_log_rng(fmin(eta_obs, max_rng_eta), phi);
    } else {
      log_lik[n] = poisson_log_lpmf(y[n] | eta_obs);
      y_rep_obs[n] = poisson_log_rng(fmin(eta_obs, max_rng_eta));
    }
  }
}
