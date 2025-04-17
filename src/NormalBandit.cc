#include "NormalBandit.h"

// Static random number generator
static std::default_random_engine generator;
static std::normal_distribution<double> distribution(0, 1);

NormalBandit::NormalBandit(const int n, double e, double l, double v, double Qmax): Bandit(n, e, l), var{v} {
    #pragma omp parallel for
    for (int i=0; i<N; i++){
        true_values[i] = distribution(generator) * var;  // Scale by variance
        q[i] = Qmax;
        UCBvalues[i] = 10.0;  // More optimistic initial value
        nt[i] = 0;
        preferences[i] = 0;
        pii[i] = 0;
    }
    avg_reward = 0;
};

double NormalBandit::sample_return(int a){
    // Use smaller noise variance (0.1 * var) for rewards
    return true_values[a] + distribution(generator) * 0.1 * var;
};