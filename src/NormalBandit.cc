#include "NormalBandit.h"
#include <random>

NormalBandit::NormalBandit(const int n, const double e, const double l, const double v,
                           const double Qmax): Bandit(n, e, l), var{v} {
    static std::random_device rd;
    static std::mt19937 generator(rd());
    std::normal_distribution<double> distribution(0, var);
    
    for (int i=0; i<N; i++){
        true_values[i] = distribution(generator);
        q[i] = Qmax;
        UCBvalues[i] = 10000;
        nt[i] = 0;
        preferences[i] = 0;
        pii[i] = 0;
    }
    avg_reward = 0;
};

double NormalBandit::sample_return(const int a){
    static std::random_device rd;
    static std::mt19937 generator(rd());
    std::normal_distribution<double> distribution(true_values[a], var);
    return distribution(generator);
};