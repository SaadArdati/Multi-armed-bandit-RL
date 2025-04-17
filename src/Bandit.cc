#include "Bandit.h"
#include <algorithm>
#include <omp.h>

Bandit::Bandit(const int n, double e, double l) {
    N = n;
    epsilon = e;
    learning_rate = l;
    q = new double[N];
    preferences = new double[N];
    true_values = new double[N];
    nt = new int[N];
    UCBvalues = new double[N];
    q_temperature = new double[N];
    pii = new double[N];
};

int Bandit::take_action(){

    int action = 0;
    double rand_num = ((double) rand() / (RAND_MAX));

    if (rand_num < epsilon){       //random action
        action = rand() % (( N ));
    } else {       //greedy action
        action = std::distance(q, std::max_element(q, q + N));
    }

    if (action == std::distance(true_values, std::max_element(true_values, true_values + N))){
        best_action = 1;
    } else { best_action = 0; }

    nt[action] += 1;

    return action;
};

int Bandit::UCB(int t, double c){
    int action = 0;
    const double t_float = std::max(1.0, static_cast<double>(t));

    #pragma omp parallel for
    for (int j=0; j<N; j++){
        if (nt[j] != 0){
            double alpha = 1.0 / std::sqrt(nt[j]);
            UCBvalues[j] = q[j] + c * std::sqrt(std::log(t_float) / nt[j]);
        }
        else{
            UCBvalues[j] = 10.0 + c * std::sqrt(std::log(t_float));
        }
    }
    action = std::distance(UCBvalues, std::max_element(UCBvalues, UCBvalues + N));

    if (action == std::distance(true_values, std::max_element(true_values, true_values + N))){
        best_action = 1;
    } else { 
        best_action = 0; 
    }

    nt[action] += 1;
    return action;
};

int Bandit::Boltzmann_exploration(double T){
    int action = 0;
    double max_val = 0;
    double denom = 0;

    std::vector<double> weights;

    #pragma omp parallel for reduction(max:max_val)
    for (int i=0; i<N; i++){
        q_temperature[i] = q[i]/T;
        if (q[i] > max_val){
            max_val = q[i];
        }
    }

    #pragma omp parallel for reduction(+:denom)
    for (int i=0; i<N; i++){
        denom += exp(q_temperature[i] - max_val);
    }

    #pragma omp parallel for
    for (int i=0; i<N; i++){
        weights.push_back(exp(q_temperature[i] - max_val)/denom);
    }

    std::random_device rd;
    std::mt19937 generator(rd());

    std::discrete_distribution<int> distribution (weights.begin(), weights.end());
    action = distribution(generator);

    if (action ==  std::distance(true_values, std::max_element(true_values, true_values + N))){
        best_action = 1;
    } else { best_action = 0; }

    return action;
};

int Bandit::gradientBanditAction(){
    double denom = 0;
    int action;

    std::vector<double> pi;

    #pragma omp parallel for reduction(+:denom)
    for (int i=0; i<N; i++){
        denom += exp(preferences[i]);
    }

    #pragma omp parallel for
    for (int i=0; i<N; i++){
        pii[i] = exp(preferences[i])/denom;
        pi.push_back(exp(preferences[i])/denom);
    }

    std::random_device rd;
    std::mt19937 generator(rd());

    std::discrete_distribution<int> distribution (pi.begin(), pi.end());
    action = distribution(generator);

    if (action ==  std::distance(true_values, std::max_element(true_values, true_values + N))){
        best_action = 1;
    } else { best_action = 0; }

    return action;
};

void Bandit::print_q() const noexcept {
    std::cout<<"\n";
    for (int i=0; i<N; i++){
        std::cout<<"i="<<i<<", q["<<i<<"]="<<q[i]<<"\n";
    }
    std::cout<<"\n";
};

void Bandit::print_true_values() const noexcept {
    std::cout<<"\n";
    for (int i=0; i<N; i++){
        std::cout<<"i="<<i<<", tru_value["<<i<<"]="<<true_values[i]<<"\n";
    }
    std::cout<<"\n";
};

void Bandit::print_action_preferences() const noexcept {
    std::cout<<"\n";
    for (int i=0; i<N; i++){
        std::cout<<"i="<<i<<", act_pref["<<i<<"]="<<preferences[i]<<"\n";
    }
    std::cout<<"\n";
};


void Bandit::update_q(double r, int a){
    double alpha = 1.0 / std::sqrt(nt[a]);
    q[a] += alpha * (r - q[a]);
};

void Bandit::update_q_n(double r, int a, int n){
    if (n==0){
        q[a] += (r - q[a]);
    } else {
    q[a] += (1/n)*(r - q[a]);
    }
};


void Bandit::update_avg_reward(int n, double r){
    if (n == 0){
        avg_reward += 1*(r - avg_reward);
    }
    else {
        avg_reward += 1/n*(r - avg_reward);
    }
}

void Bandit::update_action_preferences(double r, int a){
    #pragma omp parallel for
    for (int i=0; i<N; i++){
        if (i==a){
            preferences[i] += learning_rate*(r-avg_reward)*(1-pii[i]);
        } else {
            preferences[i] -= learning_rate*(r-avg_reward)*pii[i];
        }
    }
};


int Bandit::get_best_action(){
    return best_action;
};