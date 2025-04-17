#include "BernoulliBandit.h"
#include "NormalBandit.h"
#include "Experiment.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <iomanip>

int main() {
    // Simulation parameters
    const int N = 10;               // Number of arms
    const int run_length = 1000;    // Length of each run
    const int n_runs = 1000;        // Number of independent runs
    const double epsilon = 0.1;     // Exploration parameter
    const double learning_rate = 0.1;
    const double Qmax = 0.0;        // Initial Q value

    // Bandit type (0 for Bernoulli, 1 for Normal)
    const int bandit_type = 0;
    
    // Exploration strategy
    // 0: epsilon-greedy, 1: Boltzmann, 2: UCB, 3: gradient bandit
    const int exploration_strategy = 0;
    
    // Additional parameters for specific strategies
    const double c = 2.0;       // UCB parameter
    const double T = 0.5;       // Temperature for Boltzmann
    const double var = 1.0;     // Variance for Normal bandit

    // Arrays to store results
    double *rewards = new double[n_runs * run_length];
    int *optimal_actions = new int[n_runs * run_length];
    double *mean_rewards = new double[run_length]();
    double *std_rewards = new double[run_length]();
    double *opt_action_percentage = new double[run_length]();
    
    std::cout << "Starting simulation with " << n_runs << " runs of " << run_length << " steps each." << std::endl;
    std::cout << "Number of arms: " << N << std::endl;
    std::cout << "Using " << (bandit_type == 0 ? "Bernoulli" : "Normal") << " bandit" << std::endl;
    std::cout << "Exploration strategy: ";
    switch(exploration_strategy) {
        case 0: std::cout << "epsilon-greedy (epsilon=" << epsilon << ")"; break;
        case 1: std::cout << "Boltzmann (T=" << T << ")"; break;
        case 2: std::cout << "UCB (c=" << c << ")"; break;
        case 3: std::cout << "Gradient bandit"; break;
    }
    std::cout << std::endl;
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Run simulations
    for (int i = 0; i < n_runs; i++) {
        if (i % 100 == 0) {
            std::cout << "Run " << i << " of " << n_runs << std::endl;
        }
        
        // Create bandit based on type
        Bandit* bandit;
        if (bandit_type == 0) {
            bandit = new BernoulliBandit(N, epsilon, learning_rate, Qmax);
        } else {
            bandit = new NormalBandit(N, epsilon, learning_rate, var, Qmax);
        }
        
        // Create experiment
        Experiment experiment(epsilon, learning_rate, run_length);
        
        // Run experiment with selected strategy
        switch(exploration_strategy) {
            case 0:
                experiment.single_run(*bandit);
                break;
            case 1:
                experiment.single_run_Boltzmann(*bandit, T);
                break;
            case 2:
                experiment.single_run_UCB(*bandit, c);
                break;
            case 3:
                experiment.single_run_gradient(*bandit);
                break;
        }
        
        // Get results
        double* run_rewards = experiment.get_returns();
        int* run_optimal = experiment.get_opt_actions();
        
        // Print true values for first and last run
        if (i == 0 || i == n_runs - 1) {
            std::cout << "\nTrue arm values for run " << i << ":" << std::endl;
            bandit->print_true_values();
        }
        
        // Store results
        for (int j = 0; j < run_length; j++) {
            rewards[i * run_length + j] = run_rewards[j];
            optimal_actions[i * run_length + j] = run_optimal[j];
        }
        
        // Print final Q values for first and last run
        if (i == 0 || i == n_runs - 1) {
            std::cout << "\nFinal Q values for run " << i << ":" << std::endl;
            bandit->print_q();
        }
        
        delete bandit;
    }
    
    // Calculate statistics
    for (int j = 0; j < run_length; j++) {
        // Calculate mean reward at each time step
        for (int i = 0; i < n_runs; i++) {
            mean_rewards[j] += rewards[i * run_length + j];
            if (optimal_actions[i * run_length + j] == 1) {
                opt_action_percentage[j] += 1.0;
            }
        }
        mean_rewards[j] /= n_runs;
        opt_action_percentage[j] = (opt_action_percentage[j] / n_runs) * 100.0; // Convert to percentage
        
        // Calculate standard deviation
        for (int i = 0; i < n_runs; i++) {
            std_rewards[j] += std::pow(rewards[i * run_length + j] - mean_rewards[j], 2);
        }
        std_rewards[j] = std::sqrt(std_rewards[j] / n_runs);
    }
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Output final statistics
    std::cout << "\n==== Simulation Results ====" << std::endl;
    std::cout << "Total execution time: " << duration.count() / 1000.0 << " seconds" << std::endl;
    
    // Save results to file
    std::string filename = bandit_type == 0 ? "results_bernoulli.txt" : "results_normal.txt";
    std::ofstream outfile(filename);
    outfile << "# Time step, Average reward, Std deviation, % Optimal actions\n";
    
    // Output selected time steps (beginning, middle, end)
    std::cout << "\nAverage reward and optimal action selection over time:" << std::endl;
    std::cout << "Step\tAvg Reward\tStd Dev\t\t% Optimal Actions" << std::endl;
    for (int j : {0, 9, 49, 99, 499, 999}) {
        if (j < run_length) {
            std::cout << j+1 << "\t" << std::fixed << std::setprecision(4) 
                      << mean_rewards[j] << "\t\t" << std_rewards[j] << "\t\t" 
                      << opt_action_percentage[j] << "%" << std::endl;
        }
    }
    
    // Write all data to file
    for (int j = 0; j < run_length; j++) {
        outfile << j+1 << "\t" << mean_rewards[j] << "\t" << std_rewards[j] 
                << "\t" << opt_action_percentage[j] << "\n";
    }
    outfile.close();
    
    std::cout << "\nFull results saved to " << filename << std::endl;
    
    // Clean up
    delete[] rewards;
    delete[] optimal_actions;
    delete[] mean_rewards;
    delete[] std_rewards;
    delete[] opt_action_percentage;
    
    return 0;
} 