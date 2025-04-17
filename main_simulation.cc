#include "BernoulliBandit.h"
#include "NormalBandit.h"
#include "Experiment.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <omp.h>
#include <vector>
#include <algorithm>

// Simple aligned vector implementation
template<typename T>
class AlignedVector {
private:
    std::vector<T> data;
    static constexpr size_t alignment = 64; // Cache line size

public:
    AlignedVector(size_t n) : data(n, T(0)) {
        // Ensure vector is aligned
        if (reinterpret_cast<uintptr_t>(data.data()) % alignment != 0) {
            throw std::runtime_error("Vector not properly aligned");
        }
    }
    
    T& operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }
    T* begin() { return data.data(); }
    T* end() { return data.data() + data.size(); }
    size_t size() const { return data.size(); }
};

int main() {
    // Simulation parameters
    const int N = 100;              // Number of arms
    const int run_length = 10000;   // Steps per run
    const int n_runs = 10000;       // Number of runs
    const double epsilon = 0.1;     // Exploration rate
    const double learning_rate = 0.1; // Learning rate
    const double Qmax = 0.0;        // Initial Q value

    // Bandit type (0 for Bernoulli, 1 for Normal)
    const int bandit_type = 1;      // Using Normal bandit
    
    // Exploration strategy
    // 0: epsilon-greedy, 1: Boltzmann, 2: UCB, 3: gradient bandit
    const int exploration_strategy = 2; // Using UCB
    
    // Additional parameters for specific strategies
    const double c = 0.1;       // UCB parameter
    const double T = 0.1;       // Temperature for Boltzmann
    const double var = 1.0;     // Variance for Normal bandit

    // Get number of threads
    const int num_threads = omp_get_max_threads();
    
    // Use aligned vectors for better vectorization
    std::vector<AlignedVector<double>> thread_rewards;
    std::vector<AlignedVector<int>> thread_optimal_actions;
    for (int i = 0; i < num_threads; ++i) {
        thread_rewards.emplace_back(run_length);
        thread_optimal_actions.emplace_back(run_length);
    }

    // Open log file for OpenMP results
    std::ofstream logfile("openmp_results.log");
    logfile << "=== OpenMP Multi-armed Bandit Simulation Results ===\n";
    logfile << "Configuration:\n";
    logfile << "Number of arms: " << N << "\n";
    logfile << "Run length: " << run_length << "\n";
    logfile << "Number of runs: " << n_runs << "\n";
    logfile << "Number of threads: " << num_threads << "\n";
    logfile << "Bandit type: " << (bandit_type == 0 ? "Bernoulli" : "Normal") << "\n";
    logfile << "Exploration strategy: ";
    switch(exploration_strategy) {
        case 0: logfile << "epsilon-greedy (epsilon=" << epsilon << ")"; break;
        case 1: logfile << "Boltzmann (T=" << T << ")"; break;
        case 2: logfile << "UCB (c=" << c << ")"; break;
        case 3: logfile << "Gradient bandit"; break;
    }
    logfile << "\n\n";
    
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
    
    // Run simulations in parallel with dynamic scheduling
    const int chunk_size = std::max(1, n_runs / (num_threads * 4)); // Dynamic chunk size
    #pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        
        #pragma omp for schedule(dynamic, chunk_size)
        for (int i = 0; i < n_runs; i++) {
            if (i % 100 == 0) {
                #pragma omp critical
                {
                    std::cout << "Run " << i << " of " << n_runs << " (Thread " << thread_id << ")" << std::endl;
                }
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
            
            // Store results in thread-local storage with vectorized operations
            #pragma omp simd
            for (int j = 0; j < run_length; j++) {
                thread_rewards[thread_id][j] += run_rewards[j];
                thread_optimal_actions[thread_id][j] += run_optimal[j];
            }

            // Print and log values for first and last run
            if (i == 0 || i == n_runs - 1) {
                #pragma omp critical
                {
                    std::cout << "\nTrue arm values for run " << i << ":" << std::endl;
                    bandit->print_true_values();
                    logfile << "\nTrue arm values for run " << i << ":\n";
                    bandit->print_true_values();
                    
                    std::cout << "\nFinal Q values for run " << i << ":" << std::endl;
                    bandit->print_q();
                    logfile << "\nFinal Q values for run " << i << ":\n";
                    bandit->print_q();
                }
            }
            
            delete bandit;
        }
    }
    
    // Combine results from all threads using parallel reduction
    AlignedVector<double> mean_rewards(run_length);
    AlignedVector<double> std_rewards(run_length);
    AlignedVector<double> opt_action_percentage(run_length);
    
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < run_length; j++) {
        double sum_rewards = 0.0;
        double sum_squared_rewards = 0.0;
        double sum_optimal_actions = 0.0;
        
        #pragma omp simd reduction(+:sum_rewards,sum_squared_rewards,sum_optimal_actions)
        for (int t = 0; t < num_threads; t++) {
            sum_rewards += thread_rewards[t][j];
            sum_squared_rewards += thread_rewards[t][j] * thread_rewards[t][j];
            sum_optimal_actions += thread_optimal_actions[t][j];
        }
        
        mean_rewards[j] = sum_rewards / n_runs;
        if (n_runs > 1) {
            double variance = (sum_squared_rewards - (sum_rewards * sum_rewards) / n_runs) / (n_runs - 1);
            std_rewards[j] = std::sqrt(variance);
        } else {
            std_rewards[j] = 0.0;
        }
        opt_action_percentage[j] = (sum_optimal_actions / n_runs) * 100.0;
    }
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Output final statistics
    std::cout << "\n==== Simulation Results ====" << std::endl;
    std::cout << "Total execution time: " << duration.count() / 1000.0 << " seconds" << std::endl;
    
    // Log performance metrics
    logfile << "\n=== Performance Metrics ===\n";
    logfile << "Total execution time: " << duration.count() / 1000.0 << " seconds\n";
    logfile << "\nAverage reward and optimal action selection over time:\n";
    logfile << "Step\tAvg Reward\tStd Dev\t\t% Optimal Actions\n";
    
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
            logfile << j+1 << "\t" << std::fixed << std::setprecision(4) 
                    << mean_rewards[j] << "\t\t" << std_rewards[j] << "\t\t" 
                    << opt_action_percentage[j] << "%\n";
        }
    }
    
    // Write all data to file
    for (int j = 0; j < run_length; j++) {
        outfile << j+1 << "\t" << mean_rewards[j] << "\t" << std_rewards[j] 
                << "\t" << opt_action_percentage[j] << "\n";
    }
    outfile.close();
    logfile.close();
    
    std::cout << "\nFull results saved to " << filename << std::endl;
    std::cout << "OpenMP results logged to openmp_results.log" << std::endl;
    
    return 0;
} 