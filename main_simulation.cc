#include "BernoulliBandit.h"
#include "NormalBandit.h"
#include "Experiment.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <mpi.h>

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

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

    // Calculate local number of runs for each process
    int local_n_runs = n_runs / world_size;
    if (world_rank < n_runs % world_size) {
        local_n_runs++;
    }

    // Arrays to store local results
    double *local_rewards = new double[local_n_runs * run_length]();
    int *local_optimal_actions = new int[local_n_runs * run_length]();
    double *local_mean_rewards = new double[run_length]();
    double *local_std_rewards = new double[run_length]();
    double *local_opt_action_percentage = new double[run_length]();
    
    // Initialize arrays
    for (int i = 0; i < local_n_runs * run_length; i++) {
        local_rewards[i] = 0.0;
        local_optimal_actions[i] = 0;
    }

    // Open log file for MPI results (only root process)
    std::ofstream logfile;
    if (world_rank == 0) {
        logfile.open("mpi_results.log");
        logfile << "=== Parallel Multi-armed Bandit Simulation Results ===\n";
        logfile << "Number of MPI processes: " << world_size << "\n";
        logfile << "Configuration:\n";
        logfile << "Number of arms: " << N << "\n";
        logfile << "Run length: " << run_length << "\n";
        logfile << "Total number of runs: " << n_runs << "\n";
        logfile << "Local runs per process: " << local_n_runs << "\n";
        logfile << "Bandit type: " << (bandit_type == 0 ? "Bernoulli" : "Normal") << "\n";
        logfile << "Exploration strategy: ";
        switch(exploration_strategy) {
            case 0: logfile << "epsilon-greedy (epsilon=" << epsilon << ")"; break;
            case 1: logfile << "Boltzmann (T=" << T << ")"; break;
            case 2: logfile << "UCB (c=" << c << ")"; break;
            case 3: logfile << "Gradient bandit"; break;
        }
        logfile << "\n\n";
    }
    
    if (world_rank == 0) {
        std::cout << "Starting parallel simulation with " << n_runs << " total runs of " << run_length << " steps each." << std::endl;
        std::cout << "Number of MPI processes: " << world_size << std::endl;
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
    }
    
    // Start timing
    double start_time = MPI_Wtime();
    
    // Run local simulations
    for (int i = 0; i < local_n_runs; i++) {
        if (i % 100 == 0) {
            std::cout << "Process " << world_rank << ": Run " << i << " of " << local_n_runs << std::endl;
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
        
        // Store results
        for (int j = 0; j < run_length; j++) {
            local_rewards[i * run_length + j] = run_rewards[j];
            local_optimal_actions[i * run_length + j] = run_optimal[j];
        }
        
        delete bandit;
    }
    
    // Calculate local statistics
    for (int j = 0; j < run_length; j++) {
        double sum_rewards = 0.0;
        double sum_squared_rewards = 0.0;
        double sum_optimal_actions = 0.0;
        int num_samples = 0;

        for (int i = 0; i < local_n_runs; i++) {
            double reward = local_rewards[i * run_length + j];
            sum_rewards += reward;
            sum_squared_rewards += reward * reward;
            if (local_optimal_actions[i * run_length + j] == 1) {
                sum_optimal_actions += 1.0;
            }
            num_samples++;
        }

        local_mean_rewards[j] = sum_rewards / num_samples;
        if (num_samples > 1) {
            double variance = (sum_squared_rewards - (sum_rewards * sum_rewards) / num_samples) / (num_samples - 1);
            local_std_rewards[j] = std::sqrt(variance);
        } else {
            local_std_rewards[j] = 0.0;
        }
        local_opt_action_percentage[j] = (sum_optimal_actions / num_samples) * 100.0;
    }
    
    // Allocate arrays for global results (only on root process)
    double *global_mean_rewards = nullptr;
    double *global_std_rewards = nullptr;
    double *global_opt_action_percentage = nullptr;
    
    if (world_rank == 0) {
        global_mean_rewards = new double[run_length]();
        global_std_rewards = new double[run_length]();
        global_opt_action_percentage = new double[run_length]();
    }
    
    // Gather results to root process
    MPI_Reduce(local_mean_rewards, global_mean_rewards, run_length, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_std_rewards, global_std_rewards, run_length, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_opt_action_percentage, global_opt_action_percentage, run_length, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // End timing
    double end_time = MPI_Wtime();
    double total_time = end_time - start_time;
    
    // Output final statistics (only root process)
    if (world_rank == 0) {
        // Average the results
        for (int j = 0; j < run_length; j++) {
            global_mean_rewards[j] /= world_size;
            global_std_rewards[j] /= world_size;
            global_opt_action_percentage[j] /= world_size;
        }
        
        std::cout << "\n==== Simulation Results ====" << std::endl;
        std::cout << "Total execution time: " << total_time << " seconds" << std::endl;
        
        // Log performance metrics
        logfile << "\n=== Performance Metrics ===\n";
        logfile << "Total execution time: " << total_time << " seconds\n";
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
                          << global_mean_rewards[j] << "\t\t" << global_std_rewards[j] << "\t\t" 
                          << global_opt_action_percentage[j] << "%" << std::endl;
                logfile << j+1 << "\t" << std::fixed << std::setprecision(4) 
                        << global_mean_rewards[j] << "\t\t" << global_std_rewards[j] << "\t\t" 
                        << global_opt_action_percentage[j] << "%\n";
            }
        }
        
        // Write all data to file
        for (int j = 0; j < run_length; j++) {
            outfile << j+1 << "\t" << global_mean_rewards[j] << "\t" << global_std_rewards[j] 
                    << "\t" << global_opt_action_percentage[j] << "\n";
        }
        outfile.close();
        logfile.close();
        
        std::cout << "\nFull results saved to " << filename << std::endl;
        std::cout << "MPI results logged to mpi_results.log" << std::endl;
        
        // Clean up global arrays
        delete[] global_mean_rewards;
        delete[] global_std_rewards;
        delete[] global_opt_action_percentage;
    }
    
    // Clean up local arrays
    delete[] local_rewards;
    delete[] local_optimal_actions;
    delete[] local_mean_rewards;
    delete[] local_std_rewards;
    delete[] local_opt_action_percentage;
    
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
} 