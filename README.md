# Multi-armed Bandit Reinforcement Learning

This project implements various multi-armed bandit algorithms and explores their performance through sequential and parallel implementations.

## Baseline Performance (Sequential Implementation)

The sequential implementation was tested with the following configuration:
- Number of arms: 10
- Run length: 1000 steps
- Number of independent runs: 1000
- Bandit type: Bernoulli
- Exploration strategy: epsilon-greedy (epsilon=0.1)
- Learning rate: 0.1

### Performance Metrics

1. **Execution Time**: 0.045 seconds
2. **Average Reward Progression**:
   - Step 1: 0.8493
   - Step 10: 0.8493
   - Step 50: 0.8493
   - Step 100: 0.8493
   - Step 500: 0.8493
   - Step 1000: 0.9260

3. **Optimal Action Selection**:
   - Step 1: 9.7%
   - Step 10: 9.7%
   - Step 50: 9.7%
   - Step 100: 9.7%
   - Step 500: 9.7%
   - Step 1000: 18.0%

## Parallelization Plans

### OpenMP Implementation
- Plan to parallelize the main simulation loop using OpenMP
- Expected speedup: Linear with number of cores
- Key metrics to track:
  - Speedup ratio
  - Memory usage
  - Load balancing

### MPI Implementation
- Plan to distribute runs across multiple nodes
- Expected speedup: Linear with number of nodes
- Key metrics to track:
  - Communication overhead
  - Load balancing
  - Scalability

## Building and Running

### Sequential Version
```bash
make -f Makefile.simulation
make -f Makefile.simulation run
```

### OpenMP Version (Coming Soon)
```bash
make -f Makefile.openmp
make -f Makefile.openmp run
```

### MPI Version (Coming Soon)
```bash
make -f Makefile.mpi
mpirun -np <num_processes> ./bandit_simulation_mpi
```

## Results Storage

Results are stored in the following files:
- `sequential_results.log`: Complete sequential run results
- `results_bernoulli.txt` or `results_normal.txt`: Detailed time-step data
- `openmp_results.log`: OpenMP implementation results (coming soon)
- `mpi_results.log`: MPI implementation results (coming soon)

## Dependencies
- C++14 or later
- OpenMP (for parallel version)
- MPI (for distributed version)

# Multi-Armed Bandit with Reinforcement Learning

This repository contains a C++ implementation of the Multi-Armed Bandit described in "Reinforcement Learning - An introduction" by Sutton and Barto.

---

The following exploration policies have been implemented:
* Epsilon-greedy
* Boltzmann (softmax)
* UCB
* Gradient Bandit Algorithm with softmax action preferences

---

## Quick start

To compile Gaussian Bandit code use:
```
make -f Makefile.gaus
```

run the executable with `./a.out` and insert the requested parameters. 


In the same way, to compile Bernoulli Bandit code use:
```
make -f Makefile.bern
```
