Certainly! Below is a README file for your simulation code. This README explains how to set up and run the simulation, along with descriptions of key components and configurations.

---

# Language Evolution Simulation in a Dynamic Graph

This repository contains a Python-based simulation that models language evolution in a dynamic graph. Agents (represented as nodes in a graph) evolve languages over time based on their interactions with neighboring agents. The graph structure itself evolves as edges are added or removed between nodes.

## Features

- Simulates language evolution in a dynamic network of agents.
- Supports visualization of the graph and its communities at each simulation step.
- Tracks key metrics such as global fitness, convergence, and the number of groups at convergence.
- Configurable simulation parameters for graph structure, agent behavior, and language properties.

## Requirements

The following Python libraries are required to run the simulation:

- `networkx` - For graph creation and manipulation.
- `matplotlib` - For visualization of the graph.
- `tqdm` - For progress bars.
- `json` - For reading configuration files.
- `argparse` - For command-line argument parsing.

You can install the required libraries using `pip`:

```bash
pip install networkx matplotlib tqdm
```

## File Structure

```
.
├── agent.py            # Contains the Agent class representing agents in the simulation
├── graph.py            # Contains the Graph class representing the graph structure
├── language.py         # Contains the Language class for handling languages of agents
├── main.py       # Main simulation logic (the script you will run)
├── config.json         # Example configuration file (see below)
├── README.md           # This file
```

## Configuration

The simulation relies on a JSON configuration file for specifying parameters. Below is an example of a basic configuration:

```json
{
  "simulation": {
    "edge_add_prob": 0.1,
    "edge_del_prob": 0.05,
    "learning_rate": {
      "upper": 0.1,
      "lower": 0.01
    },
    "language_size": {
      "min": 5,
      "max": 10
    }
  },
  "graph": {
    "community_sizes": [10, 15, 20],
    "p_matrix": [
      [0.8, 0.1, 0.1],
      [0.1, 0.7, 0.2],
      [0.1, 0.2, 0.6]
    ]
  },
  "language": {
    "symbol_set": ["a", "b", "c", "d", "e", "f", "g"]
  }
}
```

### Explanation of Configuration Fields:
- **simulation**:
  - `edge_add_prob`: Probability of adding an edge between any two nodes during each iteration.
  - `edge_del_prob`: Probability of removing an edge between any two nodes during each iteration.
  - `learning_rate`: Upper and lower bounds for the learning rate of agents when updating their language.
  - `language_size`: The minimum and maximum size of the languages that agents can have.
  
- **graph**:
  - `community_sizes`: A list defining the size of each community in the graph. Each community will have its own initial language.
  - `p_matrix`: The probability matrix for community membership, defining the likelihood of connections between different communities.

- **language**:
  - `symbol_set`: A list of symbols that the agents can use in their languages. Each agent's language is a subset of this set.

### Running the Simulation

To run the simulation, you can use the following command:

```bash
python main.py --config config.json --num_steps 100 --num_runs 10
```

### Command-Line Arguments:
- `--config`: Path to the JSON configuration file (e.g., `config.json`).
- `--num_steps`: The number of time steps (iterations) to run in each simulation run.
- `--num_runs`: The number of times to run the simulation (for averaging results).

### Example Output:
The script will output the following statistics after running the specified number of simulations:

```
Percentage of runs with fitness == 1.0: 45.00%
Percentage of runs that converged: 75.00%
Runs with 1 groups at convergence: 5
Runs with 2 groups at convergence: 2
Runs with 3 groups at convergence: 1
Runs with 4+ groups at convergence: 0
```

These results include:
- The percentage of runs where the global fitness reached 1.0.
- The percentage of runs that converged (i.e., when the global fitness did not change between iterations).
- The number of runs where convergence occurred with 1, 2, 3, or 4+ distinct groups.

## Visualizing the Graph

The graph is visualized at each time step if you set `num_runs > 1` in the CLI arguments. This will generate a visualization of the graph, where each node is colored according to the language (group) it belongs to.

The visualization uses the `matplotlib` library, and you will see a color-coded plot representing the community structure of the agents over time.