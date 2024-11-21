# Language Evolution in Dynamic Graphs Simulation

This project simulates the evolution of languages in a network of agents, where each agent’s language adapts over time based on interactions with neighboring agents. The simulation evolves over several iterations, modifying the agents' languages, network structure (edges), and visualizing how the language distribution changes at each step.

## Requirements


- Python 3.x
- `networkx` library for graph manipulation
- `matplotlib` for visualization

### Install Dependencies
To install the required dependencies, run the following command:

```bash
pip install networkx matplotlib
```

## Usage

To run the simulation, use the following command:

```bash
python simulate_language_evolution.py \
  --num_nodes <number_of_nodes> \
  --num_edges <number_of_edges> \
  --language_size <size_of_language> \
  --num_steps <number_of_steps> \
  --edge_add_prob <probability_of_adding_edges> \
  --edge_del_prob <probability_of_deleting_edges> \
  --learning_rate <learning_rate> \
  --mutation_rate <mutation_rate>
```

### Parameters:
- `--num_nodes`: Number of agents/nodes in the graph (default: 20)
- `--num_edges`: Number of edges in the graph (default: 40)
- `--language_size`: Number of symbols in each agent's language (default: 5)
- `--num_steps`: Number of iterations for the simulation (default: 10)
- `--edge_add_prob`: Probability of adding a new edge between nodes (default: 0.05)
- `--edge_del_prob`: Probability of deleting an existing edge between nodes (default: 0.05)
- `--learning_rate`: Learning rate for language evolution (default: 0.5)
- `--mutation_rate`: Mutation rate for language mutation (default: 0.1)

### Example Commands:
1. **Run with default values:**

   ```bash
   python simulate_language_evolution.py
   ```

2. **Run with 50 nodes, 100 edges, language size of 10, 20 steps, and custom probabilities:**

   ```bash
   python simulate_language_evolution.py --num_nodes 50 --num_edges 100 --language_size 10 --num_steps 20 --edge_add_prob 0.1 --edge_del_prob 0.1 --learning_rate 0.7 --mutation_rate 0.2
   ```

## Output:
The simulation will visualize how the agents’ languages evolve and how the graph structure changes over the course of several iterations. The plots will show at each step how the languages of the agents behave (represented by color maps) and the graph of connections.

## License
This project is licensed under the MIT License.
```

### **How to Run:**
1. Clone or download the script and the `README`.
2. Ensure you have `networkx` and `matplotlib` installed.
3. Use the command line to pass the arguments as needed to run the simulation.