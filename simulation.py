import random
import argparse
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict


class Agent:
    def __init__(self, node_id, language):
        self.id = node_id
        self.language = language

    def update_language(self, reference_language, learning_rate=0.5, mutation_rate=0.1):
        new_language = {}
        for key in self.language:
            # Gradual adaptation: blend own and reference language
            if random.random() < learning_rate:
                new_language[key] = reference_language[key]
            else:
                new_language[key] = self.language[key]
            # Mutation: small random changes
            if random.random() < mutation_rate:
                new_language[key] = random.randint(1, 5)  # Assuming values range from 1 to 5
        self.language = new_language


class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.neighbors = defaultdict(list)

        for edge in edges:
            self.neighbors[edge[0]].append(edge[1])
            self.neighbors[edge[1]].append(edge[0])

    def add_edge(self, node1, node2):
        if node2 not in self.neighbors[node1]:
            self.neighbors[node1].append(node2)
            self.neighbors[node2].append(node1)
            self.edges.append([node1, node2])

    def remove_edge(self, node1, node2):
        if node2 in self.neighbors[node1]:
            self.neighbors[node1].remove(node2)
            self.neighbors[node2].remove(node1)
            self.edges.remove([node1, node2])

    def compute_fitness(self, node):
        if len(self.neighbors[node.id]) == 0:
            return 0  # Handle nodes with no neighbors

        fitness = 0
        for neighbor in self.neighbors[node.id]:
            neighbor_node = self.nodes[neighbor]
            for message in neighbor_node.language:
                fitness += 1 if node.language[message] == neighbor_node.language[message] else 0
        return fitness / len(self.neighbors[node.id])


class Simulation:
    def __init__(self, nodes, edges):
        self.graph = Graph(nodes, edges)
        self.global_fitness = 0
        self.compute_global_fitness()

    def compute_global_fitness(self):
        total_fitness = 0
        valid_nodes = 0
        for node in self.graph.nodes:
            fitness = self.graph.compute_fitness(node)
            if fitness > 0:  # Only count nodes with valid neighbors
                total_fitness += fitness
                valid_nodes += 1
        self.global_fitness = total_fitness / valid_nodes if valid_nodes > 0 else 0

    def iterate(self, learning_rate=0.5, mutation_rate=0.1, edge_add_prob=0.05, edge_del_prob=0.05):
        fitness_array = [self.graph.compute_fitness(node) for node in self.graph.nodes]
        reference_nodes = []

        # Determine reference languages from fittest neighbors
        for node in self.graph.nodes:
            reference_node = node
            max_fitness = fitness_array[node.id]
            for neighbor in self.graph.neighbors[node.id]:
                if fitness_array[neighbor] > max_fitness:
                    reference_node = self.graph.nodes[neighbor]
                    max_fitness = fitness_array[neighbor]
            reference_nodes.append(reference_node)

        # Update each node's language with nuance
        for i, node in enumerate(self.graph.nodes):
            node.update_language(reference_nodes[i].language, learning_rate, mutation_rate)

        # Add or remove edges with small probability
        self.modify_edges(edge_add_prob, edge_del_prob)

        self.compute_global_fitness()

    def modify_edges(self, edge_add_prob, edge_del_prob):
        # Add edges with a small probability
        for node1 in range(len(self.graph.nodes)):
            for node2 in range(node1 + 1, len(self.graph.nodes)):  # Ensure no duplicate edges
                if random.random() < edge_add_prob:
                    self.graph.add_edge(node1, node2)

        # Remove edges with a small probability
        edges_to_remove = []
        for edge in self.graph.edges:
            if random.random() < edge_del_prob:
                edges_to_remove.append(edge)

        for edge in edges_to_remove:
            self.graph.remove_edge(edge[0], edge[1])

    def run(self, time_steps, learning_rate=0.5, mutation_rate=0.1, edge_add_prob=0.05, edge_del_prob=0.05):
        for step in range(time_steps):
            self.iterate(learning_rate, mutation_rate, edge_add_prob, edge_del_prob)
            self.visualize(step)

    def visualize(self, step):
        """
        Visualizes the network and the agents' languages as a color map at each iteration.
        The color is determined by the sum of the values in the agent's language.
        """
        # Assign each node a color based on the sum of their language values
        node_colors = []
        for node in self.graph.nodes:
            language_sum = sum(node.language.values())
            node_colors.append(language_sum)

        # Plot the graph
        G_nx = nx.Graph()
        G_nx.add_edges_from(self.graph.edges)
        pos = nx.spring_layout(G_nx)  # Using spring layout for better visualization

        # Draw the graph
        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(G_nx, pos, node_size=500, cmap=plt.cm.viridis, node_color=node_colors)
        nx.draw_networkx_edges(G_nx, pos, width=1.0, alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(G_nx, pos, font_size=12, font_color='black')

        # Title showing the iteration step
        plt.title(f"Language Evolution at Iteration {step + 1}")
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label="Language Sum")
        plt.axis('off')
        plt.show()


# Language Generation Logic
def generate_language(symbols, value_range=(1, 5)):
    """
    Generate a language represented by a dictionary, where:
    - Symbols are the keys.
    - Random integers (within value_range) are the values.
    """
    return {symbol: random.randint(value_range[0], value_range[1]) for symbol in symbols}


# Graph Generation Logic
def generate_graph(num_nodes, num_edges):
    """
    Generates a graph with a specific number of edges.
    """
    G = nx.gnm_random_graph(num_nodes, num_edges)
    return G


# Main Simulation Logic
def run_simulation(num_nodes, num_edges, language_size, num_steps, edge_add_prob, edge_del_prob, learning_rate,
                   mutation_rate):
    # Generate graph using NetworkX
    G = generate_graph(num_nodes, num_edges)
    edges = list(G.edges())

    # Generate random languages for each node
    symbols = [chr(97 + i) for i in range(language_size)]  # Generating symbols like 'a', 'b', 'c', ...
    nodes = [Agent(i, generate_language(symbols)) for i in range(num_nodes)]

    # Run the simulation
    simulation = Simulation(nodes, edges)
    simulation.run(num_steps, learning_rate=learning_rate, mutation_rate=mutation_rate, edge_add_prob=edge_add_prob,
                   edge_del_prob=edge_del_prob)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Simulate language evolution in a dynamic graph.")

    parser.add_argument('--num_nodes', type=int, default=20, help="Number of nodes in the graph.")
    parser.add_argument('--num_edges', type=int, default=40, help="Number of edges in the graph.")
    parser.add_argument('--language_size', type=int, default=5, help="Number of symbols in each agent's language.")
    parser.add_argument('--num_steps', type=int, default=10, help="Number of iterations for the simulation.")
    parser.add_argument('--edge_add_prob', type=float, default=0.05, help="Probability of adding a new edge.")
    parser.add_argument('--edge_del_prob', type=float, default=0.05, help="Probability of deleting an edge.")
    parser.add_argument('--learning_rate', type=float, default=0.5, help="Learning rate for language update.")
    parser.add_argument('--mutation_rate', type=float, default=0.1, help="Mutation rate for language variation.")

    args = parser.parse_args()

    # Run the simulation with the parsed arguments
    run_simulation(
        num_nodes=args.num_nodes,
        num_edges=args.num_edges,
        language_size=args.language_size,
        num_steps=args.num_steps,
        edge_add_prob=args.edge_add_prob,
        edge_del_prob=args.edge_del_prob,
        learning_rate=args.learning_rate,
        mutation_rate=args.mutation_rate
    )
