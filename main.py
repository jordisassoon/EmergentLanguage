import argparse
import json
from collections import defaultdict
import random

import networkx as nx
import tqdm
from matplotlib import pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities

from agent import Agent
from graph import Graph
from language import Language


class Simulation:
    """
    Simulation class for modeling language evolution in a dynamic graph.
    It simulates a process where agents (nodes) in a graph evolve languages over time
    based on interactions with their neighbors, with community structure and
    language evolution dynamics.
    """

    def __init__(self, config, visualize_and_print=False):
        """
        Initializes the simulation with configuration parameters.
        """
        # Load simulation parameters from the config
        self.visualize_and_print = visualize_and_print
        sim_config = config["simulation"]
        self.edge_add_prob = sim_config["edge_add_prob"]
        self.edge_del_prob = sim_config["edge_del_prob"]
        self.lr_upper = sim_config["learning_rate"]["upper"]
        self.lr_lower = sim_config["learning_rate"]["lower"]
        self.min_language_size = sim_config["language_size"]["min"]
        self.max_language_size = sim_config["language_size"]["max"]

        # Load graph parameters
        graph_config = config["graph"]
        community_sizes = graph_config["community_sizes"]
        p_matrix = graph_config["p_matrix"]

        # Load language parameters
        language_config = config["language"]
        symbol_set = language_config["symbol_set"]

        # Initialize languages for each community
        self.languages = [
            Language(symbol_set, self.min_language_size, self.max_language_size) for _ in community_sizes
        ]

        # Initialize agents (nodes) based on community sizes
        self.nodes = [Agent(i, None, self.lr_upper, self.lr_lower) for i in range(sum(community_sizes))]

        # Create the graph with nodes and community structure
        self.graph = Graph(self.nodes, community_sizes, p_matrix)

        # Assign languages to nodes based on community membership
        for node_idx, community in enumerate(self.get_node_communities()):
            self.nodes[node_idx].set_language(self.languages[community])

        # Store the nodes in the graph object
        self.graph.nodes = self.nodes

        # Compute initial global fitness
        self.global_fitness = 0
        self.compute_global_fitness()

    def get_node_communities(self):
        """
        Detect communities in the graph using greedy modularity method.
        Returns the community assignments for each node.
        """
        communities = list(greedy_modularity_communities(self.graph.graph))

        # Map each node to its community index
        node_to_community = {}
        for community_index, community in enumerate(communities):
            for node in community:
                node_to_community[node] = community_index

        # Return community assignments for all nodes
        return [node_to_community[node.id] for node in self.nodes]

    def visualize(self, step):
        """
        Visualizes the graph at a given step with nodes colored based on their group membership.
        """
        # Create a list of node dictionaries (language concept maps)
        node_dicts = [node.language.concept_map for node in self.nodes]

        # Group the nodes by their equivalent language dictionaries
        grouped_dicts = self.group_equivalent_dicts(node_dicts)

        # Map each node to a color based on the group it belongs to
        node_colors = {}
        for group_idx, group in grouped_dicts.items():
            color = group_idx  # Group number serves as a color ID
            for node_dict in group:
                # Find the node corresponding to this dictionary
                for node in self.nodes:
                    if node.language.concept_map == node_dict:
                        node_colors[node.id] = color

        # List of colors corresponding to each node
        color_list = [node_colors[node.id] for node in self.nodes]

        # Layout for graph visualization
        pos = nx.spring_layout(self.graph.graph)  # Graph layout (fixed position for nodes)

        # Plot the graph
        plt.figure(figsize=(10, 8))
        nodes = nx.draw_networkx_nodes(
            self.graph.graph,
            pos,
            node_color=color_list,
            cmap=plt.cm.tab20,  # Colormap for distinct colors
            node_size=250
        )
        nx.draw_networkx_edges(self.graph.graph, pos)  # Draw the edges
        nx.draw_networkx_labels(self.graph.graph, pos, font_color="white")  # Draw node labels

        # Add a colorbar to indicate group numbers
        sm = plt.cm.ScalarMappable(cmap=plt.cm.tab20, norm=plt.Normalize(vmin=0, vmax=max(node_colors.values())))
        sm.set_array([])
        plt.colorbar(sm, label="Group Number")

        # Add title and remove axes
        plt.title(f"Graph Visualization at Step {step}")
        plt.axis("off")  # Hide the axes
        plt.show()

    def run(self, time_steps):
        """
        Runs the simulation for a given number of time steps, printing global fitness
        and visualizing the graph at each step.
        """
        fitness_tracker = self.global_fitness
        converged = False
        groups_at_convergence = None
        for i in range(time_steps):
            if self.visualize_and_print:
                print(f'Global Fitness at iteration {i}: {self.global_fitness}')

            if self.visualize_and_print:
                # Visualize the graph at this step
                self.visualize(i)

            # Group the nodes by equivalent language concept maps
            result = self.group_equivalent_dicts([node.language.concept_map for node in self.nodes])

            if self.visualize_and_print:
                # Output the groupings
                for group_num, group in result.items():
                    print(f"Group {group_num} of size {len(group)}: {group[0]}")

            # Check number of groups at convergence
            if i == time_steps - 1:
                groups_at_convergence = len(result)

            # Perform the evolution of the graph (language changes)
            self.iterate()

            converged = fitness_tracker == self.global_fitness
            fitness_tracker = self.global_fitness

        return fitness_tracker, converged, groups_at_convergence

    @staticmethod
    def group_equivalent_dicts(dicts):
        """
        Groups equivalent dictionaries by their normalized representation.
        Returns a dictionary mapping group numbers to lists of equivalent dictionaries.
        """

        def normalize(dictionary):
            return tuple(sorted(dictionary.items()))  # Normalize dictionary to sorted tuple (key, value)

        # Group the dictionaries by their normalized representation
        groups = defaultdict(list)
        for d in dicts:
            normalized = normalize(d)
            groups[normalized].append(d)

        # Assign unique group numbers
        grouped_dict = {i: group for i, group in enumerate(groups.values())}

        return grouped_dict

    def compute_global_fitness(self):
        """
        Computes the global fitness of the entire population based on node fitness.
        """
        self.global_fitness = sum(self.graph.compute_fitness(node.id) for node in self.nodes) / len(self.nodes)

    def iterate(self):
        """
        Executes one iteration of the simulation, updating the nodes' languages based on
        the fittest neighboring node and modifying edges (with probability).
        """
        fitness_array = [self.graph.compute_fitness(node.id) for node in self.nodes]
        reference_nodes = []

        # Determine reference languages from the fittest neighbors
        for node in self.graph.nodes:
            reference_node = node
            max_fitness = fitness_array[node.id]
            for neighbor in list(self.graph.graph.neighbors(node.id)):
                if fitness_array[neighbor] > max_fitness:
                    reference_node = self.graph.nodes[neighbor]
                    max_fitness = fitness_array[neighbor]
            reference_nodes.append(reference_node)

        # Update each node's language with the language of the reference node
        for i, node in enumerate(self.graph.nodes):
            node.update_language(reference_nodes[i].language.concept_map)

        self.modify_edges(self.edge_add_prob, self.edge_del_prob)

        # Recompute global fitness after the update
        self.compute_global_fitness()

    def modify_edges(self, edge_add_prob, edge_del_prob):
        """
        Modifies the graph's edges by adding or removing edges with small probability.
        """
        # Add edges with a small probability
        for node1 in range(len(self.graph.nodes)):
            for node2 in range(node1 + 1, len(self.graph.nodes)):  # Ensure no duplicate edges
                if random.random() < edge_add_prob:
                    self.graph.add_edge(node1, node2)

        # Remove edges with a small probability
        edges_to_remove = []
        for edge in self.graph.graph.edges:
            if random.random() < edge_del_prob:
                edges_to_remove.append(edge)

        # Remove the selected edges from the graph
        for edge in edges_to_remove:
            self.graph.remove_edge(edge[0], edge[1])


def run_simulation(config_path, num_steps, visualize_and_print):
    """
    Loads the configuration from a file, initializes the simulation, and runs it.
    """
    # Load configuration from JSON file
    with open(config_path, "r") as file:
        config = json.load(file)

    # Initialize and run the simulation
    simulation = Simulation(config, visualize_and_print)
    return simulation.run(num_steps)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Simulate language evolution in a dynamic graph.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    parser.add_argument('--num_steps', type=int, default=10, help="Number of iterations for the simulation.")
    parser.add_argument('--num_runs', type=int, default=1, help="Number of times to run this experiment.")

    args = parser.parse_args()

    global_fitnesses = []
    conversions = []
    group_counts = {1: 0, 2: 0, 3: 0, '4+': 0}

    for _ in tqdm.tqdm(range(args.num_runs)):
        # Run the simulation with the provided config and steps
        fitness_tracker, converged, groups_at_convergence = run_simulation(args.config, args.num_steps, visualize_and_print=args.num_runs == 1)
        global_fitnesses.append(fitness_tracker)
        conversions.append(converged)

        if converged:
            if groups_at_convergence >= 4:
                group_counts['4+'] += 1
            else:
                group_counts[groups_at_convergence] += 1

    # Calculate the percentage of runs with fitness == 1.0
    fitness_1_percentage = sum(1 for fitness in global_fitnesses if fitness == 1.0) / args.num_runs * 100

    # Calculate the percentage of runs that converged
    converged_percentage = sum(1 for converged in conversions if converged) / args.num_runs * 100

    # Print out the results
    print(f"Percentage of runs with fitness == 1.0: {fitness_1_percentage:.2f}%")
    print(f"Percentage of runs that converged: {converged_percentage:.2f}%")
    for num_groups, count in group_counts.items():
        print(f"Runs with {num_groups} groups at convergence: {count}")
