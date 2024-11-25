import networkx as nx


class Graph:
    def __init__(self, nodes, community_sizes, p_matrix):
        self.nodes = nodes
        # Create a NetworkX graph object for visualization
        self.graph = nx.stochastic_block_model(community_sizes, p_matrix)

        # Add nodes and edges to the NetworkX graph
        for node in self.nodes:
            self.graph.add_node(node.id)

    def add_edge(self, node1: int, node2: int):
        self.graph.add_edge(node1, node2)  # Also add to NetworkX graph

    def remove_edge(self, node1, node2):
        self.graph.remove_edge(node1, node2)

    def compute_fitness(self, node: int):
        neighbors = list(self.graph.neighbors(node))

        if len(neighbors) == 0:
            return 0  # Handle nodes with no neighbors

        fitness = 0
        current_language = self.nodes[node].language
        for neighbor in neighbors:
            neighbor_language = self.nodes[neighbor].language
            fitness += current_language.similarity(neighbor_language)

        return fitness / len(neighbors)
