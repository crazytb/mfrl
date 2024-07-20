import numpy as np
import csv

class Topology:
    def __init__(self, n, model="random", density=1):
        self.n = n
        self.model = model
        self.density = density
        self.adjacency_matrix = self.make_adjacency_matrix()
        
    def make_adjacency_matrix(self) -> np.ndarray:
        if self.density < 0 or self.density > 1:
            raise ValueError("Density must be between 0 and 1.")

        n_edges = int(self.n * (self.n - 1) / 2 * self.density)
        adjacency_matrix = np.zeros((self.n, self.n))

        if self.model == "dumbbell":
            adjacency_matrix[0, self.n-1] = 1
            adjacency_matrix[self.n-1, 0] = 1
            for i in range(1, self.n//2):
                adjacency_matrix[0, i] = 1
                adjacency_matrix[i, 0] = 1
            for i in range(self.n//2+1, self.n):
                adjacency_matrix[i-1, self.n-1] = 1
                adjacency_matrix[self.n-1, i-1] = 1
        elif self.model == "linear":
            for i in range(1, self.n):
                adjacency_matrix[i-1, i] = 1
                adjacency_matrix[i, i-1] = 1
        elif self.model == "random":
            for i in range(1, self.n):
                adjacency_matrix[i-1, i] = 1
                adjacency_matrix[i, i-1] = 1
                n_edges -= 1
            if n_edges <= 0:
                return adjacency_matrix
            else:
                arr = [1]*n_edges + [0]*((self.n-1)*(self.n-2)//2 - n_edges)
                np.random.shuffle(arr)
                for i in range(0, self.n):
                    for j in range(i+2, self.n):
                        adjacency_matrix[i, j] = arr.pop()
                        adjacency_matrix[j, i] = adjacency_matrix[i, j]
        else:
            raise ValueError("Model must be dumbbell, linear, or random.")
        return adjacency_matrix

    def show_adjacency_matrix(self):
        print(self.adjacency_matrix)
        
    def get_density(self):
        return np.sum(self.adjacency_matrix) / (self.n * (self.n - 1))
    
    # def save_graph_with_labels(self, path):
    #     rows, cols = np.where(self.adjacency_matrix == 1)
    #     edges = zip(rows.tolist(), cols.tolist())
    #     G = nx.Graph()
    #     G.add_edges_from(edges)
    #     pos = nx.kamada_kawai_layout(G)
    #     nx.draw_networkx(G, pos=pos, with_labels=True)
    #     plt.savefig(path + '/adj_graph.png')

def get_adjacent_nodes(node, topology):
    return np.where(topology.adjacency_matrix[node] == 1)[0]


def run_simulation(topology, 
                   n_steps=300, 
                   n_episodes=100, 
                   transmission_probs=0.0,
                   log_file="simulation_logs.csv"):
    node_n = topology.n
    episode_length = n_steps
    total_utility = 0
    log_data = []
    
    for episode in range(n_episodes):
        episode_utility = 0
        aoi = np.zeros(node_n)
        
        for step in range(episode_length):
            actions = np.random.rand(node_n)
            # transmission_probs = np.random.rand(node_n)
            tx_trials = (transmission_probs > actions).astype(int)
            utility = np.array([0.0])
            aoi += 1 / episode_length

            # 전송을 시도하는 노드들에 대해,
            for i in np.where(tx_trials == 1)[0]:
                # j: i와 인접한 노드
                for j in get_adjacent_nodes(i, topology):
                    # j와 인접한 노드들 중 i가 아닌 노드들
                    js_adjacent_nodes = get_adjacent_nodes(j, topology)
                    js_adjacent_nodes_except_ind = js_adjacent_nodes[js_adjacent_nodes!=i]
                    if ((np.all(tx_trials[js_adjacent_nodes_except_ind]==0)) 
                        and (tx_trials[j]==0)):
                        # utility += np.log2(1 + aoi[i])
                        utility += np.tanh(5*aoi[i])
                        aoi[i] = 0
                        break

            episode_utility += utility
            log_entry = [episode, step] + aoi.tolist() + utility.tolist() + np.where(tx_trials==1)[0].tolist()
            log_data.append(log_entry)
                
        total_utility += episode_utility
    
    # Save logs to CSV
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["Episode"] + ["Step"] + [f"Node_{i}" for i in range(node_n)] + ["Utility"] + ["Transmitted_Nodes"] 
        writer.writerow(header)
        writer.writerows(log_data)
    
    average_utility = total_utility / n_episodes
    return average_utility

# Example usage
topology = Topology(12, "dumbbell")
average_utility = run_simulation(topology,
                                 n_steps=300,
                                 n_episodes=100,
                                 transmission_probs=0.2,
                                 log_file="simulation_logs.csv")
print(f"Average Utility over 100 episodes: {average_utility}")
# Average Utility over 100 episodes: [15.6847365]