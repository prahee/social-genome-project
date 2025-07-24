import scipy.io
import pandas as pd
import numpy as np
import os
from scipy.sparse import triu, issparse


input_folder = 'fb100'
output_folder = 'fb100-csv'
os.makedirs(output_folder, exist_ok=True)


all_nodes = []
all_edges = []

columns = ['gender', 'status', 'major', 'second_major', 'dorm', 'year', 'high_school']

for filename in os.listdir(input_folder):
    if filename.endswith('.mat'):
        school = filename.replace('.mat', '')
        mat_path = os.path.join(input_folder, filename)
        print(f'Processing {school}...')

        try:
            mat = scipy.io.loadmat(mat_path)
            if 'A' not in mat or 'local_info' not in mat:
                print(f"⚠️  Skipping {school} — missing 'A' or 'local_info'")
                continue

            A = mat['A']
            info = mat['local_info']

            if issparse(A):
                A = A.tocsr()

            #nodes
            nodes = pd.DataFrame(info, columns=columns)
            nodes['school'] = school
            nodes['node_id'] = nodes.index
            all_nodes.append(nodes)

            #edges
            triu_A = triu(A, k=1)
            sources, targets = triu_A.nonzero()
            edges = pd.DataFrame({
                'source': sources,
                'target': targets,
                'school': school
            })
            all_edges.append(edges)

        except Exception as e:
            print(f"Error processing {school}: {e}")
            continue

#concate & save
combined_nodes = pd.concat(all_nodes, ignore_index=True)
combined_edges = pd.concat(all_edges, ignore_index=True)

combined_nodes.to_csv(os.path.join(output_folder, 'facebook100_all_nodes.csv'), index=False)
combined_edges.to_csv(os.path.join(output_folder, 'facebook100_all_edges.csv'), index=False)