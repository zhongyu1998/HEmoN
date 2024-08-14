import h5py
import networkx as nx
import numpy as np

from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn.input_data import NiftiSpheresMasker
from scipy.sparse.csgraph import minimum_spanning_tree


def divide_interval(threshold, t):
    data_file = 'data/behavioral_ratings/raw_ratings.mat'
    data = h5py.File(data_file)

    rating_list = []
    for i in range(1, 13):
        rating_list.append(data[f'sub{i}'][:])
    ratings = np.array(rating_list).mean(axis=0)

    indices_list = [list() for _ in range(6)]
    for n, scores in enumerate(ratings.T):
        if scores.max() > threshold:
            c = scores.argmax()
            indices_list[c].append(n)

    start_list = [list() for _ in range(6)]
    end_list = [list() for _ in range(6)]

    # Determine the effective interval for emotion category `c`.
    # The effective interval is defined as the interval where the emotional rating
    # consistently exceeds the `threshold` for at least `t` consecutive time points.
    for c, indices in enumerate(indices_list):
        start = indices[0]
        end = indices[0]
        for i, n in enumerate(indices[:-t]):
            if indices[i+t] == n + t:
                end = indices[i+t]
            else:
                if end - start >= t:
                    start_list[c].append(start)
                    end_list[c].append(end)
                start = indices[i+1]
                end = indices[i+1]
        if end - start >= t:
            start_list[c].append(start)
            end_list[c].append(end)

    return start_list, end_list


def tree_construction(overall_flag, category=0):
    """
        overall_flag: whether to use the overall experience
        category: 0: happiness, 1: surprise, 2: fear, 3: sadness, 4: anger, 5: disgust
    """
    func_filename = 'data/fMRI_data/average_epi_nonlin_smooth_fwhm6.nii.gz'
    power = datasets.fetch_coords_power_2011()
    coords = np.vstack((power.rois["x"], power.rois["y"], power.rois["z"])).T

    masker = NiftiSpheresMasker(seeds=coords, radius=5.0, standardize=True, t_r=2)
    full_time_series = masker.fit_transform(func_filename)

    start_list, end_list = divide_interval(threshold=5, t=15-1)

    series_list = list()
    if overall_flag:  # overall experience of all basic emotions
        for c in range(6):
            for start, end in zip(start_list[c], end_list[c]):
                series_list.append(full_time_series[start:end-3])
    else:  # single experience of a specific basic emotion
        for start, end in zip(start_list[category], end_list[category]):
            series_list.append(full_time_series[start:end-3])
    emo_time_series = np.concatenate(series_list)

    correlation_measure = ConnectivityMeasure(kind="correlation")
    correlation_matrix = correlation_measure.fit_transform([emo_time_series])[0]
    spanning_tree = minimum_spanning_tree(1 - correlation_matrix)
    brain_tree = (spanning_tree + spanning_tree.T) > 0

    return brain_tree


def find_diameter_path(T):
    random_paths = nx.shortest_path(T, source=list(T.nodes)[0])
    source = max(random_paths, key=lambda i: len(random_paths[i]))
    source_paths = nx.shortest_path(T, source=source)
    target = max(source_paths, key=lambda i: len(source_paths[i]))
    diameter_path = source_paths[target]

    return diameter_path


def find_trunk(num_nodes, edge_list, max_level):
    trunk_list = list()

    T = nx.Graph()
    T.add_nodes_from(range(num_nodes))
    T.add_edges_from(edge_list)

    level = 0
    while T.nodes:
        level += 1
        if level <= max_level:
            level_list = list()  # a list of trunks at the current level

        isolated_nodes = list(nx.isolates(T))
        if level == 1 and len(isolated_nodes):
            level_list.append(isolated_nodes)
            T.remove_nodes_from(isolated_nodes)

        for c in list(nx.connected_components(T)):
            S = T.subgraph(c).copy()
            assert len(S.edges) != 0

            diameter_path = find_diameter_path(S)
            level_list.append(diameter_path)
            d_path_edges = list(zip(diameter_path[:-1], diameter_path[1:]))
            T.remove_edges_from(d_path_edges)
            T.remove_nodes_from(list(nx.isolates(T)))

        if level < max_level:
            trunk_list.append(level_list)
        elif not T.nodes:
            trunk_list.append(level_list)

    assert len(trunk_list) <= max_level

    return trunk_list
