"""
Graph construction utilities.

`build_batadal_graph` parses the EPANET `.inp` topology file of the
BATADAL benchmark via `wntr` and returns a directed graph in which
each pressure sensor is a node and each flow sensor is an edge. Pipe
parameters (length, roughness coefficient, diameter) come from the
EPANET file when available; when a sensor is attached to a non-pipe
link (e.g., a pump or valve, for which the Hazen-Williams equation is
not parameterised), the function falls back to a documented constant
placeholder.

`create_pyg_data` packages an arbitrary (H, Q) time series into a list
of PyTorch Geometric `Data` objects with the supplied graph structure.
"""
import torch
import wntr
from torch_geometric.data import Data


_DEFAULT_PIPE = (1.0, 130.0, 0.25)  # (length [m], roughness, diameter [m])


def _sensor_to_network_name(col_name):
    for prefix in ('L_', 'P_', 'F_'):
        if col_name.startswith(prefix):
            return col_name[len(prefix):]
    return col_name


def build_batadal_graph(inp_file_path, h_cols, q_cols):
    wn = wntr.network.WaterNetworkModel(inp_file_path)

    monitored_nodes = [_sensor_to_network_name(c) for c in h_cols]
    node_idx = {name: i for i, name in enumerate(monitored_nodes)}

    edge_index_list = []
    edge_attr_list = []

    for q_col in q_cols:
        link_name = _sensor_to_network_name(q_col)
        try:
            link = wn.get_link(link_name)
            u_name = link.start_node_name
            v_name = link.end_node_name
        except KeyError:
            u_name = monitored_nodes[0]
            v_name = monitored_nodes[1 % len(monitored_nodes)]

        u = node_idx.get(u_name, 0)
        v = node_idx.get(v_name, 1 % len(monitored_nodes))
        edge_index_list.append([u, v])

        if hasattr(link, 'length') and link.length > 0:
            L = float(link.length)
            D = float(link.diameter)
            C = float(link.roughness)
        else:
            L, C, D = _DEFAULT_PIPE

        edge_attr_list.append([L, C, D])

    edge_index = torch.tensor(edge_index_list,
                               dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    link_names = [_sensor_to_network_name(c) for c in q_cols]
    return edge_index, edge_attr, monitored_nodes, link_names


def create_pyg_data(H_tensor, Q_tensor, edge_index, edge_attr, labels=None):
    """Wrap a (H, Q) time series into a list of PyG `Data` objects."""
    data_list = []
    for i in range(H_tensor.shape[0]):
        h, q = H_tensor[i], Q_tensor[i]
        y = (torch.tensor([labels[i]], dtype=torch.float32)
             if labels is not None else None)
        data_list.append(Data(
            x=h.unsqueeze(1),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y, h=h, q=q,
        ))
    return data_list
