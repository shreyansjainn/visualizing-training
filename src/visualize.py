
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import gravis as gv
import numpy as np
from typing import Dict, List


def visualize_hmm_loss(data, phase_col: str, epoch_col: str, loss_col: str,
                       hover_data: Dict = None):

    """Visualize HMM loss plot against the Epochs

    Args:
         data (pd.DataFrame): input data containg training runs
         phase_col (str): column name containing the HMM phases info
         epoch_col (str): column name containing Epoch info
         loss_col (str): column name for loss metric of the model
                        hover_data (Dict): dictionary with column names as
                        fields and bool
         value against them for visibility in tooltip during hover

    """

    fig = px.scatter(data, x=epoch_col, y=loss_col, 
                     color=data[phase_col].tolist(), hover_data=hover_data)

    fig.add_trace(go.Scatter(mode='lines',
                             x=data[epoch_col], y=data[loss_col],
                             line_color='black', line_width=1,
                             showlegend=False, line=dict(dash='dot')))
    fig.show()


def visualize_dag(transmat, hover_dict: Dict, HEX_VALUES: List):

    """Visualize HMM State Transitions

    Args:
         transmat (np.array): state transition matrix
         hover_dict (Dict): phase wise list of features to be shown in tooltip
                            on hover
         HEX_VALUES (List): List containing HEX values of the color pallette

    """

    n = transmat.shape[0]
    dot = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if np.round(transmat[i][j], 3) > 0:
                dot.add_node(i, label=str(i+1),
                             color=HEX_VALUES[i % len(HEX_VALUES)], size=10,
                             hover=hover_dict[str(i+1)])
                dot.add_edge(i, j, label=np.round(transmat[i][j], 3),
                             label_size=10, length=1000)
    return gv.d3(dot, edge_label_data_source='label', show_edge_label=True,
                 node_label_data_source='label', node_hover_neighborhood=True,
                 node_hover_tooltip=True)
