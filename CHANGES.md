To allow for multiclass segmentation, a graph-based pipeline was followed to separate the original labels into individual entities, and then each entity was manually labeled (1 - continuous, 2 - dashed, 3 - unmarked, 0 - background).

The pipeline to separate each object into separate entities consists of four steps:

1. thinning of segmentation mask pixels
2. building of a graph structure from 8-level connectivity using NetworkX library
3. identification of graph’s endpoints, i.e., pixels with connectivity on only one direction
4. classification of separate entities by assessing the number of connections, if the graph had more than two connections, a new entity was defined
5. the separate graph entities were used as markers at SciPy’s Watershed function, to maintain simultaneously the individual entities separation and the original lane thickness



The script used is included in the repository, for full transparency.

The labels contain some missclasifications, either due to fails in separating the labels into individual entities, or by human erros during the manual labeling.
