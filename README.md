# Existential Cluster Constraints

# Idea
- Define exhaustive objective and inference for clustering with existential
cluster constraints
- Starting place: use Craig's framework to define a distribution over clusters
and then add in how we will incorporate "there exists"-constraints into the
framework
- This avoids the greedy procedure of "assign & project", part of the problem
with the first attempt
- After we have this exhaustive framework, we need to figure out how to
approximate it efficiently and effectively (this also might require
an efficient implementation)
- Some suggestions: A*, beam search, ...


# Relevant papers
- [Filtering with Abstract Particles]: http://proceedings.mlr.press/v32/steinhardt14.pdf
- [Flattening a Hierarchical Clustering through Active Learning]: https://arxiv.org/pdf/1906.09458.pdf
- [Learning with feature feedback: from theory to practice]: https://cseweb.ucsd.edu/~dasgupta/papers/ff.pdf
- [Convex Combination Belief Propagation Algorithms]: https://arxiv.org/pdf/2105.12815.pdf
- [Factor Graphs and the Sum-Product Algorithm]: http://web.cs.iastate.edu/~honavar/factorgraphs.pdf 
- [Loopy Belief Propagation for Approximate Inference: An Empirical Study]: https://arxiv.org/pdf/1301.6725.pdf
- [Extending Factor Graphs so as to Unify Directed and Undirected Graphical Models]: https://arxiv.org/pdf/1212.2486.pdf
- [Compact Representation of Uncertainty in Clustering]: https://papers.nips.cc/paper/2018/file/29c4a0e4ef7d1969a94a5f4aadd20690-Paper.pdf
- [Cluster Trellis: Data Structures & Algorithms for Exact Inference in Hierarchical Clustering]: http://proceedings.mlr.press/v130/macaluso21a/macaluso21a.pdf
- [Exact and Approximate Hierarchical Clustering Using A*]: https://arxiv.org/pdf/2104.07061.pdf
