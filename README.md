# Interactive Clustering with Feature Feedback

Proof-of-concept interactive clustering algorithm with user feedback in the
form of "there-exists" constraints.

# Usage

```bash
python src/icff.py [OPTIONS]
```

# Done
- Add more parameters to data generation
- Fix cut objective to match the paper
- Fix feedback generation (add strength parameter)
- Sort order of equivalent assignments (assign step)?
- Must and shouldn't link constraints

# TODO
- Fits computation for ml/sl
- Support for real datasets
 - BoW document clustering
 - Entity Resolution
- Different way of generating synthetic data?
- Gurobi & or-tools integration?
- Plotting/tracking experiments

# Plan B's
- Change linkage / objective
  - average
  - single
  - generic linkage function

# Paper TODOs
- Change definition of the features we are providing feedback about are not
  the only features going into the similarity function
- 
