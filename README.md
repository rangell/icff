# Interactive Clustering with Feature Feedback

Proof-of-concept interactive clustering algorithm with user feedback in the
form of "there-exists" constraints.

# Usage

```bash
python src/main.py [OPTIONS]
```

# Done
- Add more parameters to data generation
- Fix cut objective to match the paper
- Fix feedback generation (add strength parameter)


# TODO
- Different way of generating synthetic data?
- Sort order of equivalent assignments (assign step)?
- Must and shouldn't link constraints
- Support for real datasets
 - BoW document clustering
 - Entity Resolution
- Gurobi & or-tools integration?
- Plotting/tracking experiments
