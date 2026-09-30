# Seeded epidemic model comparisons

Run from this checkout (the separate simulation integration PR). All populations
are synthetic and offline; workers use spawn, two processes, and explicit seeds.
The six independent binomial models are comparison implementations, not ABC/SMC
posterior oracles. They conserve each group's population and retain the entire
time grid after extinction. Global NumPy randomness is never consumed.

```sh
python -m validation.check_reference_models
python -m validation.run_notebooks compare_models.ipynb compare_models_population.ipynb
```

Notebook execution needs optional `nbclient`, `nbformat`, and `ipykernel`. Select a
kernel with the checkout's dependencies via `--kernel`; run_notebooks sets its
working directory to validation/. Install a kernel for a dedicated environment:

```sh
python -m pip install nbclient nbformat ipykernel
python -m ipykernel install --user --name epydemix-validation
python -m validation.run_notebooks compare_models.ipynb compare_models_population.ipynb --kernel epydemix-validation
```

Both notebooks cover SIR, SEIR and SIS. The single-group notebook also constructs
the same transition graph explicitly. All epydemix trial arrays must agree exactly
between sequential, parallel and explicit-model execution. Reference-model median
curves are qualitative comparisons, not tests that two unrelated engines draw
identical trajectories or that their distributions are statistically equivalent.

The smoke workload is 32 trials over 40 daily time points, seed 43. The grouped
case uses populations [5,000, 5,000], names young/old, and contact matrix
[[1,.2],[.2,1]]. Source, outputs, and execution counts are saved in the notebooks;
the execution JSONL records source and executed-file fingerprints. Full-precision
ensemble regression fixtures and their RNG-change alert are separate in tests/data/.
