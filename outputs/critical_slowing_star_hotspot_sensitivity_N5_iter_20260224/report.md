# Hotspot Sensitivity (N5 Star Refinement, Iterative)

## Peak Summary
```csv
hotspot_multiplier,chi,topology,metric,peak_lambda,peak_tau
2.0,4,cycle,tau_dephase_global,0.61,4.848484848484849
2.0,4,path,tau_dephase_probe,1.0,7.878787878787879
2.0,4,star,tau_dephase_probe,1.55,7.878787878787879
4.0,4,cycle,tau_dephase_global,0.61,36.96969696969697
4.0,4,path,tau_dephase_probe,1.0,9.696969696969695
4.0,4,star,tau_dephase_probe,1.7,15.151515151515152
2.0,5,cycle,tau_dephase_global,0.61,4.848484848484849
2.0,5,path,tau_dephase_probe,1.0,7.878787878787879
2.0,5,star,tau_dephase_probe,1.55,7.878787878787879
4.0,5,cycle,tau_dephase_global,0.61,24.848484848484848
4.0,5,path,tau_dephase_probe,1.0,17.575757575757578
4.0,5,star,tau_dephase_probe,1.55,17.575757575757578
```

## Next 10 Ranked Steps
1. Run star lambda extension to 1.75,1.80 at chi=4,5 and multipliers 3.0,4.0 to test if peak has moved beyond current window.
2. Run star fine-grid interpolation at chi=4, multiplier=3.0 on 1.58-1.68 step 0.01 to resolve local maximum shape.
3. Run star fine-grid interpolation at chi=5, multiplier=3.0 on 1.56-1.66 step 0.01 for chi-convergence against chi=4.
4. Run cycle triplet (0.55,0.61,0.65) under multipliers 2.0 and 4.0 at chi=4,5 to test whether cycle stability survives hotspot changes.
5. Run path local window (0.95,1.00,1.05) under multipliers 2.0 and 4.0 at chi=4,5 to verify path guard stability under protocol changes.
6. Fix tester.py lambda conversion bug (lines cited in AGENTS) and rerun canonical N4 lambda~1.058 comparator to quantify residual impact.
7. Add a dedicated regression script that compares dense vs iterative outputs on fixed seeds and fails if max metric diff >1e-8.
8. Run kappa sensitivity (0.05,0.1,0.2) on star refined window at chi=4,5 with multiplier=3.0 to separate hotspot-vs-penalty effects.
9. Run deltaB sensitivity (4.0,5.0,6.0) on the same star window to test if energy-spacing scale drives remaining peak drift.
10. Promote sparse backend in scan defaults via solver-backend=auto thresholds in run recipes/docs and deprecate dense-only guidance for N5.
