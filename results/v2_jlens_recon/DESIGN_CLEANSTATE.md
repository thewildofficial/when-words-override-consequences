# Clean-state single-pass vs recurrent band edit (design, before run)
Question: does an even-count band edit work when each layer's coordinate change is computed from the CLEAN run's activation at that layer and added once (no dependence on the already-edited stream), versus the existing recurrent version where coordinates are read from the current (edited) activation?
Variant "clean_delta": at layer l, delta_l = (expected_clean - pre_clean) @ M_l^T computed from the activation of the unedited forward pass at l (same positions), then h_l' = h_l(edited) + delta_l. Recurrent (existing) = computed from h_l(edited).
Configs (countries, same 33 prompts, mask argument_through_end, alpha in {0.5, 0.75, 1.0}): bands 36-43 (8, even) and 36-42 (7, odd), each clean_delta; plus L40 (1 layer; clean and recurrent are identical, control). Existing recurrent results at alpha 1.0 are the comparison (8: 0/33; 7: 25/33); add alpha 0.5/0.75 recurrent for the same bands as strength-matched comparison. 
Pre-stated readouts: top-1 count, per-prompt top-1 and KL<1 count, median KL, median delta-RMS (both variants reported at equal alpha AND at equal measured dRMS where overlapping). Inconclusive region: 4-19 top-1 of 33 reported as inconclusive.
Not claimed: any J-space-paper protocol equivalence, any causal-use claim; scope: this operation, model, category.
Est <= $0.3.
