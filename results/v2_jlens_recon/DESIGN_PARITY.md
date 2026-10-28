# Parity vs layer-43 test (design before run). Countries, alpha 1.0, mask argument_through_end, same 33 prompts.
Configs: 5 layers 36-40 (odd, no 43); 6 layers 36-41 (even, no 43); 6 layers 38-43 (even, includes 43); 7 layers 37-43 (odd, includes 43).
Parity hypothesis predicts: 36-40 high, 36-41 ~0 (KL~0), 38-43 ~0, 37-43 high. Layer-43 hypothesis predicts: 36-40 high, 36-41 high, 38-43 ~0, 37-43 ~0. Other outcomes (mixed) are reported as such, not forced into either. "High" = at least 20/33 top-1; "~0" = at most 3/33. Between: inconclusive. Est <= $0.15.
Not claimed: that parity is the mechanism beyond these configs.
