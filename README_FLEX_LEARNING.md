
# Flex-Learning Workflow
1) Batch + save solutions:
   python run_optimization_batch.py --houses 1 --alpha 0.2 --inc "10-15" --dec "18-23,6-8" --l_import 1.0 --save_solutions --outdir outputs_batch
2) Learn windows:
   python flex_learning.py --solutions_dir outputs_batch/solutions --out_json outputs_batch/learned_windows.json --prob_thresh 0.4
3) Re-run with learned windows:
   python run_optimization_batch.py --houses 1 --alpha 0.2 --l_import 1.0 --windows_json outputs_batch/learned_windows.json --outdir outputs_batch_v2
