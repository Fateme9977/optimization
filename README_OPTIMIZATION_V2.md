
# Optimization V2 — Quick Guide
Install:
    pip install pandas numpy pulp

Run batch (test one house):
    python run_optimization_batch.py --houses 1 --alpha 0.2 --inc "10-15" --dec "18-23,6-8" --l_import 1.0 --outdir outputs_batch

Save solutions and learn windows:
    python run_optimization_batch.py --houses 1 --alpha 0.2 --inc "10-15" --dec "18-23,6-8" --l_import 1.0 --save_solutions --outdir outputs_batch
    python flex_learning.py --solutions_dir outputs_batch/solutions --out_json outputs_batch/learned_windows.json --prob_thresh 0.4
    python run_optimization_batch.py --houses 1 --alpha 0.2 --l_import 1.0 --windows_json outputs_batch/learned_windows.json --outdir outputs_batch_v2
