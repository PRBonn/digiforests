# DBH Estimation Script

This folder contains a script for estimating Diameter at Breast Height (DBH) of trees using cylinder fitting on point cloud data from the DigiForests dataset.

## Usage

The script can be run from the command line with various options:

```bash
python fit_cylinders_for_dbh.py <exp_name> <inventory_csv> [OPTIONS]
```

### Required Arguments

- `exp_name`: Name of the experiment for tracking results
- `inventory_csv`: Path to the CSV file containing ground truth inventory data

### Optional Arguments

- `--plot-folder`: Path to a single plot's data
- `--label-folder`: Path to semantic labels (if not in plot_folder)
- `--aggregated-cloud-pth`: Path to a pre-aggregated point cloud file
- `--glob-dir`: Path to a directory for processing multiple plots
- `--visualize`: Enable visualization of results (may increase runtime)

## Workflow

1. The script processes either a single plot or multiple plots (if `glob_dir` is provided)
2. For each plot, it performs cylinder fitting to estimate tree DBH
3. Calculates RMSE (Root Mean Square Error) for DBH estimation
4. Computes average RMSE across all processed plots
5. Appends results to `cylinder_fitting_results.csv` in the current directory

## Output

Results are saved in `cylinder_fitting_results.csv` with the following columns:

- `run_name`: Experiment name
- `run_num`: Run number
- `[plot_ids]`: RMSE for each processed plot
- `avg_rmse`: Average RMSE across all plots

## Notes

- If `glob_dir` is provided, it takes precedence over `plot_folder` and `aggregated_cloud_pth`
- Visualization is intended for debugging and may significantly increase runtime
