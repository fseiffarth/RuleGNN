from Reproduce import experiments_fair_real_world, experiments_standard_real_world, experiments_synthetic, \
    experiments_baseline, experiments_ablation_distance, experiments_ablation_threshold, get_gnn_comparison_data


def main():
    experiments_fair_real_world.main()
    experiments_standard_real_world.main()
    experiments_synthetic.main()
    experiments_baseline.main_baseline()
    experiments_ablation_distance.main()
    experiments_ablation_threshold.main()

    get_gnn_comparison_data.main()
