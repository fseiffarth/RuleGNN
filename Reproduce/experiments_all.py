import click

from Reproduce import experiments_fair_real_world, experiments_standard_real_world, experiments_synthetic, \
    experiments_baseline, experiments_ablation_distance, experiments_ablation_threshold, get_gnn_comparison_data


# add arguments to the main function if needed using click
@click.command()
@click.option('--num_threads', default=-1, help='Number of tasks to run in parallel')
def main(num_threads):
    experiments_fair_real_world.main_fair_real_world(num_threads)
    experiments_standard_real_world.main_standard_real_world(num_threads)
    experiments_synthetic.main_synthetic(num_threads)
    experiments_ablation_distance.main_ablation_distance(num_threads)
    experiments_ablation_threshold.main_ablation_threshold(num_threads)
    experiments_baseline.main_baseline(num_threads)
    get_gnn_comparison_data.main()


if __name__ == '__main__':
    main()
