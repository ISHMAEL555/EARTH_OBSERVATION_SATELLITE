"""
main.py

Single Entry Point for the ADCS Project
"""

import argparse
from run_all_scenarios import run_all, run_scenario_a, run_scenario_b, run_scenario_c, run_scenario_d


def main():
    parser = argparse.ArgumentParser(description="ADCS Hybrid RW + VSCMG Controller")
    parser.add_argument('--scenario', type=str, default='all',
                        choices=['all', 'a', 'b', 'c', 'd'],
                        help='Scenario to run')
    parser.add_argument('--robust', action='store_true', help='Run with robustness perturbations')

    args = parser.parse_args()

    print("="*70)
    print("ADCS Hybrid RW + VSCMG Controller - Assignment Submission")
    print("="*70)

    if args.scenario == 'all':
        run_all(robust=args.robust)
    elif args.scenario == 'a':
        run_scenario_a(robust=args.robust)
    elif args.scenario == 'b':
        run_scenario_b()
    elif args.scenario == 'c':
        run_scenario_c()
    elif args.scenario == 'd':
        run_scenario_d()

    print("\nSimulation completed successfully.")


if __name__ == "__main__":
    main()