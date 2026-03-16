
# Set this variable to the path of your CSV file, or leave as None to use the latest automatically
CSV_PATH = None

import unittest
import pandas as pd
import numpy as np

def get_latest_benchmark_csv():
    import glob, os
    result_dir = "results/benchmark_results"
    files = glob.glob(os.path.join(result_dir, "benchmark_results_*.csv"))
    if not files:
        raise FileNotFoundError("No benchmark results CSV found.")
    latest_csv = max(files, key=os.path.getctime)
    return latest_csv

class TestGammaSNRCombined(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if CSV_PATH:
            cls.df = pd.read_csv(CSV_PATH)
        else:
            cls.df = pd.read_csv(get_latest_benchmark_csv())

    def assert_gamma_decreases_with_increased_snr(self, solver, noise_type):
        df_combo = self.df[(self.df["solver"] == solver) & (self.df["noise_type"] == noise_type)]
        if df_combo.empty:
            self.skipTest(f"No data for solver={solver}, noise_type={noise_type}")
        failures = []
        group_cols = ["subject", "run_id", "orientation_type", "nnz"]
        for group_vals, group in df_combo.groupby(group_cols):
            snr = np.array(group["alpha_SNR"])
            gamma = np.array(group["gamma"])
            if len(snr) < 2:
                continue
            idx = np.argsort(snr)
            snr_sorted = snr[idx]
            gamma_sorted = gamma[idx]
            if not np.all(np.diff(gamma_sorted) < 0):
                failures.append((solver, noise_type, group_vals, snr_sorted, gamma_sorted))
        self.assertFalse(failures, f"Gamma did not decrease as SNR increased for {solver}/{noise_type}: {failures}")

    # Explicit tests for each solver/noise_type combination

    # --- BMN tests ---
    def test_gamma_decreases_with_increased_snr_BMN_oracle(self):
        self.assert_gamma_decreases_with_increased_snr('BMN', 'oracle')

    def test_gamma_decreases_with_increased_snr_BMN_baseline(self):
        self.assert_gamma_decreases_with_increased_snr('BMN', 'baseline')

    def test_gamma_decreases_with_increased_snr_BMN_spatial_CV(self):
        self.assert_gamma_decreases_with_increased_snr('BMN', 'spatial_CV')

    # --- Gamma Map tests ---
    def test_gamma_decreases_with_increased_snr_gamma_map_oracle(self):
        self.assert_gamma_decreases_with_increased_snr('gamma_map', 'oracle')

    def test_gamma_decreases_with_increased_snr_gamma_map_baseline(self):
        self.assert_gamma_decreases_with_increased_snr('gamma_map', 'baseline')
    
    def test_gamma_decreases_with_increased_snr_gamma_map_spatial_CV(self):
        self.assert_gamma_decreases_with_increased_snr('gamma_map', 'spatial_CV')

    # --- SFlex Gamma Map tests ---
    def test_gamma_decreases_with_increased_snr_sflex_gamma_map_oracle(self):
        self.assert_gamma_decreases_with_increased_snr('sflex_gamma_map', 'oracle')
        
    def test_gamma_decreases_with_increased_snr_sflex_gamma_map_baseline(self):
        self.assert_gamma_decreases_with_increased_snr('sflex_gamma_map', 'baseline')
        
    def test_gamma_decreases_with_increased_snr_sflex_gamma_map_spatial_CV(self):
        self.assert_gamma_decreases_with_increased_snr('sflex_gamma_map', 'spatial_CV')

    # --- SFlex Gamma Lambda Map tests ---
    def test_gamma_decreases_with_increased_snr_sflex_gamma_lambda_map_joint_learning(self):
        self.assert_gamma_decreases_with_increased_snr('sflex_gamma_lambda_map', 'adaptive_joint_learning')


    # --- General tests ---
    def test_snr_not_nan(self):
        self.assertFalse(self.df["alpha_SNR"].isnull().any(), "SNR contains NaN values.")

    def test_gamma_not_nan(self):
        df_non_eloreta = self.df[self.df["solver"] != "eloreta"]
        gamma_col = df_non_eloreta["gamma"]
        if gamma_col.empty:
            self.skipTest("Gamma column is empty for non-eloreta solvers.")
        self.assertFalse(gamma_col.isnull().any(), "Gamma contains NaN values for non-eloreta solvers.")

    def test_gamma_positive(self):
        df_non_eloreta = self.df[self.df["solver"] != "eloreta"]
        gamma_col = df_non_eloreta["gamma"]
        if gamma_col.empty:
            self.skipTest("Gamma column is empty for non-eloreta solvers.")
        self.assertTrue((gamma_col > 0).all(), "Gamma contains non-positive values for non-eloreta solvers.")

if __name__ == "__main__":
    class StatusLoggingResult(unittest.TextTestResult):
        def addSuccess(self, test):
            super().addSuccess(test)
            print(f"[PASS]: {test._testMethodName}")
        def addFailure(self, test, err):
            super().addFailure(test, err)
            print(f"[FAIL]: {test._testMethodName}")
        def addError(self, test, err):
            super().addError(test, err)
            print(f"[ERROR]: {test._testMethodName}")
        def addSkip(self, test, reason):
            super().addSkip(test, reason)
            print(f"[SKIP]: {test._testMethodName} ({reason})")

    runner = unittest.TextTestRunner(resultclass=StatusLoggingResult)
    runner.run(unittest.defaultTestLoader.loadTestsFromTestCase(TestGammaSNRCombined))
