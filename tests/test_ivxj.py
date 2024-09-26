import numpy as np
import pandas as pd
from ivxj.ivxj import ivxj

def test_ivxj():
    """Test ivxj function using the paper data"""
    for h in range(1,4):
        for i in range(1,5):
            # get test data
            y = np.loadtxt(f'tests/y_h{h}_i{i}.csv', dtype=np.float64, delimiter=',')
            dX = np.loadtxt(f'tests/dX_h{h}_i{i}.csv', dtype=np.float64, delimiter=',')
            rhoz = np.loadtxt(f'tests/rhoz_h{h}_i{i}.csv', dtype=np.float64, delimiter=',')
            Tlens = np.loadtxt(f'tests/Tlens_h{h}_i{i}.csv', delimiter=',').astype(int)
            X = np.loadtxt(f'tests/X_h{h}_i{i}.csv', dtype=np.float64, delimiter=',')
            expected_result = np.loadtxt(f'tests/result_h{h}_i{i}.csv', dtype=np.float64, delimiter=',')

            r1, r2, r3, r4 = ivxj(y, dX, rhoz, Tlens)
            r5, r6, r7, r8 = ivxj(y, X, rhoz, Tlens)

            actual_result = np.array([r1, r2, r3, r4, r5, r6, r7, r8], dtype= np.float64)

            gap = np.abs(actual_result - expected_result)
            relative_gap = np.abs((actual_result - expected_result)/expected_result)

            print(f"Result for h={h}, i={i}\n")
            result = np.column_stack((actual_result, expected_result, gap, relative_gap))
            df = pd.DataFrame(result, columns=["Actual", "Expected", "Error", "Relative Error"])
            print(df)
            max_value = df["Relative Error"].max()
            mean_value = df["Relative Error"].mean()
            print("\nMaximun value in Relative Error:", max_value)
            print("Mean value in Relative Error:", mean_value)

            assert (relative_gap < 0.4).all(), "relative_gap is too large!"
            assert (gap < 1).all(), "gap is too large!"

            ivxj_gap = np.array([gap[1],gap[5]])

            assert (ivxj_gap < 0.005).all(), "ivxj_gap is too large!"
    