import numpy as np
from ivxj.ivxj import ivxj

def test_ivxj():
    """Test ivxj function using the paper data"""
    for h in range(1,4):
        for i in range(1,5):
            # get test data
            y = np.loadtxt(f'tests/y_h{h}_i{i}.csv', delimiter=',')
            dX = np.loadtxt(f'tests/dX_h{h}_i{i}.csv', delimiter=',')
            rhoz = np.loadtxt(f'tests/rhoz_h{h}_i{i}.csv', delimiter=',')
            Tlens = np.loadtxt(f'tests/Tlens_h{h}_i{i}.csv', delimiter=',').astype(int)
            X = np.loadtxt(f'tests/X_h{h}_i{i}.csv', delimiter=',')
            expected_result = np.loadtxt(f'tests/result_h{h}_i{i}.csv', delimiter=',')

            r1, r2, r3, r4 = ivxj(y, dX, rhoz, Tlens)
            r5, r6, r7, r8 = ivxj(y, X, rhoz, Tlens)

            actual_result = np.array([r1, r2, r3, r4, r5, r6, r7, r8])

            assert (actual_result == expected_result).all(), "ivxj cumpute incorrectly!"
    