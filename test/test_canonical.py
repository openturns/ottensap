import tensap
import ottensap
from numpy.testing import assert_almost_equal
import numpy as np
np.random.seed(0)


def test_canonical():
    # From tensap tutorial_tensor_learning_CanonicalTensorLearning.py:
    fun, X = tensap.multivariate_functions_benchmark('borehole')
    ORDER = X.size
    DEGREE = 8
    BASES = [tensap.PolynomialFunctionalBasis(X_TRAIN.orthonormal_polynomials(),
             range(DEGREE+1)) for X_TRAIN in X.random_variables]
    BASES = tensap.FunctionalBases(BASES)
    NUM_TRAIN = 1000
    X_TRAIN = X.random(NUM_TRAIN)
    Y_TRAIN = fun(X_TRAIN)

    NUM_TEST = 10000
    X_TEST = X.random(NUM_TEST)
    Y_TEST = fun(X_TEST)

    SOLVER = tensap.CanonicalTensorLearning(ORDER, tensap.SquareLossFunction())
    SOLVER.rank_adaptation = True
    SOLVER.initialization_type = 'mean'
    SOLVER.tolerance['on_error'] = 1e-6
    SOLVER.alternating_minimization_parameters['stagnation'] = 1e-8
    SOLVER.alternating_minimization_parameters['max_iterations'] = 100
    SOLVER.linear_model_learning.regularization = False
    SOLVER.linear_model_learning.basis_adaptation = True
    SOLVER.bases = BASES
    SOLVER.training_data = [X_TRAIN, Y_TRAIN]
    SOLVER.display = True
    SOLVER.alternating_minimization_parameters['display'] = False
    SOLVER.test_error = True
    SOLVER.test_data = [X_TEST, Y_TEST]
    SOLVER.alternating_minimization_parameters['one_by_one_factor'] = False
    SOLVER.alternating_minimization_parameters['inner_loops'] = 2
    SOLVER.alternating_minimization_parameters['random'] = False
    SOLVER.rank_adaptation_options['max_iterations'] = 4
    SOLVER.model_selection = True
    SOLVER.model_selection_options['type'] = 'test_error'

    F, OUTPUT = SOLVER.solve()

    x = X.mean()
    y_ref = F.eval(x)
    otf = ottensap.TensapFunction(F)
    y = otf(x)
    print(x, y, y_ref)
    assert_almost_equal(y, y_ref, decimal=3)
