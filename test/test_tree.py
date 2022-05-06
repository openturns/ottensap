
import numpy as np
import tensap
import ottensap
import pytest

@pytest.mark.skip
def test_tree():
    # %% Function to approximate
    CHOICE = 2
    if CHOICE == 1:
        ORDER = 5
        X = tensap.RandomVector(tensap.NormalRandomVariable(), ORDER)

        def fun(x):
            return 1 / (10 + x[:, 0] + 0.5*x[:, 1])**2
    elif CHOICE == 2:
        fun, X = tensap.multivariate_functions_benchmark('borehole')
        ORDER = X.size
    else:
        raise ValueError('Bad function choice.')

    # %% Approximation basis
    DEGREE = 8
    ORTHONORMAL_BASES = True
    if ORTHONORMAL_BASES:
        BASES = [tensap.PolynomialFunctionalBasis(
            x.orthonormal_polynomials(), range(DEGREE+1)) for
                x in X.random_variables]
    else:
        BASES = [tensap.PolynomialFunctionalBasis(
            tensap.CanonicalPolynomials(), range(DEGREE+1)) for
                x in X.random_variables]
    BASES = tensap.FunctionalBases(BASES)

    # %% Training and test samples
    NUM_TRAIN = 100
    X_TRAIN = X.random(NUM_TRAIN)
    Y_TRAIN = fun(X_TRAIN)

    NUM_TEST = 10000
    X_TEST = X.random(NUM_TEST)
    Y_TEST = fun(X_TEST)

    # %% Tree-based tensor format
    # Tensor format
    # 1 - Random tree and active nodes
    # 2 - Tensor-Train
    # 3 - Hierarchial Tensor-Train
    # 4 - Binary tree
    CHOICE = 3
    if CHOICE == 1:
        print('Random tree with active nodes')
        ARITY = [2, 4]
        TREE = tensap.DimensionTree.random(ORDER, ARITY)
        IS_ACTIVE_NODE = np.full(TREE.nb_nodes, True)
        SOLVER = tensap.TreeBasedTensorLearning(TREE, IS_ACTIVE_NODE,
                                                tensap.SquareLossFunction())
    elif CHOICE == 2:
        print('Tensor-train format')
        SOLVER = tensap.TreeBasedTensorLearning.tensor_train(
            ORDER, tensap.SquareLossFunction())
    elif CHOICE == 3:
        print('Tensor Train Tucker')
        SOLVER = tensap.TreeBasedTensorLearning.tensor_train_tucker(
            ORDER, tensap.SquareLossFunction())
    elif CHOICE == 4:
        print('Binary tree')
        TREE = tensap.DimensionTree.balanced(ORDER)
        IS_ACTIVE_NODE = np.full(TREE.nb_nodes, True)
        SOLVER = tensap.TreeBasedTensorLearning(TREE, IS_ACTIVE_NODE,
                                                tensap.SquareLossFunction())
    else:
        raise NotImplementedError('Not implemented.')

    # %% Random shuffling of the dimensions associated to the leaves
    RANDOMIZE = True
    if RANDOMIZE:
        SOLVER.tree.dim2ind = np.random.permutation(SOLVER.tree.dim2ind)
        SOLVER.tree = SOLVER.tree.update_dims_from_leaves()

    # %% Learning in tree-based tensor format
    SOLVER.bases = BASES
    SOLVER.bases_eval = BASES.eval(X_TRAIN)
    SOLVER.training_data = [None, Y_TRAIN]

    SOLVER.tolerance['on_stagnation'] = 1e-6
    SOLVER.tolerance['on_error'] = 1e-6

    SOLVER.initialization_type = 'canonical'

    SOLVER.linear_model_learning.regularization = False
    SOLVER.linear_model_learning.basis_adaptation = True
    SOLVER.linear_model_learning.error_estimation = True

    SOLVER.test_error = True
    SOLVER.test_data = [X_TEST, Y_TEST]
    # SOLVER.bases_eval_test = BASES.eval(X_TEST)

    SOLVER.rank_adaptation = True
    SOLVER.rank_adaptation_options['max_iterations'] = 20
    SOLVER.rank_adaptation_options['theta'] = 0.8
    SOLVER.rank_adaptation_options['early_stopping'] = True
    SOLVER.rank_adaptation_options['early_stopping_factor'] = 10

    SOLVER.tree_adaptation = True
    SOLVER.tree_adaptation_options['max_iterations'] = 1e2
    # SOLVER.tree_adaptation_options['force_rank_adaptation'] = True

    SOLVER.alternating_minimization_parameters['stagnation'] = 1e-10
    SOLVER.alternating_minimization_parameters['max_iterations'] = 5

    SOLVER.display = True
    SOLVER.alternating_minimization_parameters['display'] = False

    SOLVER.model_selection = True
    SOLVER.model_selection_options['type'] = 'test_error'

    F, OUTPUT = SOLVER.solve()

    x = X.mean()
    y_ref = F.eval(x)
    otf = ottensap.TensapFunction(F)
    y = otf(x)
    print(x, y, y_ref)
    assert_almost_equal(y, y_ref, decimal=3)
