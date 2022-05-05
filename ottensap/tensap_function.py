import tensap
import openturns as ot
import numpy as np

def TensapFunction(ft, x):
    if not isinstance(ft, tensap.FunctionalTensor):
        raise NotImplementedError(f"Unknown type: {ft.__class__.__name__}")
    if not isinstance(ft.bases, tensap.FunctionalBases):
        raise NotImplementedError(f"Unknown bases type: {ft.bases.__class__.__name__}")
    marginals = []
    for base in ft.bases.bases:
        if not isinstance(base, tensap.PolynomialFunctionalBasis):
            raise NotImplementedError(f"Unknown base type: {base.__class__.__name__}")
        measure = base.measure
        if isinstance(measure, tensap.UniformRandomVariable):
            marginals.append(ot.Uniform(measure.inf, measure.sup))
        elif isinstance(measure, tensap.NormalRandomVariable):
            marginals.append(ot.Normal(measure.mu, measure.sigma))
        elif isinstance(measure, tensap.DiscreteRandomVariable):
            marginals.append(ot.Histogram(measure.values.flatten(), measure.probabilities))
        else:
            raise NotImplementedError(f"Unknown measure type: {measure.__class__.__name__}")
    distribution = ot.ComposedDistribution(marginals)
    polyColl = [ot.StandardDistributionPolynomialFactory(ot.AdaptiveStieltjesAlgorithm(marginal)) for marginal in marginals]
    dimension = distribution.getDimension()
    matrices = [[polyColl[i].build(j)(x[0]) for j in range(dimension + 1)] for i in range(dimension)]
    dims = list(range(dimension))
    if isinstance(ft.tensor, tensap.CanonicalTensor):
        # tensor_matrix_product
        space = ft.tensor.space
        for i, dim in enumerate(dims):
            space[dim] = np.matmul(matrices[i], space[dim])
        # eval_diag
        out = space[dims[0]]
        for k in dims[1:]:
            out *= space[k]
        out = np.matmul(out, ft.tensor.core.data)
        return [out]
    elif isinstance(ft.tensor, tensap.FullTensor):
        raise NotImplementedError
    elif isinstance(ft.tensor, tensap.SparseTensor):
        raise NotImplementedError

