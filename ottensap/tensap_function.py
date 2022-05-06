import tensap
import openturns as ot
import numpy as np

class CanonicalTensorFunction(ot.OpenTURNSPythonFunction):
    def __init__(self, polyColl, space, data, dims):
        super(CanonicalTensorFunction, self).__init__(len(dims), 1)
        self.polyColl_ = polyColl
        self.space_ = space
        self.data_ = data
        self.dims_ = dims

    def _exec(self, x):
        dimension = len(self.dims_)
        matrices = [[self.polyColl_[i].build(j)(x[i]) for j in range(dimension + 1)] for i in range(dimension)]
        # CanonicalTensor.tensor_matrix_product
        space = self.space_
        for i, dim in enumerate(self.dims_):
            space[dim] = matrices[i] @ space[dim]
        # CanonicalTensor.eval_diag
        out = space[self.dims_[0]]
        for k in self.dims_[1:]:
            out *= space[k]
        out = out @ self.data_
        return [out]


def TensapFunction(ft):
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
    dims = list(range(distribution.getDimension()))
    if isinstance(ft.tensor, tensap.CanonicalTensor):
        return ot.Function(CanonicalTensorFunction(polyColl, ft.tensor.space, ft.tensor.core.data, dims))
    elif isinstance(ft.tensor, tensap.FullTensor):
        raise NotImplementedError
    elif isinstance(ft.tensor, tensap.SparseTensor):
        raise NotImplementedError

