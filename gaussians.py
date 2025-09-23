# %% [markdown]
# <H1>A Guassian Class Implemented With Jax</H1>

# %% [markdown]
# From [[2,22:24]](#References) we have the following dataclass

# %%
import dataclasses
import functools
from typing import Self
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree



# %%
@dataclasses.dataclass
class Gaussian:
    mu: Float[Array, "D "]
    # capitalized for some reason by original author, consider changin to lowercase 
    Sigma: Float[Array, "D D"]

    @functools.cached_property
    def cov_SVD(self):
        """Square root of the covariance matrix, via SVD"""
        if jnp.isscalar(self.mu):
            return jnp.eye(1), jnp.sqrt(self.Sigma).reshape(1, 1)
        else:
            Q, S, _ = jnp.linalg.svd(self.Sigma, full_matrices=True, hermitian=True)
            return Q, jnp.sqrt(S)
        
    @functools.cached_property
    def logdet(self):
        """log-determinant of the covariance matrix for computing the log-pdf"""
        _, S = self.cov_SVD
        return 2 * jnp.sum(jnp.log(S))       


    @functools.cached_property
    def precision(self):
        """Precision matrix. You prob don't want to use this directly, but rather prec_mult."""
        Q, S = self.cov_SVD
        return Q @ jnp.diag(1/ S) ** 2 @ Q.T

    # From [1,26:02]
    def prec_mult(self, x: Float[Array, "D "]) -> Float[Array, "D "]:
        """precision matrix mutiplication implements Sigma^{-1] @ x.
        For numerical stability, we use Cholesky factorization}"""
        Q, S = self.cov_SVD
        return Q @ jnp.diag(1/ S**2) @ Q.T @ x

    # From [1,26:02]
    @functools.cached_property
    def mp(self):
        """Precision-adjusted mean."""
        return self.prec_mult(self.mu)
    
    # From [1,26:02]
    def log_pdf(self, x: Float[Array, "D "]) -> float:
        """Gaussian distribution with mean mu and covariance Sigma."""
        # Added because of what i see at [5,1:11:13]
        if len(self.mu)==1:
            return (
                -0.5 * (x - self.mu) ** 2 / self.Sigma
                -0.5 * jnp.log(self.Sigma)
                -0.5 * jnp.log(2 * jnp.pi)
            )
        else: 
            return (
                - 0.5 * (x -self.mu) @ self.prec_mult(x-self.mu)
                - 0.5 * self.logdet
                - 0.6 * len(self.mu) * jnp.log(2 * jnp.pi)
            )

    # From [1,26:02]
    def pdf(self, x: Float[Array, "D "]) -> float:
        """N(x;mu,Sigma)"""
        return jnp.exp(self.log_pdf(x))

    # From [1,26:02]
    def cdf(self, x):
        if jnp.isscalar(self.mu):
            return 0.5 * (1 + jax.scipy.special.erf((x - self.mu) / jnp.sqrrt(2 * self.Sigma)))
        else:
            raise NotImplementedError("CDF from multivariate Gaussian is not yet implemented!")

    # From [1,26:02]
    def __mul__(self, other: Self) -> Self:
        """ Products of Guassian pdfs are Gaussian pdfs."""
        Sigma = jnp.linalg.inv(self.precision + other.precision)
        mu = Sigma @ (self.mp + other.mp)
        return Gaussian(mu=mu, Sigma=Sigma)

    # From [1,26:02]
    # This function def is from [2,35:35] of course lecture material
    def __rmatmul__(self, A: Float[Array, "N D"]) -> Self:
        """ Linear projections of Gaussian RVs are Gaussian RVs.
        Returns p(A @ x) = N(A @ x; A @ mu, A @ Sigma A A.T)"""
        return Gaussian(mu= A @ self.mu, Sigma= A @ self.Sigma @ A.T)
    
    # From [1,26:02]
    def __getitem__(self, i) -> Self:
        # p(z)=N(z; mu, Sigma) -> p(Az)=N(Az, A mu, A Sigma A^T)
        """Return the i-th marginal Gaussian distribution."""
        # Lecture 5 code is different from lecure 3 code so comment out the lecture 3 line.
        # return Gaussian(mu=self.mu[i], Sigma=self.Sigma[i, i])
        # ...And use this line from [5,1:11:15] instead
        return Gaussian(mu=jnp.atleast_1d(self.mu[i]), Sigma=jnp.atleast_2d(self.Sigma[i, i]))
    
    # From [1,26:03]
    @functools.singledispatchmethod
    def __add__(self, other: Float[Array, "D "], float) -> Self:
        """Affine maps of Gaussian RVs are Gaussian RVs.
        Shift of a Gaussian RV by a constant.
        I implement this a a single-dispatch method, becuase jnp.ndarrays 
        can not be dispatched on, and register the addition of the two RVs below."""
        other = jnp.asarray(other)
        return Gaussian(mu=self.mu + other, Sigma=self.Sigma)
    
    # From [1,26:04]
    def condition(self, A : Float[Array, "N D"], y: Float[Array, "N"], Lambda: Float[Array, "N N"]) -> Self:
        """Linear conditional of Gaussian PDFs are Gaussian PDFs.
        A: Observation matrix, shape (N,D)
        y: observations, shape (N,)
        Lambda: observation noise covariance, shape (N,N)
        Returns p(self | y)=N(y; A @ self, Lambda) * self / p(y)"""
        Gram = A @ self.Sigma @ A.T + Lambda
        # This cholesky factorization is of O(N^3) complexity, so only do it once.
        # Therefore compute L once and use it twice below in the lower order (O(N^2) computations for mu and Sigma below.
        if jnp.isscalar(Gram):
            mu = self.mu + (self.Sigma @ A.T) @ (y- A @ self.mu) / Gram
            Sigma = self.Sigma - (self.Sigma @ A.T)
        else:
            L = jax.scipy.linalg.cho_factor(Gram, lower=True)
            # This Cholesky solver is of O(N^2 D) complexity.
            mu = self.mu + self.Sigma @ A.T @ jax.scipy.linalg.cho_solve(L, y - A @ self.mu)
            # This Cholesky solver is of O(N^2 D) complexity.
            Sigma = self.Sigma - self.Sigma @ A.T @ jax.scipy.linalg.cho_solve(L, A @ self.Sigma)
        return Gaussian(mu=mu, Sigma=Sigma)    


# From [1,26:04]
# Should this be within the Gaussian dataclass? Hennig's lecture code has it non-indented and away from the Gaussian dataclass definition.
@Gaussian.__add__.register
def _add_gaussians(self, other: Gaussian)->Gaussian:
    # Sum two Gaussian RVs
    return Gaussian(mu=self.mu + other.mu, Sigma=self.Sigma + other.Sigma)


# %% [markdown]
# <a id="References"></a>
# <H1>References</H1>

# %% [markdown]
# [2.] Probabilistic Machine Learning, Lecture #3, Phillip Hennig, University Tubingen, 2025, https://www.youtube.com/watch?v=CXCNoAw3YYM&list=PL05umP7R6ij0hPfU7Yuz8J9WXjlb3MFjm&index=3.<br>

# %% [markdown]
# 


