import math
import time

import numpy as np
from scipy.linalg import block_diag
from partner_juggling.tracker.data_types import CovarianceMatrix



class LinearGaussianMeasurementModel():
    r"""This is a class implementation of a time-invariant 1D
    Linear-Gaussian Measurement Model.

    The model is described by the following equations:

    .. math::

      y_t = H_k*x_t + v_k,\ \ \ \   v(k)\sim \mathcal{N}(0,R)

    where ``H_k`` is a (:py:attr:`~ndim_meas`, :py:attr:`~ndim_state`) \
    matrix and ``v_k`` is Gaussian distributed.

    """
    def __init__(self, ndim_state, mapping, noise_covar):
        if not isinstance(noise_covar, CovarianceMatrix):
            noise_covar = CovarianceMatrix(noise_covar)
        self.ndim_state = ndim_state
        self.mapping = mapping
        self.noise_covar = noise_covar

    @property
    def ndim_meas(self):
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return len(self.mapping)

    def matrix(self, **kwargs):
        """Model matrix :math:`H(t)`

        Returns
        -------
        :class:`numpy.ndarray` of shape \
        (:py:attr:`~ndim_meas`, :py:attr:`~ndim_state`)
            The model matrix evaluated given the provided time interval.
        """

        model_matrix = np.zeros((self.ndim_meas, self.ndim_state))
        for dim_meas, dim_state in enumerate(self.mapping):
            if dim_state is not None:
                model_matrix[dim_meas, dim_state] = 1

        return model_matrix

    def function(self, state, noise=False, **kwargs):
        """Model function :math:`h(t,x(t),w(t))`

        Parameters
        ----------
        state: :class:`~.State`
            An input state
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `False`, in which case no noise will be added
            if 'True', the output of :meth:`~.Model.rvs` is added)

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_meas`, 1)
            The model function evaluated given the provided time interval.
        """

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0

        return self.matrix(**kwargs)@state.state_vector + noise

    def covar(self, **kwargs):
        """Returns the measurement model noise covariance matrix.

        Returns
        -------
        :class:`~.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_meas`, :py:attr:`~ndim_meas`)
            The measurement noise covariance.
        """

        return self.noise_covar


class LinearGaussianTransitionModel():

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of model state dimensions.
        """

        return self.matrix().shape[0]


class CombinedLinearGaussianTransitionModel():
    r"""Combine multiple models into a single model by stacking them.

    The assumption is that all models are Linear and Gaussian.
    Time Variant, and Time Invariant models can be combined together.
    If any of the models are time variant the keyword argument "time_interval"
    must be supplied to all methods
    """    
    def matrix(self, **kwargs):
        """Model matrix :math:`F`

        Returns
        -------
        : :class:`numpy.ndarray` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
        """

        transition_matrices = [
            model.matrix(**kwargs) for model in self.model_list]
        return block_diag(*transition_matrices)


class ConstantNthDerivative():
    r"""Discrete model based on the Nth derivative with respect to time being
    constant, to set derivative use keyword argument
    :attr:`constant_derivative`

     The model is described by the following SDEs:

        .. math::
            :nowrap:

            \begin{eqnarray}
                dx^{(N-1)} & = & x^{(N)} dt & | {(N-1)th \ derivative \ on \
                X-axis (m)} \\
                dx^{(N)} & = & q\cdot dW_t,\ W_t \sim \mathcal{N}(0,q^2) & | \
                Nth\ derivative\ on\ X-axis (m/s^{N})
            \end{eqnarray}

    It is hard to represent the matrix form of these due to the fact that they
    vary with N, examples for N=1 and N=2 can be found in the
    :class:`~.ConstantVelocity` and :class:`~.ConstantAcceleration` models
    respectively. To aid visualisation of :math:`F_t` the elements are
    calculated as the terms of the taylor expansion of each state variable.
    """
    def __init__(self, noise_diff_coeff, constant_derivative):
        self.noise_diff_coeff = noise_diff_coeff
        self.constant_derivative = constant_derivative

    @property
    def ndim_state(self):
        return self.constant_derivative + 1

    def matrix(self, time_interval, **kwargs):
        # time_interval_sec = time_interval.total_seconds()
        N = self.constant_derivative
        Fmat = np.zeros((N + 1, N + 1))
        dt = time_interval
        for i in range(0, N + 1):
            for j in range(i, N + 1):
                Fmat[i, j] = (dt ** (j - i)) / math.factorial(j - i)

        return Fmat

    def covar(self, time_interval, **kwargs):
        # time_interval_sec = time_interval.total_seconds()
        dt = time_interval
        N = self.constant_derivative
        if N == 1:
            covar = np.array([[dt**3 / 3, dt**2 / 2],
                              [dt**2 / 2, dt]])
        else:
            Fmat = self.matrix(time_interval, **kwargs)
            Q = np.zeros((N + 1, N + 1))
            Q[N, N] = 1
            igrand = Fmat @ Q @ Fmat.T
            covar = np.zeros((N + 1, N + 1))
            for l in range(0, N + 1):  # noqa: E741
                for k in range(0, N + 1):
                    covar[l, k] = (igrand[l, k]*dt / (1 + N*2 - l - k))
        covar *= self.noise_diff_coeff
        return CovarianceMatrix(covar)


class ConstantVelocity(ConstantNthDerivative):
    r"""This is a class implementation of a discrete, time-variant 1D
    Linear-Gaussian Constant Velocity Transition Model.

    The target is assumed to move with (nearly) constant velocity, where
    target acceleration is modelled as white noise.

    The model is described by the following SDEs:

        .. math::
            :nowrap:

            \begin{eqnarray}
                dx_{pos} & = & x_{vel} d & | {Position \ on \
                X-axis (m)} \\
                dx_{vel} & = & q\cdot dW_t,\ W_t \sim \mathcal{N}(0,q^2) & | \
                Speed on\ X-axis (m/s)
            \end{eqnarray}

    Or equivalently:

        .. math::
            x_t = F_t x_{t-1} + w_t,\ w_t \sim \mathcal{N}(0,Q_t)

    where:

        .. math::
            x & = & \begin{bmatrix}
                        x_{pos} \\
                        x_{vel}
                \end{bmatrix}

        .. math::
            F_t & = & \begin{bmatrix}
                        1 & dt\\
                        0 & 1
                \end{bmatrix}

        .. math::
            Q_t & = & \begin{bmatrix}
                        \frac{dt^3}{3} & \frac{dt^2}{2} \\
                        \frac{dt^2}{2} & dt
                \end{bmatrix} q
    """
    @property
    def constant_derivative(self):
        """For constant velocity, this is 1."""
        return 1


class ConstantAcceleration(ConstantNthDerivative):
    r"""This is a class implementation of a discrete, time-variant 1D Constant
    Acceleration Transition Model.

    The target acceleration is modeled as a zero-mean white noise random
    process.

    The model is described by the following SDEs:

        .. math::
            :nowrap:

            \begin{eqnarray}
                dx_{pos} & = & x_{vel} d & | {Position \ on \
                X-axis (m)} \\
                dx_{vel} & = & x_{acc} d & | {Speed \
                on\ X-axis (m/s)} \\
                dx_{acc} & = & q W_t,\ W_t \sim
                \mathcal{N}(0,q^2) & | {Acceleration \ on \ X-axis (m^2/s)}

            \end{eqnarray}

    Or equivalently:

        .. math::
            x_t = F_t x_{t-1} + w_t,\ w_t \sim \mathcal{N}(0,Q_t)

    where:

        .. math::
            x & = & \begin{bmatrix}
                         x_{pos} \\
                         x_{vel} \\
                         x_{acc}
                    \end{bmatrix}

        .. math::
            F_t & = & \begin{bmatrix}
                           1 & dt & \frac{dt^2}{2} \\
                           0 & 1 & dt \\
                           0 & 0 & 1
                      \end{bmatrix}

        .. math::
            Q_t & = & \begin{bmatrix}
                        \frac{dt^5}{20} & \frac{dt^4}{8} & \frac{dt^3}{6} \\
                        \frac{dt^4}{8} & \frac{dt^3}{3} & \frac{dt^2}{2} \\
                        \frac{dt^3}{6} & \frac{dt^2}{2} & dt
                      \end{bmatrix} q
    """
    @property
    def constant_derivative(self):
        """For constant acceleration, this is 2."""
        return 2


class ConstantAccelerationTransitionModel():
    def __init__(self, noise_diff_coeff, n_dims, constant_dt = None):
        self.noise_diff_coeff = noise_diff_coeff
        self.n_dims = n_dims
        self.transition_matrix = block_diag(*[np.zeros((3,3)) for _ in range(n_dims)])
        self.constant_dt = constant_dt

        if constant_dt is not None:
            self.transition_matrix = self.matrix(constant_dt)
            self.covariance_matrix = self.covar(constant_dt)


    
    def matrix(self, time_interval, **kwargs):
        # Fmat = np.zeros((3,3))
        # for i in range(0, 3):
        #     for j in range(i, 3):
        #         self.Fmat[i, j] = (time_interval ** (j - i)) / math.factorial(j - i)
        # start = time.time()
        # self.transition_matrix = block_diag(*[self.Fmat] * self.n_dims)
        transition_matrix = np.array([
            [1, time_interval, time_interval**2/2,  0,          0,              0,              0,      0,              0],
            [0,    1,          time_interval,       0,          0,              0,              0,      0,              0],
            [0,    0,              1,               0,          0,              0,              0,      0,              0],
            [0,    0,              0,               1,      time_interval,  time_interval**2/2, 0,      0,              0],
            [0,    0,              0,               0,          1,          time_interval,      0,      0,              0],
            [0,    0,              0,               0,          0,              1,              0,      0,              0],
            [0,    0,              0,               0,          0,              0,              1,  time_interval,time_interval**2/2],
            [0,    0,              0,               0,          0,              0,              0,      1,          time_interval],
            [0,    0,              0,               0,          0,              0,              0,      0,              1]])
        # end = time.time()
        # print("Transition matrix computation time: ", (end - start)*1000,"ms")
        return transition_matrix

    # def covar(self, time_interval, **kwargs):
    #     # start = time.time()
    #     Q = np.zeros((3, 3))
    #     Q[2, 2] = 1
    #     self.Fmat = np.array([[1, time_interval, time_interval**2/2],
    #                          [0,    1,          time_interval],
    #                          [0,    0,              1]])
    #     igrand = self.Fmat @ Q @ self.Fmat.T
    #     covar = np.zeros((3, 3))
    #     for l in range(0, 3):
    #         for k in range(0, 3):
    #             covar[l, k] = (igrand[l, k]*time_interval / (1 + 2*2 - l - k))
    #     covar *= self.noise_diff_coeff
    #     covariance_matrix = block_diag(*([covar] * self.n_dims))
    #     # end = time.time()
    #     # print("Covariance computation time: ", (end - start)*1000,"ms")
    #     return covariance_matrix
    
    def covar(self, time_interval, **kwargs):
        # start = time.time()
        dt = time_interval
        covariance_matrix = np.array([
            [dt**5/20,    dt**4/8,    dt**3/6,    0,          0,          0,          0,          0,          0],
            [dt**4/8,     dt**3/3,    dt**2/2,    0,          0,          0,          0,          0,          0],
            [dt**3/6,     dt**2/2,    dt,         0,          0,          0,          0,          0,          0],
            [0,           0,          0,          dt**5/20,   dt**4/8,    dt**3/6,    0,          0,          0],
            [0,           0,          0,          dt**4/8,    dt**3/3,    dt**2/2,    0,          0,          0],
            [0,           0,          0,          dt**3/6,    dt**2/2,    dt,         0,          0,          0],
            [0,           0,          0,          0,          0,          0,          dt**5/20,   dt**4/8,    dt**3/6],
            [0,           0,          0,          0,          0,          0,          dt**4/8,    dt**3/3,    dt**2/2],
            [0,           0,          0,          0,          0,          0,          dt**3/6,    dt**2/2,    dt]
        ])
        covariance_matrix*= self.noise_diff_coeff
        # end = time.time()
        # print("Covariance computation time: ", (end - start)*1000,"ms")
        return covariance_matrix

class ConstantVelocityTransitionModel():
    def __init__(self, noise_diff_coeff, constant_dt = None):
        self.noise_diff_coeff = noise_diff_coeff
        self.transition_matrix = block_diag(*[np.zeros((3,3))])
        self.constant_dt = constant_dt

        if constant_dt is not None:
            self.transition_matrix = self.matrix(constant_dt)
            self.covariance_matrix = self.covar(constant_dt)


    
    def matrix(self, time_interval, **kwargs):
        # start = time.time()
        transition_matrix = np.array([
            [1, time_interval, 0,       0,       0,           0,      0,      0,          0],
            [0,    1,          0,       0,       0,           0,      0,      0,          0],
            [0,    0,          1,       0,       0,           0,      0,      0,          0],
            [0,    0,          0,       1,   time_interval,   0,      0,      0,          0],
            [0,    0,          0,       0,       1,           0,      0,      0,          0],
            [0,    0,          0,       0,       0,           1,      0,      0,          0],
            [0,    0,          0,       0,       0,           0,      1,  time_interval,  0],
            [0,    0,          0,       0,       0,           0,      0,      1,          0],
            [0,    0,          0,       0,       0,           0,      0,      0,          1]])
        # end = time.time()
        # print("Transition matrix computation time: ", (end - start)*1000,"ms")
        return transition_matrix

    def covar(self, time_interval, **kwargs):
        # start = time.time()
        dt = time_interval
        covariance_matrix = np.array([
            [dt**3/3,    dt**2/2,     0,          0,          0,          0,          0,          0,          0],
            [dt**2/2,     dt,         0,          0,          0,          0,          0,          0,          0],
            [0,           0,          0,          0,          0,          0,          0,          0,          0],
            [0,           0,          0,          dt**3/3,    dt**2/2,    0,          0,          0,          0],
            [0,           0,          0,          dt**2/2,    dt,         0,          0,          0,          0],
            [0,           0,          0,          0,          0,          0,          0,          0,          0],
            [0,           0,          0,          0,          0,          0,          dt**3/3,    dt**2/2,    0],
            [0,           0,          0,          0,          0,          0,          dt**2/2,    dt,         0],
            [0,           0,          0,          0,          0,          0,          0,          0,          0]
        ])
        covariance_matrix*= self.noise_diff_coeff
        # end = time.time()
        # print("Covariance computation time: ", (end - start)*1000,"ms")
        return covariance_matrix

class ConstantPositionTransitionModel():
    def __init__(self, noise_diff_coeff, constant_dt = None):
        self.noise_diff_coeff = noise_diff_coeff
        self.transition_matrix = block_diag(*[np.zeros((3,3))])
        self.constant_dt = constant_dt

        if constant_dt is not None:
            self.transition_matrix = self.matrix(constant_dt)
            self.covariance_matrix = self.covar(constant_dt)


    
    def matrix(self, time_interval, **kwargs):
        # start = time.time()
        transition_matrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        # end = time.time()
        # print("Transition matrix computation time: ", (end - start)*1000,"ms")
        return transition_matrix

    def covar(self, time_interval, **kwargs):
        # start = time.time()
        dt = time_interval
        covariance_matrix = np.array([[dt,0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, dt,0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, dt,0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        covariance_matrix*= self.noise_diff_coeff
        # end = time.time()
        # print("Covariance computation time: ", (end - start)*1000,"ms")
        return covariance_matrix
    
    
    
class ConstantVelocityXYConstantAccelerationZTransitionModel():
    def __init__(self, noise_diff_coeff, constant_dt = None):
        self.noise_diff_coeff = noise_diff_coeff
        # self.transition_matrix = block_diag(*[np.zeros((3,3))])
        self.constant_dt = constant_dt

        # if constant_dt is not None:
        #     self.transition_matrix = self.matrix(constant_dt)
        #     self.covariance_matrix = self.covar(constant_dt)


    
    def matrix(self, time_interval, **kwargs):
        # start = time.time()
        transition_matrix = np.array([
            [1, time_interval, 0,       0,       0,           0,      0,      0,          0],
            [0,    1,          0,       0,       0,           0,      0,      0,          0],
            [0,    0,          0,       0,       0,           0,      0,      0,          0],
            [0,    0,          0,       1,   time_interval,   0,      0,      0,          0],
            [0,    0,          0,       0,       1,           0,      0,      0,          0],
            [0,    0,          0,       0,       0,           0,      0,      0,          0],
            [0,    0,          0,       0,       0,           0,      1,  time_interval, time_interval**2/2],
            [0,    0,          0,       0,       0,           0,      0,      1,          time_interval],
            [0,    0,          0,       0,       0,           0,      0,      0,          1]])

        return transition_matrix

    def covar(self, time_interval, **kwargs):
        dt = time_interval
        # covariance_matrix = np.array([
        #     [dt**3/3,    dt**2/2,     0,          0,          0,          0,          0,          0,          0],
        #     [dt**2/2,     dt,         0,          0,          0,          0,          0,          0,          0],
        #     [0,           0,          0,          0,          0,          0,          0,          0,          0],
        #     [0,           0,          0,          dt**3/3,    dt**2/2,    0,          0,          0,          0],
        #     [0,           0,          0,          dt**2/2,    dt,         0,          0,          0,          0],
        #     [0,           0,          0,          0,          0,          0,          0,          0,          0],
        #     [0,           0,          0,          0,          0,          0,          dt**5/20,   dt**4/8,    dt**3/6],
        #     [0,           0,          0,          0,          0,          0,          dt**4/8,    dt**3/3,    dt**2/2],
        #     [0,           0,          0,          0,          0,          0,          dt**3/6,    dt**2/2,    dt]
        # ])
        
        covariance_matrix = np.array([
            [dt**3/3,    dt**2/2,     0,          0,          0,          0,          0,          0,          0],
            [dt**2/2,     dt,         0,          0,          0,          0,          0,          0,          0],
            [0,           0,          0,          0,          0,          0,          0,          0,          0],
            [0,           0,          0,          dt**3/3,    dt**2/2,    0,          0,          0,          0],
            [0,           0,          0,          dt**2/2,    dt,         0,          0,          0,          0],
            [0,           0,          0,          0,          0,          0,          0,          0,          0],
            [0,           0,          0,          0,          0,          0,          dt**3/3,    dt**2/2,    0],
            [0,           0,          0,          0,          0,          0,          dt**2/2,    dt,         0],
            [0,           0,          0,          0,          0,          0,          0,          0,          0]
        ])
        covariance_matrix*= self.noise_diff_coeff
        return covariance_matrix
    
    
    
class ConstantVelocityXYConstantAccelerationZ_covTransitionModel():
    def __init__(self, noise_diff_coeff, constant_dt = None):
        self.noise_diff_coeff = noise_diff_coeff
        # self.transition_matrix = block_diag(*[np.zeros((3,3))])
        self.constant_dt = constant_dt

        # if constant_dt is not None:
        #     self.transition_matrix = self.matrix(constant_dt)
        #     self.covariance_matrix = self.covar(constant_dt)


    
    def matrix(self, time_interval, **kwargs):
        # start = time.time()
        transition_matrix = np.array([
            [1, time_interval, 0,       0,       0,           0,      0,      0,          0],
            [0,    1,          0,       0,       0,           0,      0,      0,          0],
            [0,    0,          0,       0,       0,           0,      0,      0,          0],
            [0,    0,          0,       1,   time_interval,   0,      0,      0,          0],
            [0,    0,          0,       0,       1,           0,      0,      0,          0],
            [0,    0,          0,       0,       0,           0,      0,      0,          0],
            [0,    0,          0,       0,       0,           0,      1,  time_interval, time_interval**2/2],
            [0,    0,          0,       0,       0,           0,      0,      1,          time_interval],
            [0,    0,          0,       0,       0,           0,      0,      0,          1]])

        return transition_matrix

    def covar(self, time_interval, **kwargs):
        dt = time_interval
        covariance_matrix = np.array([
            [dt**3/3,    dt**2/2,     0,          0,          0,          0,          0,          0,          0],
            [dt**2/2,     dt,         0,          0,          0,          0,          0,          0,          0],
            [0,           0,          0,          0,          0,          0,          0,          0,          0],
            [0,           0,          0,          dt**3/3,    dt**2/2,    0,          0,          0,          0],
            [0,           0,          0,          dt**2/2,    dt,         0,          0,          0,          0],
            [0,           0,          0,          0,          0,          0,          0,          0,          0],
            [0,           0,          0,          0,          0,          0,          dt**5/20,   dt**4/8,    dt**3/6],
            [0,           0,          0,          0,          0,          0,          dt**4/8,    dt**3/3,    dt**2/2],
            [0,           0,          0,          0,          0,          0,          dt**3/6,    dt**2/2,    dt]
        ])
        
        # covariance_matrix = np.array([
        #     [dt**3/3,    dt**2/2,     0,          0,          0,          0,          0,          0,          0],
        #     [dt**2/2,     dt,         0,          0,          0,          0,          0,          0,          0],
        #     [0,           0,          0,          0,          0,          0,          0,          0,          0],
        #     [0,           0,          0,          dt**3/3,    dt**2/2,    0,          0,          0,          0],
        #     [0,           0,          0,          dt**2/2,    dt,         0,          0,          0,          0],
        #     [0,           0,          0,          0,          0,          0,          0,          0,          0],
        #     [0,           0,          0,          0,          0,          0,          dt**3/3,    dt**2/2,    0],
        #     [0,           0,          0,          0,          0,          0,          dt**2/2,    dt,         0],
        #     [0,           0,          0,          0,          0,          0,          0,          0,          0]
        # ])
        covariance_matrix*= self.noise_diff_coeff
        return covariance_matrix