from partner_juggling.tracker.data_types import StateVector, CovarianceMatrix, GaussianStatePrediction, GaussianMeasurementPrediction, GaussianStateUpdate 
import numpy as np


class KalmanPredictor():
    r"""A predictor class which forms the basis for the family of Kalman
    predictors. This class also serves as the (specific) Kalman Filter
    :class:`~.Predictor` class. Here

    .. math::

      f_k( \mathbf{x}_{k-1}) = F_k \mathbf{x}_{k-1},  \ b_k( \mathbf{u}_k) =
      B_k \mathbf{u}_k \ \mathrm{and} \ \mathbf{\nu}_k \sim \mathcal{N}(0,Q_k)

    Notes
    -----
    In the Kalman filter, transition model must be linear.

    Raises
    ------
    ValueError
        If no :class:`~.TransitionModel` is specified.

    """
    def __init__(self, transition_models):
        self.transition_models = transition_models
        self.transition_model = None
        self.last_prior = None
        self.last_transition_model = None

    def _transition_function(self, prior, transition_model, **kwargs):
        r"""Applies the linear transition function to a single vector in the
        absence of a control input, returns a single predicted state.

        Parameters
        ----------
        prior : :class:`~.GaussianState`
            The prior state, :math:`\mathbf{x}_{k-1}`

        **kwargs : various, optional
            These are passed to :meth:`~.LinearGaussianTransitionModel.matrix`

        Returns
        -------
        : :class:`~.State`
            The predicted state

        """

        self.transition_model = transition_model
        

            
        if self.transition_model is self.transition_models["ConstantVelocityXYConstantAccelerationZ"]:
            prior.state_vector[8] = -9.81 # does not change as noise is zero
            pass
            

        
        transition_matrix = self.transition_model.matrix(**kwargs)
        
        pred_state = transition_matrix @ prior.state_vector 
        
        return  pred_state
        


    def _predict_over_interval(self, prior, timestamp):
        """Private function to get the prediction interval (or None)

        Parameters
        ----------
        prior : :class:`~.State`
            The prior state

        timestamp : :class:`datetime.datetime`, optional
            The (current) timestamp

        Returns
        -------
        : :class:`datetime.timedelta`
            time interval to predict over

        """

        # Deal with undefined timestamps
        if timestamp is None or prior.timestamp is None:
            predict_over_interval = None
        else:
            predict_over_interval = timestamp - prior.timestamp

        return predict_over_interval

    def _predicted_covariance(self, prior, predict_over_interval, **kwargs):
        """Private function to return the predicted covariance. Useful in that
        it can be overwritten in children.

        Parameters
        ----------
        prior : :class:`~.GaussianState`
            The prior class
        predict_over_interval : :class`~.timedelta`

        Returns
        -------
        : :class:`~.CovarianceMatrix`
            The predicted covariance matrix

        """
        prior_cov = prior.covar
        if self.transition_model.constant_dt is None:
            trans_m = self.transition_model.matrix(time_interval=predict_over_interval.total_seconds(), **kwargs)
            trans_cov = self.transition_model.covar(time_interval=predict_over_interval.total_seconds(), **kwargs)
            
        else:
            trans_m = self.transition_model.transition_matrix
            trans_cov = self.transition_model.covariance_matrix
        
        return trans_m @ prior_cov @ trans_m.T + trans_cov

    def predict(self, prior, transition_model, timestamp=None):
        r"""The predict function

        Parameters
        ----------
        prior : :class:`~.State`
            :math:`\mathbf{x}_{k-1}`
        timestamp : :class:`datetime.datetime`, optional
            :math:`k`
        **kwargs :
            These are passed, via :meth:`~.KalmanFilter.transition_function` to
            :meth:`~.LinearGaussianTransitionModel.matrix`

        Returns
        -------
        : :class:`~.GaussianStatePrediction`
            :math:`\mathbf{x}_{k|k-1}`, the predicted state and the predicted
            state covariance :math:`P_{k|k-1}`

        """

        # Get the prediction interval
        predict_over_interval = self._predict_over_interval(prior, timestamp)

        # Prediction of the mean
        x_pred = self._transition_function(
            prior, transition_model, time_interval=predict_over_interval.total_seconds())

        # Prediction of the covariance
        p_pred = self._predicted_covariance(prior, predict_over_interval)

        # And return the state in the correct form
        return GaussianStatePrediction(prior, x_pred, p_pred, timestamp=timestamp,
                                     transition_model=self.transition_model)

class KalmanUpdater():
    r"""A class which embodies Kalman-type updaters; also a class which
    performs measurement update step as in the standard Kalman filter.

    The Kalman updaters assume :math:`h(\mathbf{x}) = H \mathbf{x}` with
    additive noise :math:`\sigma = \mathcal{N}(0,R)`. Daughter classes can
    overwrite to specify a more general measurement model
    :math:`h(\mathbf{x})`.

    :meth:`update` first calls :meth:`predict_measurement` function which
    proceeds by calculating the predicted measurement, innovation covariance
    and measurement cross-covariance,

    .. math::

        \mathbf{z}_{k|k-1} &= H_k \mathbf{x}_{k|k-1}

        S_k &= H_k P_{k|k-1} H_k^T + R_k

        \Upsilon_k &= P_{k|k-1} H_k^T

    where :math:`P_{k|k-1}` is the predicted state covariance.
    :meth:`predict_measurement` returns a
    :class:`~.GaussianMeasurementPrediction`. The Kalman gain is then
    calculated as,

    .. math::

        K_k = \Upsilon_k S_k^{-1}

    and the posterior state mean and covariance are,

    .. math::

        \mathbf{x}_{k|k} &= \mathbf{x}_{k|k-1} + K_k (\mathbf{z}_k - H_k
        \mathbf{x}_{k|k-1})

        P_{k|k} &= P_{k|k-1} - K_k S_k K_k^T

    These are returned as a :class:`~.GaussianStateUpdate` object.
    """
    def __init__(self, measurement_model, force_symmetric_covariance=False):
        self.measurement_model = measurement_model
        self.force_symmetric_covariance = force_symmetric_covariance

    def _measurement_matrix(self, predicted_state=None, measurement_model=None,
                            **kwargs):
        r"""This is straightforward Kalman so just get the Matrix from the
        measurement model.

        Parameters
        ----------
        predicted_state : :class:`~.GaussianState`
            The predicted state :math:`\mathbf{x}_{k|k-1}`, :math:`P_{k|k-1}`
        measurement_model : :class:`~.MeasurementModel`
            The measurement model. If omitted, the model in the updater object
            is used
        **kwargs : various
            Passed to :meth:`~.MeasurementModel.matrix`

        Returns
        -------
        : :class:`numpy.ndarray`
            The measurement matrix, :math:`H_k`

        """
        if measurement_model is None:
            measurement_model = self.measurement_model
        return measurement_model.matrix(**kwargs)

    def _measurement_cross_covariance(self, predicted_state, measurement_matrix):
        """
        Return the measurement cross covariance matrix, :math:`P_{k~k-1} H_k^T`

        Parameters
        ----------
        predicted_state : :class:`GaussianState`
            The predicted state which contains the covariance matrix :math:`P` as :attr:`.covar`
            attribute
        measurement_matrix : numpy.array
            The measurement matrix, :math:`H`

        Returns
        -------
        :  numpy.ndarray
            The measurement cross-covariance matrix

        """
        return predicted_state.covar @ measurement_matrix.T

    def _innovation_covariance(self, m_cross_cov, meas_mat, meas_mod):
        """Compute the innovation covariance

        Parameters
        ----------
        m_cross_cov : numpy.ndarray
            The measurement cross covariance matrix
        meas_mat : numpy.ndarray
            Measurement matrix
        meas_mod : :class:~.MeasurementModel`
            Measurement model

        Returns
        -------
        : numpy.ndarray
            The innovation covariance

        """
        return meas_mat @ m_cross_cov + meas_mod.covar()

    def _posterior_mean(self, predicted_state, kalman_gain, measurement, measurement_prediction):
        r"""Compute the posterior mean, :math:`\mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + K_k
        \mathbf{y}_k`, where the innovation :math:`\mathbf{y}_k = \mathbf{z}_k -
        h(\mathbf{x}_{k|k-1}).

        Parameters
        ----------
        predicted_state : :class:`State`, :class:`Prediction`
            The predicted state
        kalman_gain : numpy.ndarray
            Kalman gain
        measurement : :class:`Detection`
            The measurement
        measurement_prediction : :class:`MeasurementPrediction`
            Predicted measurement

        Returns
        -------
        : :class:`StateVector`
            The posterior mean estimate
        """
        post_mean = predicted_state.state_vector + \
            kalman_gain @ (measurement.state_vector - measurement_prediction.state_vector)
        return post_mean.view(StateVector)

    def _posterior_covariance(self, hypothesis):
        """
        Return the posterior covariance for a given hypothesis

        Parameters
        ----------
        hypothesis: :class:`~.Hypothesis`
            A hypothesised association between state prediction and measurement. It returns the
            measurement prediction which in turn contains the measurement cross covariance,
            :math:`P_{k|k-1} H_k^T and the innovation covariance,
            :math:`S = H_k P_{k|k-1} H_k^T + R`

        Returns
        -------
        : :class:`~.CovarianceMatrix`
            The posterior covariance matrix rendered via the Kalman update process.
        : numpy.ndarray
            The Kalman gain, :math:`K = P_{k|k-1} H_k^T S^{-1}`

        """
        kalman_gain = hypothesis.measurement_prediction.cross_covar @ \
            np.linalg.inv(hypothesis.measurement_prediction.covar)

        post_cov = hypothesis.prediction.covar - kalman_gain @ \
            hypothesis.measurement_prediction.covar @ kalman_gain.T

        return post_cov.view(CovarianceMatrix), kalman_gain

    def predict_measurement(self, predicted_state, measurement_model=None,
                            **kwargs):
        r"""Predict the measurement implied by the predicted state mean

        Parameters
        ----------
        predicted_state : :class:`~.GaussianState`
            The predicted state :math:`\mathbf{x}_{k|k-1}`, :math:`P_{k|k-1}`
        measurement_model : :class:`~.MeasurementModel`
            The measurement model. If omitted, the model in the updater object
            is used
        **kwargs : various
            These are passed to :meth:`~.MeasurementModel.function` and
            :meth:`~.MeasurementModel.matrix`

        Returns
        -------
        : :class:`GaussianMeasurementPrediction`
            The measurement prediction, :math:`\mathbf{z}_{k|k-1}`

        """
        # If a measurement model is not specified then use the one that's
        # native to the updater
        if measurement_model is None:
            measurement_model = self.measurement_model

        pred_meas = measurement_model.function(predicted_state, **kwargs)

        hh = self._measurement_matrix(predicted_state=predicted_state,
                                      measurement_model=measurement_model,
                                      **kwargs)

        # The measurement cross covariance and innovation covariance
        meas_cross_cov = self._measurement_cross_covariance(predicted_state, hh)
        innov_cov = self._innovation_covariance(meas_cross_cov, hh, measurement_model)

        return GaussianMeasurementPrediction(predicted_state, pred_meas, innov_cov, cross_covar=meas_cross_cov)

    def update(self, hypothesis, **kwargs):
        r"""The Kalman update method. Given a hypothesised association between
        a predicted state or predicted measurement and an actual measurement,
        calculate the posterior state.

        Parameters
        ----------
        hypothesis : :class:`~.SingleHypothesis`
            the prediction-measurement association hypothesis. This hypothesis
            may carry a predicted measurement, or a predicted state. In the
            latter case a predicted measurement will be calculated.
        **kwargs : various
            These are passed to :meth:`predict_measurement`

        Returns
        -------
        : :class:`~.GaussianStateUpdate`
            The posterior state Gaussian with mean :math:`\mathbf{x}_{k|k}` and
            covariance :math:`P_{x|x}`

        """
        # Get the predicted state out of the hypothesis
        predicted_state = hypothesis.prediction

        # If there is no measurement prediction in the hypothesis then do the
        # measurement prediction (and attach it back to the hypothesis).
        if hypothesis.measurement_prediction is None:
            # Get the measurement model out of the measurement if it's there.
            # If not, use the one native to the updater (which might still be
            # none)
            measurement_model = hypothesis.measurement.measurement_model
            measurement_model = self._check_measurement_model(
                measurement_model)

            # Attach the measurement prediction to the hypothesis
            hypothesis.measurement_prediction = self.predict_measurement(
                predicted_state, measurement_model=measurement_model, **kwargs)

        # Kalman gain and posterior covariance
        posterior_covariance, kalman_gain = self._posterior_covariance(hypothesis)

        # Posterior mean
        posterior_mean = self._posterior_mean(predicted_state, kalman_gain,
                                              hypothesis.measurement,
                                              hypothesis.measurement_prediction)

        if self.force_symmetric_covariance:
            posterior_covariance = \
                (posterior_covariance + posterior_covariance.T)/2

        return GaussianStateUpdate(
            hypothesis.prediction,
            posterior_mean, posterior_covariance,
            timestamp=hypothesis.measurement.timestamp, hypothesis=hypothesis)
