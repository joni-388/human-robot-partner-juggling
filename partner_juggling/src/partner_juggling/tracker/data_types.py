from collections.abc import Sequence
from collections import deque
import numpy as np
from typing import Sequence

MAX_LENGTH_TRACK_STATES = 200
class Matrix(np.ndarray):
    """Matrix wrapper for :class:`numpy.ndarray`

    This class returns a view to a :class:`numpy.ndarray` It's called same as
    to :func:`numpy.asarray`.
    """

    def __new__(cls, *args, **kwargs):
        array = np.asarray(*args, **kwargs)
        return array.view(cls)

    def __array_wrap__(self, array):
        return self._cast(array)

    @classmethod
    def _cast(cls, val):
        # This tries to cast the result as either a StateVector or Matrix type if applicable.
        if isinstance(val, np.ndarray):
            if val.ndim == 2 and val.shape[1] == 1:
                return val.view(StateVector)
            else:
                return val.view(Matrix)
        else:
            return val

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc in (np.isfinite, np.matmul):
            # Custom types break here, so simply convert to floats.
            inputs = [np.asfarray(input_) if isinstance(input_, Matrix) else input_
                      for input_ in inputs]
        else:
            # Change to standard ndarray
            inputs = [np.asarray(input_) if isinstance(input_, Matrix) else input_
                      for input_ in inputs]
        if 'out' in kwargs:
            kwargs['out'] = tuple(np.asarray(out) if isinstance(out, Matrix) else out
                                  for out in kwargs['out'])

        result = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        if result is NotImplemented:
            return NotImplemented
        else:
            return self._cast(result)


class StateVector(Matrix):
    r"""State vector wrapper for :class:`numpy.ndarray`

    This class returns a view to a :class:`numpy.ndarray`, but ensures that
    its initialised as an :math:`N \times 1` vector. It's called same as
    :func:`numpy.asarray`. The StateVector will attempt to convert the data
    given to a :math:`N \times 1` vector if it can easily be done. E.g.,
    ``StateVector([1., 2., 3.])``, ``StateVector ([[1., 2., 3.,]])``, and
    ``StateVector([[1.], [2.], [3.]])`` will all return the same 3x1 StateVector.

    It also overrides the behaviour of indexing such that my_state_vector[1] returns the second
    element (as `int`, `float` etc), rather than a StateVector of size (1, 1) as would be the case
    without this override. Behaviour of indexing with lists, slices or other indexing is
    unaffected (as you would expect those to return StateVectors). This override avoids the need
    for client to specifically index with zero as the second element (`my_state_vector[1, 0]`) to
    get a native numeric type. Iterating through the StateVector returns a sequence of numbers,
    rather than a sequence of 1x1 StateVectors. This makes the class behave as would be expected
    and avoids 'gotchas'.

    Note that code using the pattern `my_state_vector[1, 0]` will continue to work.

    When slicing would result in return of a invalid shape for a StateVector (i.e. not `(n, 1)`)
    then a :class:`~.Matrix` view will be returned.

    .. note ::
        It is not recommended to use a StateVector for indexing another vector. Doing so will lead
        to unexpected effects. Use a :class:`tuple`, :class:`list` or :class:`np.ndarray` for this.
    """

    def __new__(cls, *args, **kwargs):
        array = np.asarray(*args, **kwargs)
        # For convenience handle shapes that can be easily converted in a
        # Nx1 shape
        if array.ndim == 1:
            array = array.reshape((array.shape[0], 1))
        elif array.ndim == 2 and array.shape[0] == 1:
            array = array.T

        if not (array.ndim == 2 and array.shape[1] == 1):
            raise ValueError(
                "state vector shape should be Nx1 dimensions: got {}".format(
                    array.shape))
        return array.view(cls)

    def __getitem__(self, item):
        # If item has two elements, it is a tuple and should be left alone.
        # If item is a slice object, or an ndarray, we would expect a StateVector returned,
        #   so leave it alone.
        # If item is an int, we would expected a number returned, so we should append 0  to the
        #   item and extract the first (and only) column
        # Note that an ndarray of ints is an instance of int
        #   i.e. isinstance(np.array([1]), int) == True
        if isinstance(item, int):
            item = (item, 0)
        # Cast here, so StateVector isn't returned with invalid shape (e.g. (n, ))
        return self._cast(super().__getitem__(item))

    def __setitem__(self, key, value):
        if isinstance(key, int):
            key = (key, 0)
        return super().__setitem__(key, value)

    def flatten(self, *args, **kwargs):
        return self._cast(super().flatten(*args, **kwargs))

    def ravel(self, *args, **kwargs):
        return self._cast(super().ravel(*args, **kwargs))


class StateVectors(Matrix):
    """Wrapper for :class:`numpy.ndarray for multiple State Vectors`

    This class returns a view to a :class:`numpy.ndarray` that is in shape
    (num_dimensions, num_components), customising some numpy functions to ensure
    custom types are handled correctly. This can be initialised by a sequence
    type (list, tuple; not array) that contains :class:`StateVector`, otherwise
    it's called same as :func:`numpy.asarray`.
    """

    def __new__(cls, states, *args, **kwargs):
        if isinstance(states, Sequence) and not isinstance(states, np.ndarray):
            if isinstance(states[0], StateVector):
                return np.hstack(states).view(cls)
        array = np.asarray(states, *args, **kwargs)
        if array.shape[1] == 1:
            return array.view(StateVector)
        return array.view(cls)

    def __iter__(self):
        statev_gen = super(StateVectors, self.T).__iter__()
        for statevector in statev_gen:
            yield StateVector(statevector)

    def __getitem__(self, item):
        return self._cast(super().__getitem__(item))

    @classmethod
    def _cast(cls, val):
        out = super()._cast(val)
        if type(out) == Matrix and out.ndim == 2:
            # Assume still set of State Vectors
            return out.view(StateVectors)
        else:
            return out

    def __array_function__(self, func, types, args, kwargs):
        if func is np.average:
            return self._average(*args, **kwargs)
        elif func is np.mean:
            return self._mean(*args, **kwargs)
        elif func is np.cov:
            return self._cov(*args, **kwargs)
        else:
            return super().__array_function__(func, types, args, kwargs)

    @staticmethod
    def _mean(state_vectors, axis=None, dtype=None, out=None, keepdims=np._NoValue):
        if state_vectors.dtype != np.object_:
            # Can just use standard numpy mean if not using custom objects
            return np.mean(np.asarray(state_vectors), axis, dtype, out, keepdims)
        elif axis == 1 and out is None:
            state_vector = np.average(state_vectors, axis)
            if dtype:
                return state_vector.astype(dtype)
            else:
                return state_vector
        else:
            return NotImplemented

    @staticmethod
    def _average(state_vectors, axis=None, weights=None, returned=False):
        if state_vectors.dtype != np.object_:
            # Can just use standard numpy averaging if not using custom objects
            state_vector = np.average(np.asarray(state_vectors), axis=axis, weights=weights)
            # Convert type as may have type of weights
            state_vector = StateVector(state_vector.astype(np.float_, copy=False))
        elif axis == 1:  # Need to handle special cases of averaging potentially
            state_vector = StateVector(
                np.empty((state_vectors.shape[0], 1), dtype=state_vectors.dtype))
            for dim, row in enumerate(np.asarray(state_vectors)):
                type_ = type(row[0])  # Assume all the same type
                if hasattr(type_, 'average'):
                    # Check if type has custom average method
                    state_vector[dim, 0] = type_.average(row, weights=weights)
                else:
                    # Else use numpy built in, converting to float array
                    state_vector[dim, 0] = type_(np.average(np.asfarray(row), weights=weights))
        else:
            return NotImplemented

        if returned:
            return state_vector, np.sum(weights)
        else:
            return state_vector

    @staticmethod
    def _cov(state_vectors, y=None, rowvar=True, bias=False, ddof=None, fweights=None,
             aweights=None):

        if state_vectors.dtype != np.object_:
            # Can just use standard numpy averaging if not using custom objects
            cov = np.cov(np.asarray(state_vectors), y, rowvar, bias, ddof, fweights, aweights)
        elif y is None and rowvar and not bias and ddof == 0 and fweights is None:
            # Only really handle simple usage here
            avg, w_sum = np.average(state_vectors, axis=1, weights=aweights, returned=True)

            X = np.asfarray(state_vectors - avg)
            if aweights is None:
                X_T = X.T
            else:
                X_T = (X*np.asfarray(aweights)).T
            cov = X @ X_T.conj()
            cov *= np.true_divide(1, float(w_sum))
        else:
            return NotImplemented
        return CovarianceMatrix(np.atleast_2d(cov))


class CovarianceMatrix(Matrix):
    """Covariance matrix wrapper for :class:`numpy.ndarray`.

    This class returns a view to a :class:`numpy.ndarray`, but ensures that
    its initialised at a *NxN* matrix. It's called similar to
    :func:`numpy.asarray`.
    """

    def __new__(cls, *args, **kwargs):
        array = np.asarray(*args, **kwargs)
        if not array.ndim == 2:
            raise ValueError("Covariance should have ndim of 2: got {}"
                             "".format(array.ndim))
        return array.view(cls)

class State():
    """State type.

    Most simple state type, which only has time and a state vector."""
    def __init__(self, state_vector, timestamp):
        # Don't cast away subtype of state_vector if not necessary
        if state_vector is not None and not isinstance(state_vector, (StateVector, StateVectors)):
            state_vector = StateVector(state_vector)
        self.timestamp = timestamp
        self.state_vector = state_vector

    @property
    def ndim(self):
        """The number of dimensions represented by the state."""
        return self.state_vector.shape[0]

class Track():
    """Track type
    """
    def __init__(self, states, id, marker_id, transition_model=None):
        self.states = deque(maxlen=MAX_LENGTH_TRACK_STATES)
        self.id = id
        self.marker_id = marker_id
        self.transition_model = transition_model
        self.detections = deque(maxlen=MAX_LENGTH_TRACK_STATES)
        for state in states:
            self.states.append(state)
        
class GaussianState(State):
    """Gaussian State type

    This is a simple Gaussian state object, which, as the name suggests,
    is described by a Gaussian state distribution.
    """
    def __init__(self, state_vector, covar, timestamp):
        # Don't cast away subtype of covar if not necessary
        if not isinstance(covar, CovarianceMatrix):
            covar = CovarianceMatrix(covar)
        super().__init__(state_vector, timestamp)
        self.covar = covar
        if self.state_vector.shape[0] != self.covar.shape[0]:
            raise ValueError(
                "state vector and covariance should have same dimensions")

    @property
    def mean(self):
        """The state mean, equivalent to state vector"""
        return self.state_vector

class Detection(State):
    """Detection type"""
    def __init__(self, state_vector, timestamp, measurement_model, marker_id):
        super().__init__(state_vector, timestamp)
        self.measurement_model = measurement_model
        self.marker_id = marker_id

class MissedDetection(Detection):
    """Detection type for a missed detection

    This is same as :class:`~.Detection`, but it is used in
    MultipleHypothesis to indicate the null hypothesis (no
    detections are associated with the specified track).
    state_vector: StateVector = Property(default=None, doc="State vector. Default `None`.")
    """
    def __init__(self, state_vector=None, timestamp=None):
        super().__init__(state_vector, timestamp, measurement_model=None, marker_id=None)

    def __bool__(self):
        return False

class GaussianStatePrediction(GaussianState):
    """ GaussianStatePrediction type

    This is a simple Gaussian state prediction object, which, as the name
    suggests, is described by a Gaussian distribution.
    """
    def __init__(self, prior, x_pred, p_pred, timestamp, transition_model):
        super().__init__(x_pred, p_pred, timestamp)
        self.prior = prior
        self.transition_model = transition_model

class GaussianMeasurementPrediction(GaussianState):
    """ GaussianMeasurementPrediction type

    This is a simple Gaussian measurement prediction object, which, as the name
    suggests, is described by a Gaussian distribution.
    """
    def __init__(self, prediction, pred_meas, innov_cov, cross_covar):
        super().__init__(pred_meas, innov_cov, prediction.timestamp)
        self.cross_covar = cross_covar
        if self.cross_covar is not None \
                and self.cross_covar.shape[1] != self.state_vector.shape[0]:
            raise ValueError("cross_covar should have the same number of "
                             "columns as the number of rows in state_vector")

class GaussianStateUpdate(GaussianState):
    """ GaussianStateUpdate type

    This is a simple Gaussian state update object, which, as the name
    suggests, is described by a Gaussian distribution.
    """
    def __init__(self, prediction, post_mean, post_covar, timestamp, hypothesis):
        super().__init__(post_mean, post_covar, timestamp)
        self.hypothesis = hypothesis

class SingleHypothesis():
    """A hypothesis based on a single measurement.
    prediction: Prediction = Property(doc="Predicted track state")
    measurement: Detection = Property(doc="Detection used for hypothesis and updating")
    measurement_prediction: MeasurementPrediction = Property(
        default=None, doc="Optional track prediction in measurement space")
    """
    def __init__(self, prediction, measurement, measurement_prediction):
        self.prediction = prediction
        self.measurement = measurement
        self.measurement_prediction = measurement_prediction
    
    def __bool__(self):
        return (not isinstance(self.measurement, MissedDetection)) and \
               (self.measurement is not None)

class MultipleHypothesis():
    """Multiple Hypothesis base type

    A Multiple Hypothesis is a container to store a collection of hypotheses.
    single_hypotheses: Sequence[SingleHypothesis] = Property(
        default=None,
        doc="The initial list of :class:`~.SingleHypothesis`. Default `None` "
            "which initialises with empty list.")
    normalise: bool = Property(
        default=False,
        doc="Normalise probabilities of :class:`~.SingleHypothesis`. Default "
            "is `False`.")
    total_weight: float = Property(
        default=1,
        doc="When normalising, weights will sum to this. Default is 1.")
    """
    def __init__(self, single_hypotheses=None, normalise=False, total_weight=1):
        if single_hypotheses is None:
            single_hypotheses = []

        if any(not isinstance(hypothesis, SingleHypothesis)
               for hypothesis in single_hypotheses):
            raise ValueError("Cannot form MultipleHypothesis out of "
                             "non-SingleHypothesis inputs!")
        self.single_hypotheses = single_hypotheses
        self.normalise = normalise
        self.total_weight = total_weight

        # normalise the weights of 'single_hypotheses', if indicated
        if self.normalise:
            self.normalise_probabilities()

    def __len__(self):
        return self.single_hypotheses.__len__()

    def __contains__(self, index):
        # check if 'single_hypotheses' contains any SingleHypotheses with
        # Detection matching 'index'
        if isinstance(index, Detection):
            for hypothesis in self.single_hypotheses:
                if hypothesis.measurement is index:
                    return True
            return False

        # check if 'single_hypotheses' contains any SingleHypotheses with
        # Prediction matching 'index'
        if isinstance(index, GaussianStatePrediction):
            for hypothesis in self.single_hypotheses:
                if hypothesis.prediction is index:
                    return True
            return False

        # check if 'single_hypotheses' contains any SingleHypotheses
        # matching 'index'
        if isinstance(index, SingleHypothesis):
            return index in self.single_hypotheses

    def __iter__(self):
        for hypothesis in self.single_hypotheses:
            yield hypothesis

    def __getitem__(self, index):

        # retrieve SingleHypothesis by array index
        if isinstance(index, int):
            return self.single_hypotheses[index]

        # retrieve SingleHypothesis by measurement
        if isinstance(index, Detection):
            for hypothesis in self.single_hypotheses:
                if hypothesis.measurement is index:
                    return hypothesis
            return None

        # retrieve SingleHypothesis by prediction
        if isinstance(index, GaussianStatePrediction):
            for hypothesis in self.single_hypotheses:
                if hypothesis.prediction is index:
                    return hypothesis
            return None

    def normalise_probabilities(self, total_weight=None):
        if total_weight is None:
            total_weight = self.total_weight

        # verify that SingleHypotheses composing this MultipleHypothesis
        # all have Probabilities
        if any(not hasattr(hypothesis, 'probability')
               for hypothesis in self.single_hypotheses):
            raise ValueError("MultipleHypothesis not composed of Probability"
                             " hypotheses!")

        sum_weights = sum(
            hypothesis.probability for hypothesis in self.single_hypotheses)

        for hypothesis in self.single_hypotheses:
            hypothesis.probability =\
                (hypothesis.probability * total_weight)/sum_weights

    def get_missed_detection_probability(self):
        for hypothesis in self.single_hypotheses:
            if isinstance(hypothesis.measurement, MissedDetection):
                if hasattr(hypothesis, 'probability'):
                    return hypothesis.probability
        return None

class DistanceJointHypothesis():
    """Distance scored Joint Hypothesis subclass.

    Notes
    -----
    As smaller distance is 'better', comparison logic is reversed
    i.e. smaller distance is a greater likelihood.
    """

    def __init__(self, hypotheses):
        self.hypotheses = hypotheses

    @property
    def distance(self):
        return sum(hypothesis.distance for hypothesis in self.hypotheses.values())

    def __lt__(self, other):
        return self.distance > other.distance

    def __le__(self, other):
        return self.distance >= other.distance

    def __eq__(self, other):
        return self.distance == other.distance

    def __gt__(self, other):
        return self.distance < other.distance

    def __ge__(self, other):
        return self.distance <= other.distance
    
class SingleDistanceHypothesis(SingleHypothesis):
    """Distance scored hypothesis subclass.

    Notes
    -----
    As smaller distance is 'better', comparison logic is reversed
    i.e. smaller distance is a greater likelihood.
    distance: float = Property(doc="Distance between detection and prediction")
    """
    def __init__(self, prediction, measurement, distance, measurement_prediction=None):
        super().__init__(prediction, measurement, measurement_prediction)
        self.distance = distance

    def __lt__(self, other):
        return self.distance > other.distance

    def __le__(self, other):
        return self.distance >= other.distance

    def __eq__(self, other):
        return self.distance == other.distance

    def __gt__(self, other):
        return self.distance < other.distance

    def __ge__(self, other):
        return self.distance <= other.distance

    @property
    def weight(self):
        try:
            return 1 / self.distance
        except ZeroDivisionError:
            return float('inf')

class DistanceHypothesiser():
    """Prediction Hypothesiser based on a Measure

    Generate track predictions at detection times and score each hypothesised
    prediction-detection pair using the distance of the supplied
    :class:`~.Measure` class.
    predictor: Predictor = Property(doc="Predict tracks to detection times")
    updater: Updater = Property(doc="Updater used to get measurement prediction")
    measure: Measure = Property(
        doc="Measure class used to calculate the distance between two states.")
    missed_distance: float = Property(
        default=float('inf'),
        doc="Distance for a missed detection. Default is set to infinity")
    include_all: bool = Property(
        default=False,
        doc="If `True`, hypotheses beyond missed distance will be returned. Default `False`")
    """
    def __init__(self, predictor, updater, measure, missed_distance=float('inf'), include_all=False):
        self.predictor = predictor
        self.updater = updater
        self.measure = measure
        self.missed_distance = missed_distance
        self.include_all = include_all

    def hypothesise(self, track, detections, timestamp):
        """ Evaluate and return all track association hypotheses.

        For a given track and a set of N available detections, return a
        MultipleHypothesis object with N+1 detections (first detection is
        a 'MissedDetection'), each with an associated distance measure..

        Parameters
        ----------
        track : Track
            The track object to hypothesise on
        detections : set of :class:`~.Detection`
            The available detections
        timestamp : datetime.datetime
            A timestamp used when evaluating the state and measurement
            predictions. Note that if a given detection has a non empty
            timestamp, then prediction will be performed according to
            the timestamp of the detection.

        Returns
        -------
        : :class:`~.MultipleHypothesis`
            A container of :class:`~SingleDistanceHypothesis` objects

        """
        hypotheses = list()

        # Common state & measurement prediction
        prediction = self.predictor.predict(track, timestamp=timestamp)
        # Missed detection hypothesis with distance as 'missed_distance'
        hypotheses.append(
            SingleDistanceHypothesis(
                prediction,
                MissedDetection(timestamp=timestamp),
                self.missed_distance
                ))

        # True detection hypotheses
        for detection in detections:

            # Re-evaluate prediction
            prediction = self.predictor.predict(track, timestamp=detection.timestamp)

            # Compute measurement prediction and distance measure
            measurement_prediction = self.updater.predict_measurement(prediction, detection.measurement_model)
            distance = self.measure(measurement_prediction, detection)

            if self.include_all or distance < self.missed_distance:
                # True detection hypothesis
                hypotheses.append(
                    SingleDistanceHypothesis(
                        prediction,
                        detection,
                        distance,
                        measurement_prediction))

        return MultipleHypothesis(sorted(hypotheses, reverse=True))
    