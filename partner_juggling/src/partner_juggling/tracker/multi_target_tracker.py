import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import time
import itertools
from functools import lru_cache
import copy
from datetime import datetime, timedelta
from scipy.spatial import distance

from partner_juggling.tracker.data_types import GaussianState, Track, DistanceHypothesiser, DistanceJointHypothesis, MissedDetection, SingleDistanceHypothesis, MultipleHypothesis
from partner_juggling.tracker.models import CombinedLinearGaussianTransitionModel, ConstantAcceleration, ConstantAccelerationTransitionModel, ConstantVelocityTransitionModel,ConstantVelocityXYConstantAccelerationZ_covTransitionModel, ConstantVelocityXYConstantAccelerationZTransitionModel,ConstantPositionTransitionModel, ConstantNthDerivative, LinearGaussianMeasurementModel
from partner_juggling.tracker.kalman import KalmanPredictor, KalmanUpdater

class MultiTargetTracker:
    """ Data association - Multi-target tracking, using global nearest neighbour 
    (Reference: https://stonesoup.readthedocs.io/en/latest/auto_tutorials/06_DataAssociation-MultiTargetTutorial.html)
    
    n_targets : Number of objects to be tracked
    n_dims : Number of state dimensions per object
    """
    def __init__(self, start_time,
                 state_init,
                 multi_target_tracker_params,
                 n_targets = 2,
                 n_dims = 2,
                 loop_rate = None,
                 use_only_kalman = False,
                 use_no_acc_prediction = True,
                 approximate_velocity_measurement = True):
        
        self.approximate_velocity_measurement = approximate_velocity_measurement
        self.use_only_kalman = use_only_kalman
        self.use_no_acc_prediction = use_no_acc_prediction
        
        self.start_time = start_time
        self.constant_derivative = multi_target_tracker_params["transition_model_constant_derivative"]
        self.transition_models = self.get_transition_models(n_targets, n_dims, loop_rate, multi_target_tracker_params["transition_model_noise_diff_coeff_const_pos"], multi_target_tracker_params["transition_model_noise_diff_coeff_const_acc"])
        self.measurement_model = self.get_measurement_model(n_targets, n_dims, multi_target_tracker_params["measurement_noise_var"])
        self.tracks = self.get_priors(state_init, n_targets, n_dims, start_time, multi_target_tracker_params["init_state_noise_var"])

        self.predictor = KalmanPredictor(self.transition_models)
        self.updater = KalmanUpdater(self.measurement_model)
        self.missed_distance = multi_target_tracker_params["missed_distance"]
        self.measure = multi_target_tracker_params["distance_measure"]
        self.hypothesiser = DistanceHypothesiser(self.predictor, self.updater, measure=self.measure, missed_distance=self.missed_distance)
        

    def get_track(self, track_id):
        """Get a track by its ID."""
        for track in self.tracks:
            if track.id == track_id:
                return track

        raise ValueError(f"Track with ID {track_id} not found.")
        return

    def get_transition_models(self, n_targets, n_dims, loop_rate, noise_diff_coeff_const_pos, noise_diff_coeff_const_acc):
        transition_models = {}
        # for n in range(n_dims):
        #     transition_models.append(ConstantNthDerivative(constant_derivative=self.constant_derivative, noise_diff_coeff=noise_diff_coeff, constant_dt = None))
        # transition_model = CombinedLinearGaussianTransitionModel(transition_models, constant_dt= None)
        transition_models["ConstantAcceleration"] = ConstantAccelerationTransitionModel(noise_diff_coeff_const_acc, n_dims)
        transition_models["ConstantPosition"] = ConstantPositionTransitionModel(noise_diff_coeff_const_pos)
        transition_models["ConstantVelocityXYConstantAccelerationZ_covTransitionModel"] = ConstantVelocityXYConstantAccelerationZ_covTransitionModel(noise_diff_coeff_const_acc)
        if self.use_no_acc_prediction: 
            # transition_models["ConstantVelocity"] = ConstantVelocityTransitionModel(noise_diff_coeff_const_acc)
            transition_models["ConstantVelocityXYConstantAccelerationZ"] = ConstantVelocityXYConstantAccelerationZTransitionModel(noise_diff_coeff_const_acc)
        return transition_models
    
    def get_measurement_model(self, n_targets, n_dims, noise_covar):
        if self.approximate_velocity_measurement:
            measurement_model = LinearGaussianMeasurementModel(ndim_state=(self.constant_derivative+1)*n_dims, 
                                           mapping=np.array([0,1,3,4,6,7]), # postion and velocity
                                           noise_covar=noise_covar*np.identity(n_dims*2)) # velocity and postion
        else:
            measurement_model = LinearGaussianMeasurementModel(ndim_state=(self.constant_derivative+1)*n_dims, 
                                            mapping=range(0,(self.constant_derivative+1)*n_dims, (self.constant_derivative+1)), 
                                            noise_covar=noise_covar*np.identity(n_dims))
        return measurement_model
    
    def get_priors(self, state_init, n_targets, n_dims, timestamp, init_state_noise_covar):
        tracks = []
        # print("Initial state:", state_init)
        # original
        # for n_target in range(0,n_targets):
        #     prior = GaussianState(state_init[n_target*n_dims*(self.constant_derivative+1):(n_target+1)*n_dims*(self.constant_derivative+1)], np.diag(init_state_noise_covar*np.ones((self.constant_derivative+1)*n_dims)), timestamp)
        #     tracks.append(Track([prior],id=n_target, marker_id=None))
        #jonathan hardcoded id
        for n_target in range(0,n_targets):
            #(self, state_vector, covar, timestamp)
            state_vector = state_init[n_target*n_dims*(self.constant_derivative+1):(n_target+1)*n_dims*(self.constant_derivative+1)]
            covar = np.diag(init_state_noise_covar*np.ones((self.constant_derivative+1)*n_dims))
            covar[[1,2,4,5,7,8], [1,2,4,5,7,8]] = 0.
            prior = GaussianState(state_vector, covar, timestamp)
            
            # prior = GaussianState(state_init[n_target*n_dims*(self.constant_derivative+1):(n_target+1)*n_dims*(self.constant_derivative+1)], np.diag(init_state_noise_covar*np.ones((self.constant_derivative+1)*n_dims)), timestamp)
            
            transition_model = self.transition_models["ConstantPosition"]
            
            tracks.append(Track([prior],id=n_target, marker_id=None, transition_model=transition_model))
            
        return tracks

    def measurement_update_with_association(self, measurement, timestamp):
        # start = time.time()
        # start = time.time()
        # for track in self.tracks:
        #     for detection in measurement:
        #         if detection.marker_id == track.marker_id:
        #             hypotheses[track] = SingleDistanceHypothesis(,,)         
        # start = time.time()   
        # for track in self.tracks:
        #     for detection in measurement:
        #         if detection.marker_id == track.marker_id:
        #             hypotheses[track] = SingleDistanceHypothesis(,,)         
        hypotheses = self.associate(self.tracks, measurement, timestamp)
        # end = time.time()
        # print("Data association loop time: ", (end - start)*1000)
        for track in self.tracks:
            hypothesis = hypotheses.hypotheses[track]
            if hypothesis.measurement:
                # start = time.time()
                post = self.updater.update(hypothesis)
                # track.states[-1] = post
                track.states.append(post)
                track.detections.append(hypothesis.measurement)
                # append all states
                track.marker_id = hypothesis.measurement.marker_id
                # end = time.time()
                # print("Kalman update time:", (end-start)*1000, "ms")
            else:  # When data associator says no detections are good enough, we'll keep the prediction
                # start = time.time()
                # print("Prediction used")
                # track.states[-1] = hypothesis.prediction
                track.states.append(hypothesis.prediction)
                # end = time.time()
                # print("Prediction loop time:", end-start)
        # end = time.time()
        # print("Tracking loop time: ", (end - start)*1000,"ms")


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
        # start_time = time.time()        
        hypotheses = list()
        prediction = self.predictor.predict(track.states[-1],  timestamp=timestamp, transition_model=track.transition_model)
        if detections:
            # Common state & measurement prediction
            # # Compute measurement prediction and distance measure
            measurement_prediction = self.updater.predict_measurement(
                    prediction, detections[0].measurement_model)
            # end_time = time.time()
            # if (np.any(np.linalg.eigvals(measurement_prediction.covar) < 0)):
            #     print("Measurement prediction covariance matrix:", measurement_prediction.covar)
            # print("Prediction time: ", (end_time-start_time)*1000, "ms")        
            # Missed detection hypothesis with distance as 'missed_distance'
        hypotheses.append(
            SingleDistanceHypothesis(
                prediction,
                MissedDetection(timestamp=timestamp),
                self.missed_distance
                ))


        # start_time = time.time()        
        # True detection hypotheses
        for detection in detections:
                
            distance = self.measure(measurement_prediction, detection)
            
            # print("Distance:", distance)
            # end_time = time.time()
            # print("Kalman predict + update + Euclidean distance computation time: ", (end_time-start_time)*1000, "ms")

            if distance < self.missed_distance:
                # True detection hypothesis
                hypotheses.append(
                    SingleDistanceHypothesis(
                        prediction,
                        detection,
                        distance,
                        measurement_prediction))
        # end_time = time.time()
        # print("True detection hypothesis loop time: ", (end_time-start_time)*1000, "ms")

        multiple_hypothesis= MultipleHypothesis(sorted(hypotheses, reverse=True))
        # end_time = time.time()
        # print("Hypothesis generation time: ", (end_time-start_time)*1000, "ms")

        return multiple_hypothesis
    
    def generate_hypotheses(self, tracks, detections, timestamp):
        return {track: self.hypothesise(
                    track, detections, timestamp)
                for track in tracks}

    def generate_single_hypothesis(self, track, detection, timestamp):
        prediction = self.predictor.predict(track.states[-1], timestamp=timestamp, transition_model=track.transition_model)
        measurement_prediction = self.updater.predict_measurement(prediction, detection.measurement_model)
        distance = 0.0
        return SingleDistanceHypothesis(prediction, detection, distance, measurement_prediction)
 
        
    
    def enumerate_joint_hypotheses(self, hypotheses):
        """Enumerate the possible joint hypotheses.

        Create a list of all possible joint hypotheses from the individual
        hypotheses and determine whether each is valid.

        Parameters
        ----------
        hypotheses : dict of :class:`~.Track`: :class:`~.Hypothesis`
            A list of all hypotheses linking predictions to detections,
            including missed detections

        Returns
        -------
        joint_hypotheses : list of :class:`DistanceJointHypothesis`
            A list of all valid joint hypotheses with a distance score on each
        """

        # Create a list of dictionaries of valid track-hypothesis pairs
        joint_hypotheses = [
            DistanceJointHypothesis({
                track: hypothesis
                for track, hypothesis in zip(hypotheses, joint_hypothesis)})
            for joint_hypothesis in itertools.product(*hypotheses.values())
            if self.isvalid(joint_hypothesis)]

        return joint_hypotheses
    
    def isvalid(self,joint_hypothesis):
        """Determine whether a joint_hypothesis is valid.

        Check the set of hypotheses that define a joint hypothesis to ensure a
        single detection is not associated to more than one track.

        Parameters
        ----------
        joint_hypothesis : :class:`JointHypothesis`
            A set of hypotheses linking each prediction to a single detection

        Returns
        -------
        bool
            Whether joint_hypothesis is a valid set of hypotheses
        """

        number_hypotheses = len(joint_hypothesis)
        unique_hypotheses = len(
            {hyp.measurement for hyp in joint_hypothesis if hyp})
        number_null_hypotheses = sum(not hyp for hyp in joint_hypothesis)

        # joint_hypothesis is invalid if one detection is assigned to more than
        # one prediction. Multiple missed detections are valid.
        if unique_hypotheses + number_null_hypotheses == number_hypotheses:
            return True
        else:
            return False
        
    def associate(self, tracks, detections, timestamp):
        # start = time.time()
        remaining_detections = []
        associated_detections = []
        associated_tracks = []
        single_hypotheses = {}
        
        if not self.use_only_kalman: 
            for track in tracks:
                for detection in detections:
                    if detection.marker_id == track.marker_id:
                        single_hypotheses[track] = self.generate_single_hypothesis(track, detection, timestamp)
                        associated_detections.append(detection)
                        associated_tracks.append(track)
                    else:
                        pass

        remaining_tracks = [track for track in tracks if track not in associated_tracks]
        remaining_detections = [detection for detection in detections if detection not in associated_detections]

        # Generate a set of hypotheses for each track on each detection
        remaining_hypotheses = self.generate_hypotheses(remaining_tracks, remaining_detections, timestamp)
        # end = time.time()
        # print("Hypotheses generation time: ", (end - start)*1000,"ms")

        # start = time.time()
        # Link hypotheses into a set of joint_hypotheses and evaluate
        joint_hypotheses = self.enumerate_joint_hypotheses(remaining_hypotheses)
        associations = max(joint_hypotheses)
        associations.hypotheses.update(single_hypotheses)
        # end = time.time()
        # print("Hypotheses association time: ", (end - start)*1000,"ms")
        return associations

class Euclidean():
    """Euclidean distance measure

    This measure returns the Euclidean distance between a pair of
    :class:`~.State` objects.

    The Euclidean distance between a pair of state vectors :math:`u` and
    :math:`v` is defined as:

    .. math::
         \sqrt{\sum_{i=1}^{N}{(u_i - v_i)^2}}

    """
    def __call__(self, state1, state2):
        r"""Calculate the Euclidean distance between a pair of state vectors

        Parameters
        ----------
        state1 : :class:`~.State`
        state2 : :class:`~.State`

        Returns
        -------
        float
            Euclidean distance between two input :class:`~.State`

        """
        # Calculate Euclidean distance between two state
        state_vector1 = getattr(state1, 'mean', state1.state_vector)
        state_vector2 = getattr(state2, 'mean', state2.state_vector)

        return distance.euclidean(state_vector1[:, 0], state_vector2[:, 0])

class Mahalanobis():
    r"""Mahalanobis distance measure

    This measure returns the Mahalanobis distance between a pair of
    :class:`~.State` objects taking into account the distribution (i.e.
    the :class:`~.CovarianceMatrix`) of the first :class:`.State` object

    The Mahalanobis distance between a distribution with mean :math:`\mu` and
    Covariance matrix :math:`\Sigma` and a point :math:`x` is defined as:

    .. math::
            \sqrt{( {\mu - x})  \Sigma^{-1}  ({\mu - x}^T )}

    state_covar_inv_cache_size : Number of covariance matrix inversions to cache. Setting to `0` will disable the
            cache, whilst setting to `None` will not limit the size of the cache. Default is "128."
    mapping : Mapping array which specifies which elements within the state vectors are to be assessed as part of the measure
    """
    def __init__(self, state_covar_inv_cache_size=128, mapping = None):
        self.state_covar_inv_cache_size = state_covar_inv_cache_size
        self.mapping = mapping
        if self.state_covar_inv_cache_size is None or self.state_covar_inv_cache_size > 0:
            self._inv_cov = lru_cache(maxsize=self.state_covar_inv_cache_size)(self._inv_cov)

    def __getstate__(self):
        result = copy.copy(self.__dict__)
        result["_inv_cov"] = None
        return result

    def __setstate__(self, state):
        self.__dict__ = state
        if self.state_covar_inv_cache_size is None or self.state_covar_inv_cache_size > 0:
            self._inv_cov = lru_cache(maxsize=self.state_covar_inv_cache_size)(type(self)._inv_cov)
        else:
            self._inv_cov = type(self)._inv_cov

    def __call__(self, state1, state2, skip_velocities=False, down_weight_velocities=False):
        r"""Calculate the Mahalanobis distance between a pair of state objects

        Parameters
        ----------
        state1 : :class:`~.State`
        state2 : :class:`~.State`

        Returns
        -------
        float
            Mahalanobis distance between a pair of input :class:`~.State` objects

        """
        state_vector1 = getattr(state1, 'mean', state1.state_vector)
        state_vector2 = getattr(state2, 'mean', state2.state_vector)
        if self.mapping is not None:
            print(state_vector1.shape)
            u = state_vector1[self.mapping, 0]
            v = state_vector2[self.mapping2, 0]
            # extract the mapped covariance data
            vi = self._inv_cov(state1, tuple(self.mapping))
        elif skip_velocities:
            mapping = [0,3,5]
            mapping = [0,2,4]
            u = state_vector1[mapping, 0]
            v = state_vector2[mapping, 0]
            vi = self._inv_cov(state1, tuple(mapping))
        elif down_weight_velocities:
            down_weight_factor = 0.3
            mapping_pos = [0,2,4]
            mapping_vel = [1,3,5]            
            u = state_vector1[:, 0]
            u[mapping_vel] = down_weight_factor*state_vector1[mapping_vel, 0]
            v = state_vector2[:, 0]
            v[mapping_vel] = down_weight_factor*state_vector2[mapping_vel, 0]
            
            vi = self._inv_cov(state1)    
        else:
            u = state_vector1[:, 0]
            v = state_vector2[:, 0]
            vi = self._inv_cov(state1)
        delta = u - v
        
        
        # # Check for NaN or Inf in state vectors
        # if np.any(np.isnan(state_vector1)) or np.any(np.isnan(state_vector2)):
        #     print("NaN values detected in state vectors")
        # if np.any(np.isinf(state_vector1)) or np.any(np.isinf(state_vector2)):
        #     print("Inf values detected in state vectors")

        # # Check for NaN or Inf in the inverse covariance matrix
        # if np.any(np.isnan(vi)) or np.any(np.isinf(vi)):
        #     print("NaN or Inf values detected in inverse covariance matrix")
        
        # print("Delta:", delta)
        # print("Inverse Covariance Matrix (vi):", vi)
        # mahalanobis_value = np.dot(np.dot(delta, vi), delta)
        # print("Mahalanobis value before sqrt:", mahalanobis_value)
            
        # if mahalanobis_value < 0:
        #     print("Negative Mahalanobis value detected, setting to 0")
        #     mahalanobis_value = 0  # Set to 0 to avoid sqrt of negative number    
    
        # condition_number = np.linalg.cond(vi)
        # if condition_number > 1e10:
        #     print("Warning: Inverse covariance matrix is ill-conditioned with condition number:", condition_number)

            
        # # mahalanobis_value = np.dot(np.dot(delta, vi), delta)
        # # # Prevent negative values under the square root (numerical stability)
        # # mahalanobis_value = max(mahalanobis_value, 0)
        # # return np.sqrt(mahalanobis_value)   

        # print('state1.covar: ', state1.covar)

        return np.sqrt(np.dot(np.dot(delta, vi), delta))

    @staticmethod
    def _inv_cov(state, mapping=None):
        if mapping:
            rows = np.array(mapping, dtype=np.intp)
            columns = np.array(mapping, dtype=np.intp)
            covar = state.covar[rows[:, np.newaxis], columns]
        else:
            covar = state.covar

        return np.linalg.inv(covar)


    


