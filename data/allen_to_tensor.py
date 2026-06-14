# author: Guillaume Bellec
# date: 2020 / 2021
from tqdm import tqdm
import os
import shutil
from sys import getsizeof
import multiprocessing
from utils.functions import parmap
import pickle
#from models.preprocessing import get_preprocessing_transforms
from PIL import Image

import numpy as np
import pandas as pd
import warnings

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from skimage.transform import resize
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from utils.functions import to_numpy


class AllenToTensor(object):

    def __init__(self, session_id=791319847,
                 stimulus="natural_movie_one", dt=0.01,
                 cache_root="./",
                 ephys_data_directory='ecephys_cache_dir',
                 allen_to_tensor_directory='allen_to_tensor_cache_dir',
                 boc_cache_directory='boc_cache_dir/manifest.json',
                 verbose=True):

        if verbose:
            print("Init AllenToTensor sessions_id {} stimulus {}".format(session_id, stimulus) )

        ephys_data_directory = os.path.join(cache_root, ephys_data_directory)
        allen_to_tensor_directory = os.path.join(cache_root, allen_to_tensor_directory)
        boc_cache_directory = os.path.join(cache_root, boc_cache_directory)

        self.boc = BrainObservatoryCache(manifest_file=boc_cache_directory)
        self.session_id = session_id
        self.stimulus = stimulus
        self.dt = dt
        self.verbose = verbose

        self.ephys_data_directory = ephys_data_directory
        self.allen_to_tensor_directory = allen_to_tensor_directory

        self._session = None

        if verbose: print("Created AllenToTensor for session:", session_id)

    def download_natural_scenes(self):
        data_set = self.boc.get_ophys_experiment_data(501498760)
        scenes = data_set.get_stimulus_template('natural_scenes')
        return scenes

    def download_natural_movie_one(self):
        data_set = self.boc.get_ophys_experiment_data(501498760)
        movie = data_set.get_stimulus_template('natural_movie_one')
        return movie

    def download_natural_movie_three(self):
        data_set = self.boc.get_ophys_experiment_data(501940850)
        scenes = data_set.get_stimulus_template('natural_movie_three')
        return scenes

    def get_session(self):
        if self._session != None:
            return self._session

        if self._session == None:

            verbose = self.verbose
            session_id = self.session_id

            if verbose:
                print("Open cache: ", self.ephys_data_directory)

            cache = self.get_ephys_cache()
            sessions = cache.get_session_table()
            sessions.head()

            if verbose: print("Open session: ", session_id)
            self._session = cache.get_session_data(session_id)
            # if verbose: print("Session is now loaded.")
            # self.session_metadata = self.session.metadata
            # if verbose:
            #    print(self.session_metadata)
            return self._session

    @staticmethod
    def get_session_id_list(session_type=None, allen_to_tensor_directory='./ecephys_cache_dir'):
        manifest_path = os.path.join(allen_to_tensor_directory, "manifest.json")
        cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

        sessions = cache.get_session_table()
        sessions.head()

        if session_type is not None:
            return sessions[sessions['session_type'] == session_type].index

        return sessions.index

    def get_ephys_cache(self):
        manifest_path = os.path.join(self.ephys_data_directory, "manifest.json")
        ephys_cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
        return ephys_cache

    def get_unit_mask(self, area=None):
        all_indices = self.get_units_indices()
        area_indices = self.get_units_indices(area)
        return np.isin(all_indices, area_indices)

    @staticmethod
    def full_list_of_areas():
        D = AllenToTensor.get_area_selections()
        full_list = D['visual'] + D['hippo'] + D['thalamus'] + D['midbrain']
        return full_list

    @staticmethod
    def get_area_selections():

        # custom selection
        visual_hierarchy_from_Siegle_et_al_2020 = ["LGd", "VISp", "VISrl", "VISl", "VISal", "VISpm", "VISam"]

        # anatomical partition
        visual_areas_list = ["VISp", "VISrl", "VISl", "VISal", "VISpm", "VISam"]
        hippocampal_formation_list = ["CA1", "CA3", "DG", "SUB", "ProS"]
        thalamus_list = ["LGd", "LP"]
        midbrain_list = ["APN"]

        D = {'visual': visual_areas_list,
             'hippo': hippocampal_formation_list,
             'thalamus': thalamus_list,
             'midbrain': midbrain_list,
             'siegle2020': visual_hierarchy_from_Siegle_et_al_2020}

        return D

    def compute_unit_signs_and_waveforms(self, unit_ids, verbose=False):

        session = self.get_session()

        rise_time_list = []
        peak_width_list = []
        waveform_list = []

        # compute rise time and peak width for each unit
        for k, unit_id in enumerate(unit_ids):
            wf = session.mean_waveforms[unit_id]
            waveform_list += [wf]
            time_line = np.array(wf.time) * 1000 # in ms
            n_channel, n_time = wf.shape

            i_0_5 = np.argmax(time_line >= 0.5)
            i_1_0 = np.argmax(time_line >= 1.0)
            i_1_5 = np.argmax(time_line >= 1.5)
            i_2_0 = np.argmax(time_line >= 2.0)

            channel_score = (wf[:, i_0_5:i_1_0].max('time') - wf[:, i_0_5:i_2_0].min('time'))  # wf_np.std(1)
            i_best = np.argmax(channel_score.data)

            waveform = np.array(wf[i_best])

            i_trough = int(np.argmin(waveform[:i_1_0]))
            i_peak = int(i_trough + np.argmax(waveform[i_trough:i_2_0]))
            w_trough = waveform[i_trough]
            w_peak = waveform[i_peak]
            t_trough = time_line[i_trough]
            t_peak = time_line[i_peak]

            if not t_peak >= t_trough:
                warnings.warn("peak-trough: {},{:.2f} and {},{:.2f}".format(i_peak, t_peak, i_trough, t_trough))

            w_thr = (w_peak + waveform[-1]) / 2
            i_half_peak1 = int(i_peak)
            i_half_peak2 = int(i_peak)
            while waveform[i_half_peak1] > w_thr:
                i_half_peak1 -= 1

            w_half_peak1 = waveform[i_half_peak1]
            t_half_peak1 = time_line[i_half_peak1]

            while waveform[i_half_peak2] > w_thr:
                i_half_peak2 += 1
                if i_half_peak2 == n_time - 1: break

            if not i_half_peak2 >= i_half_peak1:
                warnings.warn("found half width at: {} and {} (peak is {})".format(i_half_peak1,i_half_peak2,i_peak))

            w_half_peak2 = waveform[i_half_peak2]
            t_half_peak2 = time_line[i_half_peak2]

            if not t_half_peak2 >= t_half_peak1:
                msg = "found half width at: {},{:.2f} and {},{:.2f} (trough is {},{:.2f} and peak is {}, {:.2f})".format(\
                    i_half_peak1, t_half_peak2, i_half_peak2, t_half_peak1, i_trough, t_trough, i_peak, t_peak)
                warnings.warn(msg)

            rise_time = float(t_peak - t_trough)
            peak_width = float(t_half_peak2 - t_half_peak1)

            rise_time_list += [rise_time]
            peak_width_list += [peak_width]

        data = np.array([rise_time_list, peak_width_list])

        z = gaussian_kde(data)(data)
        outliers = z <= 2 * np.min(z)
        #print("outliers: {:.2f}".format(np.sum(outliers == True) / np.size(outliers)))
        #data_no_outliers = data[:, np.logical_not(outliers)]

        kmeans = KMeans(n_clusters=2,)
        kmeans.fit(data.T)
        # overwrite the cluster centers from session 771160300
        # kmeans.cluster_centers_ = np.array([[0.642, 0.676],[0.298, 0.342]])
        c = kmeans.predict(data.T)
        c_uniques = np.unique(c)
        c_freq = [np.sum(c == i) / np.size(c) for i in c_uniques]
        rise_time_per_c = [np.mean(np.array(rise_time_list)[c == v]) for v in c_uniques]
        c_excitatory = c_uniques[np.argmax(rise_time_per_c)] # excitatory are slow spiking
        sign = np.where(c == c_excitatory, np.ones_like(c), - np.ones_like(c))
        sign[outliers] = 0

        if verbose:
            import matplotlib.pyplot as plt
            _, ax_list = plt.subplots(2)

            print("cluster centers", kmeans.cluster_centers_)

            for i in np.unique(sign):
                print("sign={}: {:2f}".format(i, np.sum(sign == i) / np.size(sign)))

            ax = ax_list[0]
            ax.scatter(data[0], data[1], c=sign, cmap='jet')
            ax.legend()
            ax.set_xlabel("rise time")
            ax.set_ylabel("peak width")
            ax.set_xlim([0,1])
            ax.set_ylim([0,1])

            for s, unit_id in zip(sign,unit_ids):
                wf = session.mean_waveforms[unit_id]
                time_line = wf.time * 1000
                channel_score = (wf.max('time') - wf.min('time'))  # wf_np.std(1)
                i_best = np.argmax(channel_score)
                waveform = wf[i_best]
                color_dict = {-1: "blue", 1: "red", 0: "green"}
                ax_list[1].plot(time_line, waveform, color=color_dict[s], alpha=0.2, lw=2)

            plt.show()

        return sign,  waveform_list

    def compute_unit_electrode_depth(self, record_unit_ids):
        sess = self.get_session()
        df = sess.units.loc[record_unit_ids]
        probe_id = df['probe_id']
        v_loc = df['probe_vertical_position']
        area = df['ecephys_structure_acronym']
        return probe_id, v_loc, area

    def compute_unit_ccf_positions(self, record_unit_ids):
        sess = self.get_session()

        df = sess.units.loc[record_unit_ids]
        x = df['anterior_posterior_ccf_coordinate']
        y = df['dorsal_ventral_ccf_coordinate']
        z = df['left_right_ccf_coordinate']
        xyz = [x, y, z]
        xyz = [a.to_numpy() for a in xyz]
        return np.stack(xyz, -1)

    def compute_units_indices(self, area=None):
        cache = self.get_ephys_cache()

        area_dict = self.get_area_selections()
        full_list = self.full_list_of_areas()

        # default filtering parameters as recommended
        filtered_units = cache.get_units(amplitude_cutoff_maximum=0.1,
                                         presence_ratio_minimum=0.9,
                                         isi_violations_maximum=0.5)

        if area is None:
            return self.compute_units_indices(full_list)

        session = self.get_session()
        if isinstance(area,str) and area in full_list:
            area_list = [area]
        elif isinstance(area,str) and area in area_dict.keys():
            area_list = area_dict[area]
        elif isinstance(area, list):
            area_list = area
        else:
            raise NotImplementedError("area \'{}\' not understood".format(area))

        # example: Area ["VISl", "VISrl", "VISpm", "VISam"]
        units = session.units[session.units["ecephys_structure_acronym"].isin(area_list)]
        units = units.index.intersection(filtered_units.index)
        return units

    def get_area_string(self, area):
        if not area: area_string = "all"
        elif isinstance(area,list): area_string = "_".join(area)
        elif isinstance(area,str): area_string = area
        else: raise ValueError("Expected area request: {} ".format(area))
        return area_string

    def get_units_indices(self, area=None):
        area_string = self.get_area_string(area)
        file_name = "unit_indices_all_areas" if not area else "unit_indices_" + area_string
        fn = lambda: self.compute_units_indices(area)
        unit_ids = self.compute_and_save_or_load_from_cache(file_name, fn, cache_specs=[self.session_id, self.stimulus])
        return unit_ids

    def get_unit_ids_signs_and_waveforms(self, area=None):
        area_string = self.get_area_string(area)
        file_name = "unit_indices_signs_and_waveforms_data_" + area_string

        def fn():
            unit_ids = self.get_units_indices(area)
            signs, waveforms = self.compute_unit_signs_and_waveforms(unit_ids)
            return unit_ids, signs, waveforms

        return self.compute_and_save_or_load_from_cache(file_name, fn, cache_specs=[self.session_id])

    def truncate_and_concatenate_time_lines(self, time_line_list):
        # all repetitions should have the same length
        min_length = min([t.size for t in time_line_list])
        median_length = np.median([t.size for t in time_line_list])

        # all the trials should have almost the same length and it should be close to the normal trial length
        # this raises an error if one trial was badly recorded and it's very short.
        assert min_length >= median_length * 0.8, "Median number of time steps per trials would be {} but one trial has only {} time steps".format(
            median_length, min_length)

        time_lines = [t[:min_length] for t in time_line_list]
        time_lines = np.stack(time_lines)
        return time_lines

    def get_scene_presentations_of_stim(self):
        session = self.get_session()

        if self.stimulus == "drifting_gratings" and session.session_type == "functional_connectivity":
            return session.get_stimulus_table(stimulus_names="drifting_gratings_75_repeats")

        scene_presentations = session.get_stimulus_table(stimulus_names=self.stimulus)
        assert len(scene_presentations) > 0
        return scene_presentations

    def get_drifting_gratings_time_line(self):
        stimulus = self.stimulus
        dt = self.dt

        assert stimulus in ["drifting_gratings"]
        scene_presentations = self.get_scene_presentations_of_stim()

        time_lines = []
        for idx in scene_presentations.index:
            t_start = scene_presentations["start_time"][idx]
            t_stop = scene_presentations["stop_time"][idx]
            time_lines.append(np.arange(t_start, t_stop - dt, dt))

        return self.truncate_and_concatenate_time_lines(time_lines)

    def get_movie_timeline(self):
        session = self.get_session()
        stimulus = self.stimulus
        dt = self.dt

        assert stimulus in ["natural_movie_one", "natural_movie_three"]
        scene_presentations = self.get_scene_presentations_of_stim()

        first_frames = scene_presentations[scene_presentations["frame"] == 0]
        number_of_frames = first_frames.index[1] - first_frames.index[0]
        last_frames = scene_presentations[scene_presentations["frame"] == number_of_frames - 1]

        start_times = []
        stop_times = []
        durations = []
        time_lines = []
        for k_rep in range(len(first_frames.index)):
            t_start = scene_presentations["start_time"][first_frames.index[k_rep]]
            t_stop = scene_presentations["stop_time"][last_frames.index[k_rep]]

            start_times.append(t_start)
            stop_times.append(t_stop)
            durations.append(t_stop - t_start)
            time_lines.append(np.arange(t_start, t_stop - dt, dt))

        # all repetitions should have the same length
        min_length = min([t.size for t in time_lines])
        for time_line in time_lines:
            assert time_line.size < min_length + 5, "Found time line with length {} and the shorter has length {}".format(time_line.size,min_length)

        time_lines = [t[:min_length] for t in time_lines]
        time_lines = np.stack(time_lines)
        return time_lines

    def get_natural_scenes_timeline(self):

        session = self.get_session()
        stimulus = self.stimulus
        dt = self.dt

        assert stimulus in ["natural_scenes"]
        scene_presentations = self.get_scene_presentations_of_stim()

        t_start = scene_presentations["start_time"][scene_presentations.index[0]]
        t_stop = scene_presentations["stop_time"][scene_presentations.index[-1]]
        time_lines = np.arange(t_start, t_stop - dt, dt)

        # add a first axis for the repetition index
        time_lines = time_lines[None, ...]

        return time_lines

    def get_time_line(self):
        file_name = "time_line"
        fn = lambda: self.compute_time_line()
        return self.compute_and_save_or_load_from_cache(file_name, fn, cache_specs=[self.session_id, self.stimulus, self.dt])

    def compute_time_line(self):

        stimulus = self.stimulus

        if "natural_movie" in stimulus:
            time_line = self.get_movie_timeline()
        elif stimulus == "natural_scenes":
            time_line = self.get_natural_scenes_timeline()
        elif stimulus == "drifting_gratings":
            time_line = self.get_drifting_gratings_time_line()
        else:
            raise ValueError("Stimulus {} not understood".format(stimulus))

        return time_line

    def get_frame_indices(self):
        raise DeprecationWarning()
        file_name = "frames"
        fn = lambda: self.compute_frame_indices(self.get_time_line())
        return self.compute_and_save_or_load_from_cache(file_name, fn, cache_specs=[self.session_id, self.stimulus, self.dt])

    def compute_frame_indices_for_drifting_gratings(self):
        raise NotImplementedError()
        session = self.get_session()
        stimulus = self.stimulus

        assert stimulus in ["drifting_gratings"]
        scene_presentations = self.get_scene_presentations_of_stim()

        condition_ids = np.unique(scene_presentations["stimulus_condition_id"])

        n_trials, n_time = self.get_time_line().shape
        frame_indices = np.zeros((n_trials, n_time), dtype=int)
        condition_idx_range = self.get_trial_condition_vector()
        for k,idx in enumerate(scene_presentations.index):
            condition_idx = scene_presentations["stimulus_condition_id"][idx]
            i = np.argmax(condition_idx_range == condition_idx)
            frame_indices[k,:] = np.arange(n_time) + i * n_time

        return frame_indices

    def get_trial_condition_vector(self,key):
        file_name = "trial_condition_vector_" + key
        fn = lambda: self.compute_trial_condition_vector(key)
        return self.compute_and_save_or_load_from_cache(file_name, fn, cache_specs=[self.session_id, self.stimulus, key])

    def get_drifting_condition_specs(self):
        file_name = "drifting_condition_spec_dict"
        fn = lambda: self.compute_drifting_condition_specs()
        return self.compute_and_save_or_load_from_cache(file_name, fn, cache_specs=[self.session_id, self.stimulus])

    def compute_drifting_condition_specs(AtoT):
        condition_ids = AtoT.compute_trial_condition_vector("stimulus_condition_id")
        condition_spec_keys = ["stimulus_condition_id", 'orientation', 'contrast', "temporal_frequency"]
        condition_spec_dicts = {}

        spec_vectors = [AtoT.compute_trial_condition_vector(k) for k in condition_spec_keys]

        for i_c, c_id in enumerate(condition_ids):

            if not c_id in condition_spec_dicts.keys():
                condition_spec_dicts[c_id] = {}
                for key, spec_vector in zip(condition_spec_keys, spec_vectors):
                    condition_spec_dicts[c_id][key] = spec_vector[i_c]
            else:
                for key, spec_vector in zip(condition_spec_keys, spec_vectors):
                    assert condition_spec_dicts[c_id][key] == spec_vector[i_c]

        return condition_spec_dicts

    def compute_trial_condition_vector(self, key):
        if not key in ["stimulus_condition_id", "orientation", "temporal_frequency", "contrast"]:
            warnings.warn(f"not sure the key {key} can provide a meaningful output")

        session = self.get_session()
        stimulus = self.stimulus
        scene_presentations = self.get_scene_presentations_of_stim()

        condition_val_list = []
        for k, idx in enumerate(scene_presentations.index):
            condition_val = scene_presentations[key][idx]
            condition_val_list.append(condition_val)

        condition_vector = np.array(condition_val_list)
        n_trial, n_time = self.get_time_line().shape

        assert condition_vector.size == n_trial, "condition vector {} has size {} but there are {} trials".format(condition_vector, condition_vector.shape,n_trial)
        return condition_vector

    def compute_frame_indices(self, time_lines):
        if self.stimulus == "drifting_gratings": return self.compute_frame_indices_for_drifting_gratings()

        session = self.get_session()
        stimulus = self.stimulus
        dt = self.dt

        scene_presentations = self.get_scene_presentations_of_stim()

        frames = -np.ones(time_lines.shape, dtype=int)

        for k, idx_stim in enumerate(scene_presentations.index):
            frame_idx = scene_presentations["frame"][idx_stim]

            start_time = scene_presentations["start_time"][idx_stim]
            stop_time = scene_presentations["stop_time"][idx_stim]

            overlap = np.maximum(0, np.minimum(stop_time, time_lines + dt) - np.maximum(start_time, time_lines))
            frames[overlap > dt / 2] = frame_idx

        number_of_time_steps_without_frames = np.sum(frames == -1)
        if number_of_time_steps_without_frames > 0:
            print(np.where(frames == -1))
            msg = "There are {}/{} time steps that could not be associated to any frames.".format(
                number_of_time_steps_without_frames, frames.size)

            if self.stimulus != "natural_scenes": raise ValueError(msg)

        return frames

    def find_file(self, folder, file_name, strict_extension=None):
        if strict_extension:
            assert strict_extension.startswith(".")
            full_file_name = file_name + strict_extension
            full_path = os.path.join(folder, full_file_name)
            if os.path.exists(full_path):
                return full_file_name
            else:
                return None

        for f in os.listdir(folder):
            if f == file_name + ".npy":
                return f
            elif f == file_name + ".pickle":
                return f
        return None

    def compute_and_save_or_load_from_cache(self, file_name, lambda_function, strict_extension=None, cache_specs=None):

        if cache_specs is None: cache_specs = [self.session_id, self.stimulus, self.dt]
        prefix = "_".join([str(i) for i in cache_specs])

        folder = os.path.join(self.allen_to_tensor_directory, prefix)
        os.makedirs(folder, exist_ok=True)

        full_file_name = self.find_file(folder, file_name, strict_extension)

        if full_file_name is not None:
            full_path = os.path.join(folder, full_file_name)
            if full_path.endswith(".npy"):
                O = np.load(full_path, allow_pickle=True)
            elif full_path.endswith(".pickle"):
                with open(full_path, "rb") as f:
                    O = pickle.load(f)
            else:
                raise NotImplementedError(f"cannot load file {full_path}")

            if self.verbose:
                size = getsizeof(O) / 1024 ** 2
                print("Loaded \'{}\' from cache, size: {:.4f} Mb".format(full_file_name, size))
            return O

        else:
            # compute
            O = lambda_function()
            # save
            if O is not None:
                if not os.path.exists(self.allen_to_tensor_directory): os.mkdir(self.allen_to_tensor_directory)
                if not os.path.exists(folder): os.mkdir(folder)

                if isinstance(O, np.ndarray) and (strict_extension != ".pickle"):
                    full_file_name = file_name + ".npy"
                    np.save(os.path.join(folder, full_file_name), O)
                else:
                    full_file_name = file_name + ".pickle"
                    with open(os.path.join(folder, full_file_name), "wb") as f:
                        pickle.dump(O,f)

                if self.verbose:
                    size = getsizeof(O) / 1024 ** 2

                    print("Saved \'{}\' to cache, size: {:.4f} Mb ".format(full_file_name, size))

        return O

    def get_all_valid_units(self):
        return self.get_units_indices(None)

    def behavioral_dt(self):
        n_group_dt = int(1 / 30 / self.dt)
        group_dt = self.dt * n_group_dt
        return group_dt, n_group_dt

    def compute_behavioral_data(self):
        group_dt, n_group_dt = self.behavioral_dt()
        time_line = self.get_time_line()
        pads = time_line[:, -1:]
        time_line = np.concatenate([time_line] + [pads] * (n_group_dt - 1), 1)
        grouped_time_line = time_line[:, ::n_group_dt]

        n_trials, T_grouped = grouped_time_line.shape

        sess = self.get_session()
        pupil_data = sess.get_pupil_data()
        r_data = sess.running_speed

        pupil_tensor = np.zeros((n_trials, T_grouped, pupil_data.shape[-1]))
        running_tensor = np.zeros((n_trials, T_grouped))

        for i_trial in tqdm(range(n_trials)):
            for i_dt_group in range(T_grouped):
                t0 = grouped_time_line[i_trial, i_dt_group]

                pupil_mask = np.logical_and(pupil_data.index >= t0, pupil_data.index < t0 + group_dt)
                pupil_tensor[i_trial, i_dt_group, :] = pupil_data[pupil_mask].mean().values

                running_mask = np.logical_and(r_data['start_time'] >= t0 - group_dt / 2,
                                              r_data['end_time'] < t0 + group_dt * 3 / 2)
                running_tensor[i_trial, i_dt_group] = r_data[running_mask]['velocity'].mean()

        pupil_tensor = (pupil_tensor - pupil_tensor.mean((0,1))) / (1e-12 + pupil_tensor.std((0,1)))
        running_tensor = (running_tensor - running_tensor.mean(0)) / (1e-12 + running_tensor.std(0))

        return grouped_time_line, pupil_tensor, running_tensor

    def compute_raster_plot(self):
        if self.verbose: print("Starting raster plot computation.")
        session = self.get_session()

        scene_presentations = self.get_scene_presentations_of_stim()
        if self.verbose: "Scene presentations are loaded."
        time_lines = self.get_time_line()

        if self.verbose: print("time line shape: ", time_lines.shape)

        unit_indices = self.get_units_indices()

        spikes = session.presentationwise_spike_times(
            stimulus_presentation_ids=scene_presentations.index.values,
            unit_ids=unit_indices
        )

        start_times = time_lines
        stop_times = time_lines + self.dt

        start_times = start_times.flatten()
        stop_times = stop_times.flatten()
        reversed_start_times = np.flip( - start_times)
        total_times = len(start_times)
        assert all(np.diff(start_times) > 0)
        assert all(np.diff(stop_times) > 0)

        if self.verbose:
            print("Number of units:", unit_indices.size)
            print("Number of spikes:", len(spikes.index))

        n_units = len(unit_indices)
        spike_times_list = []
        for k_unit in range(n_units):
            unit_idx = unit_indices[k_unit]
            z = spikes[spikes["unit_id"] == unit_idx]
            spike_times = np.array([z.index[k] for k in range(len(z.index))])
            spike_times_list.append(spike_times)

        pbar = tqdm(total=n_units)

        def process_unit(spike_times):
            sub_raster = np.zeros(start_times.size, dtype=np.int8)

            for spike_time in spike_times:

                i = np.searchsorted(stop_times, spike_time)
                if i < total_times:
                    if start_times[i] <= spike_time < stop_times[i]:
                        # this if condition is important to avoid the spikes happening before start_times[i]
                        # but after stop_times[i-1], these spikes are in the inter-trial interval
                        sub_raster[i] = 1

            pbar.update(1)
            return sub_raster

        def process_unit_tensor(spike_times):

            sub_raster = np.zeros((time_lines.shape[0], time_lines.shape[1],), dtype=np.int8)

            for spike_time in spike_times:

                wh = np.where(np.logical_and(spike_time >= start_times, spike_time < stop_times))
                
                if (len(wh[0]) > 1):
                    print("Spike time:", spike_time)
                    print("time bin positions:", wh)
                    raise ValueError("A spike can be in only one time bin, got {}".format(len(wh[0])))
                elif len(wh[0]) == 0:
                    pass
                    # k_missed += 1
                else:
                    sub_raster[wh[0][0], wh[1][0]] = 1

            pbar.update(1)
            return sub_raster

        n_jobs = min(max(1,multiprocessing.cpu_count() -1),10)
        print("Starting parallel loop over unit indices with {} jobs.".format(n_jobs))

        rasters = parmap(process_unit, spike_times_list, n_jobs)
        raster = np.stack(rasters, 1)
        assert raster.sum() > 0
        raster = raster.reshape((time_lines.shape[0], time_lines.shape[1], len(rasters)))
        pbar.close()

        print("The raster of shape {} has {} spikes. (total spikes={})".format(raster.shape, np.sum(raster), len(spikes.index)))
        return raster

    def compute_vertical_position_dict_range(AtoT):
        sess = AtoT.get_session()

        # get all units
        probe_id = sess.units['probe_id']
        v_loc = sess.units['probe_vertical_position']
        unit_area = sess.units['ecephys_structure_acronym']

        depth_range_dict = {}
        for probe in np.unique(probe_id):
            depth_range_dict[probe] = {}
            for area in np.unique(unit_area):
                arr = v_loc[np.logical_and(probe_id == probe, unit_area == area)]
                depth_range_dict[probe][area] = arr.to_numpy()

        return depth_range_dict

    def get_vertical_position_dict_range(self):
        file_name = "vertical_position_dict_range"
        fn = lambda: self.compute_vertical_position_dict_range()
        return self.compute_and_save_or_load_from_cache(file_name, fn, cache_specs=[self.session_id])

    def get_raster_plot(self):
        file_name = "raster"
        fn = lambda: self.compute_raster_plot()
        return self.compute_and_save_or_load_from_cache(file_name, fn, cache_specs=[self.session_id, self.stimulus, self.dt])

    def get_behavioral_data(self):
        file_name = "behavior"
        fn = lambda: self.compute_behavioral_data()
        return self.compute_and_save_or_load_from_cache(file_name, fn, cache_specs=[self.session_id, self.stimulus, self.dt])

    def compute_relative_depth(AtoT, record_unit_ids, n_min=10):
        unit_probe_id, unit_v_loc, unit_area = AtoT.compute_unit_electrode_depth(record_unit_ids)

        vertial_pos_dict_range = AtoT.get_vertical_position_dict_range()

        relative_depth = []
        for probe_id, v_loc, area in zip(unit_probe_id, unit_v_loc, unit_area):
            v_pos = vertial_pos_dict_range[probe_id][area]

            relative_v_pos = (v_loc - np.min(v_pos)) / (np.max(v_pos) - np.min(v_pos)) \
                if len(v_pos) >= n_min else np.nan
            relative_depth += [1 - relative_v_pos]
        return relative_depth

    def get_relative_depth(self, unit_ids, n_min=10):
        file_name = "relative_depth"
        h = hash(unit_ids.data.tobytes())
        fn = lambda: self.compute_relative_depth(unit_ids, n_min=n_min)
        return self.compute_and_save_or_load_from_cache(file_name, fn, cache_specs=[self.session_id, h, n_min])

    def compute_dorsal_ventral_ccf(AtoT, unit_ids):
        sess = AtoT.get_session()
        dorsal_ventral_ccf = sess.units['dorsal_ventral_ccf_coordinate']
        return dorsal_ventral_ccf[unit_ids].values

    def get_dorsal_ventral_ccf(self, unit_ids):
        file_name = "dorsal_ventral_ccf"
        unit_ids = to_numpy(unit_ids)
        h = hash(unit_ids.data.tobytes())
        fn = lambda: self.compute_dorsal_ventral_ccf(unit_ids)
        return self.compute_and_save_or_load_from_cache(file_name, fn, cache_specs=[self.session_id, h])

    def compute_dict_of_unit_ids_per_area(self):
        full_list = AllenToTensor.full_list_of_areas()

        neuron_unit_dict = {}
        for area in full_list:
            area_unit_ids = self.get_units_indices(area)
            neuron_unit_dict[area] = area_unit_ids
        return neuron_unit_dict

    def get_dict_of_unit_ids_per_area(self):
        file_name = "neuron_units_dict"
        fn = lambda: self.compute_dict_of_unit_ids_per_area()
        return self.compute_and_save_or_load_from_cache(file_name, fn, ".pickle", cache_specs=[self.session_id])

    def get_raster_plot_of_selected_units(self, area=None):
        raster = self.get_raster_plot()
        area_unit_ids = self.get_units_indices(area)

        # the two 'use_mask' options should be equal,
        # not sure which is fastest.
        all_units = list(self.get_units_indices())
        selection = [all_units.index(u) for u in area_unit_ids]
        sub_raster = raster[:, :, selection]

        return area_unit_ids, sub_raster

    def download_or_create_stimulus_template(self):
        if self.stimulus == "natural_movie_one":
            return self.download_natural_movie_one()
        elif self.stimulus == "natural_movie_three":
            return self.download_natural_movie_three()
        elif self.stimulus == "natural_scenes":
            return self.download_natural_scenes()
        elif self.stimulus == "drifting_gratings":
            return self.create_drifting_gratings_movies()

    def create_drifting_gratings_movies(self):
        """
        This function returns three dictionaries all indexed by the condition ids.
        The condition ids are integers but do not start at 0, are not not neccesarily continuous.

        The first dict contains the specifications: temporal frequency, spatial freq,

        """

        # start psychopy, on server disable open_gl messages
        from psychopy import visual, core
        scene_presentations = self.get_scene_presentations_of_stim()

        # these are the ids referenced in the database.
        condition_keys = ["temporal_frequency", "spatial_frequency", "orientation", "contrast", "duration"]

        stimulus_spec_dict = {}

        # fetch the right orientation and temporal frequency for every movie
        for idx in scene_presentations.index:
            condition_id = scene_presentations["stimulus_condition_id"][idx]

            if scene_presentations["size"][idx] != "[250.0, 250.0]" or scene_presentations["phase"][idx] != "[14037.96666667, 14037.96666667]":
                # for now we hard coe these values because we did not manage to part those string to numpy arrays
                # this should fails for adata
                raise NotImplementedError()

            if not condition_id in stimulus_spec_dict.keys():
                s = "condition id {}:".format(condition_id)
                d = {}
                for k in condition_keys:
                    d[k] = scene_presentations[k][idx]
                    s += "  {}: {}".format(k, d[k])
                print(s)
                stimulus_spec_dict[condition_id] = d

            # double check that all conditions have the correct value
            d = stimulus_spec_dict[condition_id]
            for k in d.keys():
                v = scene_presentations[k][idx]
                if k == "duration":
                    assert np.abs(d[k] - v) < 0.1
                else:
                    assert d[k] == v, "mismatch for spec: {} got {} and {}".format(k, d[k], v)

        # We keep negative values to encode for a zero movie. Therefore they can exist.
        # if np.any(orientation_list == -1): raise ValueError("Unassigned condition")
        # if np.any(temporal_frequency_list == -1): raise ValueError("Unassigned condition")

        # recreate movies, and make then at the "dt" sampling frequency
        movie_dict = {}
        frame_rate_dict = {}

        for condition_id in stimulus_spec_dict.keys():
            spec_dict = stimulus_spec_dict[condition_id]
            movie, frame_rate = self.create_drifting_gratings_movie(size=[128, 128],**spec_dict)

            frame_rate_dict[condition_id] = frame_rate
            m = np.array(movie, dtype=np.uint8).mean(-1) # gray scale
            m = np.stack([resize(I, (64,64)) for I in m], 0) # resize
            movie_dict[condition_id] = m

        # normalize:
        mean = np.mean([m.mean() for m in movie_dict.values()])
        std = np.sqrt(sum(m.var() for m in movie_dict.values()) / len(movie_dict.values()))
        for condition_id in movie_dict.keys():
            movie_dict[condition_id] = (movie_dict[condition_id] - mean) / std

        # In agreement the frame indices will be of the form: condition_idx * n_time + time
        return stimulus_spec_dict, movie_dict, frame_rate_dict


    def create_drifting_gratings_movie(self, orientation, contrast, duration,
                                       temporal_frequency=2,
                                       spatial_frequency=0.04,
                                       size=[250, 250],
                                       phase=[14037.96666667, 14037.96666667]):

        from psychopy.clock import Clock
        from psychopy.visual import Window
        from psychopy.visual.grating import GratingStim

        size = np.array(size, dtype=np.float).astype(np.int)

        if orientation < 0 or temporal_frequency <0:
            print("Warning: creating blank movie.")
            shp = [1, size[1], size[0], 3]
            movie = np.zeros(shp, dtype=int)
            frame_rate = 0.0
            return movie, frame_rate

        win = Window(size)
        size = max(size) * 2 # avoid that the movie if cropped by the window boundary

        frame_rate = win.getActualFrameRate(nIdentical=60, nMaxFrames=100, nWarmUpFrames=10, threshold=1)
        for k in range(100):
            if frame_rate is not None: break
            else: frame_rate = win.getActualFrameRate(nIdentical=60, nMaxFrames=100, nWarmUpFrames=10, threshold=1)

        grat_stim = GratingStim(win=win, contrast=contrast, tex="sin", units="pix",
                                       size=size, sf=spatial_frequency, ori=orientation, phase=phase)
        clock = Clock()

        frame_list = []

        while (clock.getTime() < duration):
            t = clock.getTime()
            phase = t * temporal_frequency
            grat_stim.setPhase(phase)
            grat_stim.draw()
            pic = win.getMovieFrame()
            frame_list.append(np.array(pic))
            win.flip()
        win.close()

        movie = np.stack(frame_list)
        movie = np.array(movie, dtype=np.uint8)
        return (movie, frame_rate)

    def get_stimulus_template(self):
        file_name = "stimulus_template"
        fn = lambda: self.download_or_create_stimulus_template()
        return self.compute_and_save_or_load_from_cache(file_name, fn, cache_specs=[self.stimulus])


if __name__ == "__main__":
    from allen_dataset import get_session_id_list
    session_id_list = get_session_id_list()
    session_idx = 1
    AtoT = AllenToTensor(stimulus="drifting_gratings", session_id=session_id_list[1])
    raster = AtoT.get_raster_plot()
    print("raster has shape {} and {} spikes".format(raster.shape, raster.sum()))

    unit_ids, masked_visp_raster_plot = AtoT.get_raster_plot_of_selected_units("VISp")
    print("VISp raster plot has shape {} and {} spikes.".format(masked_visp_raster_plot.shape,
                                                                masked_visp_raster_plot.sum()))

    for i in range(10):
        n_spikes = masked_visp_raster_plot[:, :, i].sum(axis=(0, 1))
        print("{}. unit {} got {} spikes".format(i, unit_ids[i], n_spikes))

    template = AtoT.get_stimulus_template()
    print("The movie template has shape {} (dt is {}s)".format(template.shape, AtoT.dt))

    del AtoT
