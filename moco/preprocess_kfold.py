import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import mne, os
from mne.datasets.sleep_physionet.age import fetch_data

from braindecode.datautil.preprocess import preprocess, Preprocessor
from braindecode.datautil.windowers import create_windows_from_events
from braindecode.datautil.preprocess import zscore
from braindecode.datasets import BaseConcatDataset, BaseDataset

from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from sklearn.utils import check_random_state

PATH = '/scratch/sslkfold/'
DATA_PATH = '/scratch/'
os.makedirs(PATH, exist_ok=True)

# Params
BATCH_SIZE = 1
POS_MIN = 1
NEG_MIN = 15
EPOCH_LEN = 7
NUM_SAMPLES = 500
SUBJECTS = np.arange(83)
RECORDINGS = [1, 2]


##################################################################################################

random_state = 1234
n_jobs = 1
sfreq = 100
high_cut_hz = 30

window_size_s = 30
sfreq = 100
window_size_samples = window_size_s * sfreq



class SleepPhysionet(BaseConcatDataset):
    def __init__(
        self,
        subject_ids=None,
        recording_ids=None,
        preload=False,
        load_eeg_only=True,
        crop_wake_mins=30,
        crop=None,
    ):
        if subject_ids is None:
            subject_ids = range(83)
        if recording_ids is None:
            recording_ids = [1, 2]

        paths = fetch_data(
            subject_ids,
            recording=recording_ids,
            on_missing="warn",
            path= DATA_PATH,
        )

        all_base_ds = list()
        for p in paths:
            raw, desc = self._load_raw(
                p[0],
                p[1],
                preload=preload,
                load_eeg_only=load_eeg_only,
                crop_wake_mins=crop_wake_mins,
                crop=crop
            )
            base_ds = BaseDataset(raw, desc)
            all_base_ds.append(base_ds)
        super().__init__(all_base_ds)

    @staticmethod
    def _load_raw(
        raw_fname,
        ann_fname,
        preload,
        load_eeg_only=True,
        crop_wake_mins=False,
        crop=None,
    ):
        ch_mapping = {
            "EOG horizontal": "eog",
            "Resp oro-nasal": "misc",
            "EMG submental": "misc",
            "Temp rectal": "misc",
            "Event marker": "misc",
        }
        exclude = list(ch_mapping.keys()) if load_eeg_only else ()

        raw = mne.io.read_raw_edf(raw_fname, preload=preload, exclude=exclude)
        annots = mne.read_annotations(ann_fname)
        raw.set_annotations(annots, emit_warning=False)

        if crop_wake_mins > 0:
            # Find first and last sleep stages
            mask = [x[-1] in ["1", "2", "3", "4", "R"] for x in annots.description]
            sleep_event_inds = np.where(mask)[0]

            # Crop raw
            tmin = annots[int(sleep_event_inds[0])]["onset"] - crop_wake_mins * 60
            tmax = annots[int(sleep_event_inds[-1])]["onset"] + crop_wake_mins * 60
            raw.crop(tmin=max(tmin, raw.times[0]), tmax=min(tmax, raw.times[-1]))

        # Rename EEG channels
        ch_names = {i: i.replace("EEG ", "") for i in raw.ch_names if "EEG" in i}
        raw.rename_channels(ch_names)

        if not load_eeg_only:
            raw.set_channel_types(ch_mapping)

        if crop is not None:
            raw.crop(*crop)

        basename = os.path.basename(raw_fname)
        subj_nb = int(basename[3:5])
        sess_nb = int(basename[5])
        desc = pd.Series({"subject": subj_nb, "recording": sess_nb}, name="")

        return raw, desc


dataset = SleepPhysionet(
    subject_ids=SUBJECTS, recording_ids=RECORDINGS, crop_wake_mins=30
)


preprocessors = [
    Preprocessor(lambda x: x * 1e6),
    Preprocessor("filter", l_freq=None, h_freq=high_cut_hz, n_jobs=n_jobs),
]

# Transform the data
preprocess(dataset, preprocessors)


mapping = {  # We merge stages 3 and 4 following AASM standards.
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}

windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=window_size_samples,
    window_stride_samples=window_size_samples,
    preload= True,
    mapping=mapping,
)


preprocess(windows_dataset, [Preprocessor(zscore)])


###################################################################################################################################
""" Subject sampling """

rng = np.random.RandomState(1234)

NUM_WORKERS = 0 if n_jobs <= 1 else n_jobs
PERSIST = False if NUM_WORKERS <= 1 else True


subjects = np.unique(windows_dataset.description["subject"])
sub_pretext = rng.choice(subjects, 58, replace=False)
sub_test = sorted(list(set(subjects) - set(sub_pretext)))


print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(f"Pretext: {sub_pretext} \n")
print(f"Test: {sub_test} \n")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


#######################################################################################################################################


class PretextDataset(BaseConcatDataset):
    """BaseConcatDataset with __getitem__ that expects 2 indices and a target."""

    def __init__(self, list_of_ds, epoch_len=7):
        super().__init__(list_of_ds)
        self.epoch_len = epoch_len

    def __getitem__(self, index):

        data = super().__getitem__(index)[0] # Get the data
        return data


class TuneDataset(BaseConcatDataset):
    """BaseConcatDataset for train and test"""

    def __init__(self, list_of_ds):
        super().__init__(list_of_ds)

    def __getitem__(self, index):

        X = super().__getitem__(index)[0]
        y = super().__getitem__(index)[1]

        return X, y    
    
######################################################################################################################


PRETEXT_PATH = os.path.join(PATH, "pretext")
TEST_PATH = os.path.join(PATH, "test")

if not os.path.exists(PRETEXT_PATH): os.mkdir(PRETEXT_PATH)
if not os.path.exists(TEST_PATH): os.mkdir(TEST_PATH)

splitted = dict()

splitted["pretext"] = PretextDataset(
    [ds for ds in windows_dataset.datasets if ds.description["subject"] in sub_pretext],
    epoch_len = EPOCH_LEN
)

splitted["test"] = [ds for ds in windows_dataset.datasets if ds.description["subject"] in sub_test]

for sub in splitted["test"]:
    temp_path = os.path.join(TEST_PATH, str(sub.description["subject"]) + str(sub.description["recording"])+'.npz')
    np.savez(temp_path, **sub.__dict__)

########################################################################################################################

print(f'Number of pretext subjects: {len(splitted["pretext"].datasets)}')

# Dataloader
pretext_loader = DataLoader(
    splitted["pretext"],
    batch_size=BATCH_SIZE
)

for i, arr in tqdm(enumerate(pretext_loader), desc = 'pretext'):
    temp_path = os.path.join(PRETEXT_PATH, str(i) + '.npz')
    np.savez(temp_path, data = arr.numpy().squeeze(0))