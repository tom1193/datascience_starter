import os, pandas as pd
from dataclasses import dataclass
from typing import TypedDict, OrderedDict
from datetime import timedelta
from dateutil import relativedelta
from functools import cached_property

import pystarter.definitions as D
from pystarter.utils import read_tabular

class Cohort():
    def __init__(
        self,
        fpath: dict,
        fargs: dict,
        cache: str=None,
        img_dir: str=None,
    ):
        self.cache = cache
        self.imgs = img_dir
        self.data = {}
        if isinstance(fpath, str):
            self.data['main'] = read_tabular(fpath, fargs)
        else:
            for key, _ in fpath.items():
                self.data[key] = read_tabular(fpath[key], fargs[key])

        self.transforms = OrderedDict()

        # preprocess logic

    @property
    def df(self):
        if self._df is None: # lazy property, only computes the first time its called
            # load dataframe from cache
            if self.cache is None:
                # preprocess cohort
                for key, func in self.transforms.items():
                    self._df[key] = func()
            else:
                self._df = read_tabular(self.cache)
        return self._df
    
    @cached_property
    def scan_df(self):
        """assumes file structure of [img_dir]/[subject id]/[file.dcm]"""

        scanlist = []
        for id in os.scandir(self.img_dir):
            for f in os.scandir(id):
                self._scanlist.append([id.name, f.path])
        
        return pd.DataFrame(scanlist, columns=['id', 'fpath'])


class NLSTCohort(Cohort):
    def __init__(self, fpath: dict, fargs: dict, **kwargs):
        super().__init__(fpath, fargs, **kwargs)

        # preprocess
        self.transforms["id"] = lambda: self.data["main"]["patient_id"]
        self.transforms["name"] = lambda: self.data["main"]["last_name"] + "," + self.data["main"]["first_name"]
        self.transforms["age"] = lambda: self.data["main"].apply(lambda x: self.age(x))

    def age(self, x):
        # x is a single row
        birthday = pd.to_datetime(x["birthday"])
        obs_date = pd.to_datetime(x["trial_start"]) + pd.to_timedelta(x["fup_days"], unit='D')
        delta = relativedelta(obs_date, birthday)
        return int(delta.years)

class NLST



if __name__ == "__main__":
    nlst = NLSTCohort(
        fpath={'main': 'raw_nlst.csv'},
        fargs = {'dtype': {'patient_id': str}}
    )
    nlst.df.to_csv(os.path.join(D.DATA, "nlst.csv"))
    nlst.scan_df.to_csv(os.path.join(D.DATA, "nlst_scan.csv"))