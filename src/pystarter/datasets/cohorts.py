# import ..utils
# import sys
# sys.path.append("/home/local/other_project")
# import pystarter.models.encoders

import os, pandas as pd
from dataclasses import dataclass
from typing import TypedDict, OrderedDict
from datetime import timedelta
from dateutil import relativedelta
from functools import cached_property

import pystarter.definitions as D
from pystarter.utils import read_tabular

class CachedCohort():
    def __init__(
        self,
        fpath: dict,
        fargs: dict,
        cache: str=None,
        scan_cache: str=None,
        img_dir: str=None,
    ):
        """
        fpath: dict - file path of raw data
        fargs: dict - keyword args to pass to pd.read_csv and pd.read_excel
        """
        self.cache = cache
        self.scan_cache = scan_cache
        self.imgs = img_dir
        self.data = {}
        if isinstance(fpath, str):
            self.data['main'] = read_tabular(fpath, fargs)
        else:
            for key, _ in fpath.items():
                self.data[key] = read_tabular(fpath[key], fargs[key])

        self.transforms = OrderedDict()

        # preprocess logic
        self.transforms["id"] = lambda: self.data["main"]["patient_id"]
        self.transforms["name"] = lambda: self.data["main"]["last_name"] + "," + self.data["main"]["first_name"]
        self.transforms["age"] = lambda: self.data["main"].apply(lambda x: self.age(x))

    def age(self, x):
        # x is a single row
        birthday = pd.to_datetime(x["birthday"])
        obs_date = pd.to_datetime(x["trial_start"]) + pd.to_timedelta(x["fup_days"], unit='D')
        delta = relativedelta(obs_date, birthday)
        return int(delta.years)

    @property
    def df(self):
        if self._df is None: 
            if self.cache is None:
                for key, func in self.transforms.items():
                    self._df[key] = func()
            else:
                self._df = read_tabular(self.cache)
        return self._df
    
    @cached_property
    def scan_df(self):
        """assumes file structure of [img_dir]/[subject id]/[session date]/[file.dcm]"""
        if self.scan_cache is None:
            scanlist = []
            for id in os.scandir(self.img_dir):
                for session in os.scandir(id):
                    for scan in os.scandir(session):
                        self._scanlist.append([id.name, session.name, f.path])
            
            return pd.DataFrame(scanlist, columns=['id', 'scandate', 'fpath'])
        else:
            return read_tabular(self.scan_cache)


class NLSTCohort(CachedCohort):
    def __init__(
            self, 
            fpath={'main': 'raw_nlst.csv'}, 
            fargs={'dtype': {'patient_id': str}}, 
            **kwargs):
        super().__init__(fpath, fargs, **kwargs)




class NLST_CohortWrapper():
    def __init__(
        self,
        init_cache=lambda: NLSTCohort(
            cache=os.path.join(D.DATA, "nlst.csv"),
            scan_cache=os.path.join(D.DATA, "nlst_scan.csv"),
        ),
        test="/home/local/nlst_test.csv",
    ):
        self.init_cache = init_cache
        self.test = read_tabular(test)

    @cached_property
    def cohort(self):
        return self.init_cache()

    @cached_property
    def train_set(self):
        df = self.cohort.df
        return df[~df['id'].isin(self.test['id'])]
    
    
    def train_oldage(self):
        return self.train_set[self.train_set['age']>75]


if __name__ == "__main__":
    nlst = NLSTCohort(
        fpath={'main': 'raw_nlst.csv'},
        fargs = {'dtype': {'patient_id': str}}
    )
    nlst.df.to_csv(os.path.join(D.DATA, "nlst.csv"))
    nlst.scan_df.to_csv(os.path.join(D.DATA, "nlst_scan.csv"))