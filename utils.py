#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:10:21 2019

@author: henric
"""
import json
import numpy as np
import pandas as pd
import os

def save_history(history, model_name, mode='w', path='histories', force = False):
    filename = os.path.join(path, model_name)
    h = {k:np.array(v).tolist() for k,v in history.items()}
    if mode == 'a':
        try:
            with open('{}.json'.format(filename), 'r') as fp:
                old_data = json.load(fp)
        except:
            print("No file to append to. Creating...")
            return save_history(history, model_name, mode='w', path=path, force = force)
        keys = set(h.keys()).symmetric_difference(old_data.keys())
        if len(keys)>0:
            print("Warning: the number of keys do not correspond.")
            if not force:
                print("No action taken.")
                return
        h = {k:old_data[k]+h[k] for k in old_data}
    with open('{}.json'.format(filename), 'w') as fp:
        json.dump(h, fp)
        
def load_history(model_name, path='histories'):
    filename = os.path.join(path, model_name)
    with open('{}.json'.format(filename), 'r') as fp:
        data = json.load(fp)
    return data
    
def history_curation(history):
    # trnasform history and prepare to save it to a file
    df = pd.DataFrame(history)
    df.index.name = 'epoch'
    return df

def load_histories_to_df(path='histories'):
    histories_dict = {}

    history_filenames = os.listdir(path)
    for filename in history_filenames:
        histories_dict[filename[:-5]] = load_history(filename[:-5], path)
        
    metrics_dict = {k: pd.DataFrame(v) for k,v in histories_dict.items()}
    metrics_df = pd.concat(metrics_dict, axis=1)
    metrics_df.index.name = 'epoch'
    return metrics_df
    