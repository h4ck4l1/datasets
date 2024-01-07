#!/usr/locaol/bin/bash

from __future__ import absolute_import,division
import os,sys,time,re,math,contextlib2,gc
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,wait
from multiprocessing import Process
import arnie
from arnie.mfe import mfe
from arnie.bpps import bpps
import tensorflow as tf
from tensorflow import keras
from tensorflow.io import FixedLenFeature,VarLenFeature,TFRecordWriter,serialize_tensor
from tensorflow.train import Example,Features,Feature,BytesList
from tensorflow.data import Dataset,TFRecordDataset
from zipfile import ZipFile
tf.get_logger().setLevel("ERROR")
from object_detection.dataset_tools import tf_record_creation_util
from tqdm import tqdm

train_df = pd.read_csv("/content/train_data.csv/train_data.csv")
all_cols = train_df.columns
reactivity_columns = all_cols[all_cols.map(lambda x: "reactivity_0" in x)].tolist()
reactivity_error_columns = all_cols[all_cols.map(lambda x: "reactivity_error_0" in x)].tolist()

train_df["reactivity"] = train_df[reactivity_columns].values.tolist()
train_df["reactivity_error"] = train_df[reactivity_error_columns].values.tolist()
train_df = train_df.drop(columns=reactivity_columns+reactivity_error_columns)

all_cols = train_df.columns
seq_id = all_cols.get_loc("sequence_id")
seq = all_cols.get_loc("sequence")
dataset_name = all_cols.get_loc("dataset_name")
sn_filter = all_cols.get_loc("SN_filter")
reads = all_cols.get_loc("reads")
signal_to_noise = all_cols.get_loc("signal_to_noise")
reactivity = all_cols.get_loc("reactivity")
reactivity_error = all_cols.get_loc("reactivity_error")

df_2a3 = train_df[train_df.experiment_type == "2A3_MaP"].reset_index(drop=True)
df_dms = train_df[train_df.experiment_type == "DMS_MaP"].reset_index(drop=True)

del train_df
gc.collect()

seq_map = dict(A=0,C=2,G=3,U=4)
bracket_map = {"(":0,")":1,"[":2,"]":3,"{":4,"}":5,"<":6,">":7,".":8}

def SequenceId(val):
    val = serialize_tensor(tf.constant(val,dtype=tf.string)).numpy()
    return Feature(bytes_list=BytesList(value=[val]))

def Sequence(val):
    val = serialize_tensor(tf.constant([seq_map[_] for _ in val],dtype=tf.float32)).numpy()
    return Feature(bytes_list=BytesList(value=[val]))

def Reads(val):
    val = serialize_tensor(tf.constant(val,dtype=tf.float32)).numpy()
    return Feature(bytes_list=BytesList(value=[val]))

def DatasetName(val):
    val = serialize_tensor(tf.constant(val,dtype=tf.string)).numpy()
    return Feature(bytes_list=BytesList(value=[val]))

def SNFilter(val):
    val = serialize_tensor(tf.constant(val,dtype=tf.float32)).numpy()
    return Feature(bytes_list=BytesList(value=[val]))

def SignalToNoise(val):
    val = serialize_tensor(tf.constant(val,dtype=tf.float32)).numpy()
    return Feature(bytes_list=BytesList(value=[val]))

def Reactivity(val):
    val = serialize_tensor(tf.constant(val,dtype=tf.float32)).numpy()
    return Feature(bytes_list=BytesList(value=[val]))

def ReactivityError(val):
    val = serialize_tensor(tf.constant(val,dtype=tf.float32)).numpy()
    return Feature(bytes_list=BytesList(value=[val]))

def BracketSequence(val):
    val = serialize_tensor(tf.constant([bracket_map[_] for _ in mfe(val,"eternafold")],tf.float32)).numpy()
    return Feature(bytes_list=BytesList(value=[val]))

def BPPMatrix(val):
    val = serialize_tensor(tf.constant(bpps(val,"eternafold"),tf.float32)).numpy()
    return Feature(bytes_list=BytesList(value=[val]))

def Length(val):
    val = serialize_tensor(tf.constant(val,tf.float32)).numpy()
    return Feature(bytes_list=BytesList(value=[val]))


