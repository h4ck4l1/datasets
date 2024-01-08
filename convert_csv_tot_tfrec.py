#!/usr/local/bin/conda

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
    val = serialize_tensor(tf.constant(len(val),tf.float32)).numpy()
    return Feature(bytes_list=BytesList(value=[val]))

def serialize_row(row_2a3,row_dms):

    assert (row_2a3[seq_id] == row_dms[seq_id]) and (row_2a3[seq] == row_dms[seq])

    feature = dict(
        seq_id = SequenceId(row_2a3[seq_id]),
        seq = Sequence(row_2a3[seq]),
        reads_2a3 = Reads(row_2a3[reads]),
        reads_dms = Reads(row_dms[reads]),
        sn_filter_2a3 = SNFilter(row_2a3[sn_filter]),
        sn_filter_dms = SNFilter(row_dms[sn_filter]),
        dataset_name_2a3 = DatasetName(row_2a3[dataset_name]),
        dataset_name_dms = DatasetName(row_dms[dataset_name]),
        signal_to_noise_2a3 = SignalToNoise(row_2a3[signal_to_noise]),
        signal_to_noise_dms = SignalToNoise(row_dms[signal_to_noise]),
        reactivity_2a3 = Reactivity(row_2a3[reactivity]),
        reactivity_dms = Reactivity(row_dms[reactivity]),
        reactivity_error_2a3 = ReactivityError(row_2a3[reactivity_error]),
        reactivity_error_dms = ReactivityError(row_dms[reactivity_error]),
        length = Length(row_2a3[seq]),
        bpp_matrix = BPPMatrix(row_2a3[seq]),
        bracket_seq = BracketSequence(row_2a3[seq])
    )

    return Example(features=Features(feature=feature))

def sharded_tfrecords(exitstack,base_path,num_shards):
    output_tfrecord_filenames = [f"{base_path}/train_{idx}" for idx in range(num_shards)]
    return [exitstack.enter_context(TFRecordWriter(file_name,options="GZIP")) for file_name in output_tfrecord_filenames]

l = len(df_2a3)
num_shards = 150
output_filebase = "content/tf_records"

def process_row(index,row_2a3,row_dms,output_tfrecords):
    tf_example = serialize_row(row_2a3,row_dms)
    shard_index = num_shards % index
    output_tfrecords[shard_index].write(tf_example.SerializeToString()) 


def paralleliz_tfrecord_creation(df_2a3,df_dms,output_tfrecords,num_shards):
    with ProcessPoolExecutor() as executor:
        futures = []
        for index, (row_2a3,row_dms) in enumerate(tqdm(zip(df_2a3.itertuples(index=False),df_dms.itertuples(index=False)))):
            futures.append(executor.submit(process_row,index,row_2a3,row_dms,output_tfrecords))

        wait(futures)



if __name__ == "__main__":
    with contextlib2.ExitStack() as exit_stack:
        output_tfrecords = sharded_tfrecords(exit_stack,output_filebase,num_shards)
        paralleliz_tfrecord_creation(df_2a3,df_dms,output_tfrecords,num_shards)
