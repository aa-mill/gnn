#!/usr/bin/env python3
import tensorflow as tf
import functools
import h5py
import json
import os


def _parse(proto, meta):
  """Parses a trajectory from tf.Example."""
  feature_lists = {k: tf.io.VarLenFeature(tf.string)
                   for k in meta['field_names']}
  features = tf.io.parse_single_example(proto, feature_lists)
  out = {}
  for key, field in meta['features'].items():
    data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
    data = tf.reshape(data, field['shape'])
    if field['type'] == 'static':
      data = tf.tile(data, [meta['trajectory_length'], 1, 1])
    elif field['type'] == 'dynamic_varlen':
      length = tf.io.decode_raw(features['length_'+key].values, tf.int32)
      length = tf.reshape(length, [-1])
      data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
    elif field['type'] != 'dynamic':
      raise ValueError('invalid data format')
    out[key] = data
  return out


def load_dataset(path, split):
  """Load dataset."""
  with open(os.path.join(path, 'meta.json'), 'r') as fp:
    meta = json.loads(fp.read())
  ds = tf.data.TFRecordDataset(os.path.join(path, split+'.tfrecord'))
  ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
  ds = ds.prefetch(1)
  return ds


def add_targets(ds, fields, add_history):
  """Adds target and optionally history fields to dataframe."""
  def fn(trajectory):
    out = {}
    for key, val in trajectory.items():
      out[key] = val[1:-1]
      if key in fields:
        if add_history:
          out['prev|'+key] = val[0:-2]
        out['target|'+key] = val[2:]
    return out
  return ds.map(fn, num_parallel_calls=8)


def to_hdf5(load_dir, save_dir):
    for split in ['train', 'valid', 'test']:
        ds = load_dataset(load_dir, split)
        ds = add_targets(ds, ['velocity'], False)
        with h5py.File(os.path.join(save_dir, split + '.h5'), 'w') as file:
            for idx, slice in ds.enumerate().as_numpy_iterator():
                group = file.create_group(str(idx))
                for key, val in slice.items():
                    group.create_dataset(key, data=val)


def main():
    load_dir = 'data/cylinder_flow'
    save_dir = 'data/cylinder_flow'
    to_hdf5(load_dir, save_dir)


if __name__ == '__main__':
    main()