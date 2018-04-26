# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A factory-pattern class which returns classification image/label pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import visages

datasets_map = {
    'visages': visages
}


def get_dataset(name, split_name, chemin_dataset ,tfrecord_file, chemin_liste_labels, reader=None):
    """Given a dataset name and a split_name returns a Dataset.

  Args:
    name: String, the name of the dataset.
    split_name: A train/test split name.
    csv_file: The path to csv file containing association image-label
    reader: The subclass of tf.ReaderBase. If left as `None`, then the default
      reader defined by each dataset is used.
    chemin_liste_labels: the path to the complete list of labels

  Returns:
    A `Dataset` class.

  Raises:
    ValueError: If the dataset `name` is unknown.
  """
    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % name)
    return datasets_map[name].get_split(
        split_name,
        chemin_dataset,
        tfrecord_file,
        chemin_liste_labels,
        reader)
