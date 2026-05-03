import os
import torch
import time
import ml_collections


def get_DSTransNet_Student_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 60  # KV_size = 32+64+128+256   # Part2、8+16+32+64=120    60   120
    config.transformer.num_heads = 4
    config.transformer.num_layers = 1                  # Part3、1
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channels = 4  # base channel of U-Net     # Part1、   4  8
    config.n_classes = 1

    # ********** unused **********
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    return config


def get_DSTransNet_Student_config_v5():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 120  # KV_size = 32+64+128+256   # Part2、8+16+32+64=120    60   120
    config.transformer.num_heads = 4
    config.transformer.num_layers = 1                  # Part3、1
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channels = 8  # base channel of U-Net     # Part1、   4  8
    config.n_classes = 1

    # ********** unused **********
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    return config


def get_DSTransNet_Student_config_v34():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 240  # KV_size = 32+64+128+256   # Part2、8+16+32+64=120    60   120
    config.transformer.num_heads = 4
    config.transformer.num_layers = 1                  # Part3、1
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channels = 16  # base channel of U-Net     # Part1、   4  8
    config.n_classes = 1

    # ********** unused **********
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    return config
