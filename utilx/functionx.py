#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def accumulate_char_offset(words, add_between_space=0):
    r"""
    Accumulate the char offset of words with the given space between words.
    :param words:
    :param add_between_space:
    :return: the offset list for each word, and each offset is a tuple (included, excluded)
    """
    offsets = []
    cur_offset = 0
    for word in words:
        end_offset = cur_offset+len(word)
        offsets.append((cur_offset, end_offset))
        cur_offset = end_offset + add_between_space
    return offsets


def add_dict_set(dictionary, key, value):
    r"""
    Add a key into dictionary with a given value.
    :param dictionary:
    :param key:
    :param value:
    :return:
    """
    if key in dictionary:
        dictionary[key].add(value)
    else:
        dictionary[key] = set([value])


def set_grad(model, value: bool):
    r"""
    Set grad to frozen or train model parameters.
    :param model:
    :param value:
    :return:
    """
    for param in model.parameters():
        param.requires_grad = value
