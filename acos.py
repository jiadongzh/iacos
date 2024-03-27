#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import copy
import itertools
import numpy as np
import torch
from torch import optim
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from typing import List, Any
from dataclasses import dataclass
from tokenizers import Encoding
from transformers import PreTrainedTokenizerFast
from utilx.functionx import accumulate_char_offset, add_dict_set, set_grad
from modelx.rnnx import LSTMCRF

IntList = List[int]
IntListList = List[IntList]
implicit_aspect_token = '[IMPLICIT_ASPECT]'
implicit_opinion_token = '[IMPLICIT_OPINION]'

def test_lstm_crf():
    start_tag = "<START>"
    stop_tag = "<STOP>"
    embedding_dim = 5
    hidden_dim = 4

    def prepare_sequence(seq, to_ix):
        idx = [to_ix[w] for w in seq]
        return torch.tensor(idx, dtype=torch.long)

    # Make up some training data
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]

    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {"B": 0, "I": 1, "O": 2, start_tag: 3, stop_tag: 4}

    model = LSTMCRF(len(word_to_ix), tag_to_ix, embedding_dim, hidden_dim, start_tag, stop_tag)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Check predictions before training
    with torch.no_grad():
        pre_check_sent = prepare_sequence(training_data[0][0], word_to_ix)
        print(model(pre_check_sent))

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(30):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()

    # Check predictions after training
    with torch.no_grad():
        pre_check_sent = prepare_sequence(training_data[0][0], word_to_ix)
        print(model(pre_check_sent))


def acos_data_to_ddi_format(input_file, add_implicit_tokens=False, output_file=None):
    r"""
    Convert to DDI json format.
    :param input_file: the ACOS data file,
        see https://github.com/NUSTM/ACOS/tree/main/data
    :param output_file: the DDI data format,
        see https://github.com/LightTag/DDICorpus
    :param add_implicit_tokens: e.g., ['[IMPLICIT_ASPECT]', '[IMPLICIT_OPINION]']
    :return: the DDI data format
    """

    def get_phrase_char_offset(word_char_offsets,
                               phrase_word_start,
                               phrase_word_end):
        r"""
        Get the char offset of a phrase according the char offsets of words in the phrase.
        :param word_char_offsets:
        :param phrase_word_start: included
        :param phrase_word_end: excluded
        :return:  a char offset tuple (included, excluded)
        """
        char_start = word_char_offsets[phrase_word_start][0]
        char_end = word_char_offsets[phrase_word_end - 1][1]
        return char_start, char_end

    results = []
    with open(input_file, 'r', encoding='utf-8') as inputs:
        for line in inputs.readlines():
            item = {}
            line = line.strip()
            segments = line.split('\t')
            item['content'] = segments[0]
            if add_implicit_tokens:
                item['content'] += ' ' + implicit_aspect_token + ' ' + implicit_opinion_token
            words = segments[0].split(' ')
            offsets = accumulate_char_offset(words, 1)
            annotations = []
            annotation_pairs = []
            for anno in segments[1:]:
                acoses = anno.split(' ')
                assert len(acoses) == 4
                aspect_opinion_pair = {}
                # -1 for implicit aspects
                aspect_start, aspect_end = list(map(int, acoses[0].split(',')))
                if aspect_start >= 0:
                    start, end = get_phrase_char_offset(offsets, aspect_start, aspect_end)
                    annotations.append({'start': start, 'end': end, 'tag': 'Aspect'})
                    aspect_opinion_pair['aspect_start'] = start
                    aspect_opinion_pair['aspect_end'] = end
                # -1 for implicit opinions
                opinion_start, opinion_end = list(map(int, acoses[3].split(',')))
                if opinion_start >= 0:
                    start, end = get_phrase_char_offset(offsets, opinion_start, opinion_end)
                    annotations.append({'start': start, 'end': end, 'tag': 'Opinion'})
                    aspect_opinion_pair['opinion_start'] = start
                    aspect_opinion_pair['opinion_end'] = end
                aspect_opinion_pair['aspect_category'] = acoses[1]
                aspect_opinion_pair['opinion_sentiment'] = int(acoses[2])
                annotation_pairs.append(aspect_opinion_pair)
            item['annotations'] = annotations
            item['annotation_pairs'] = annotation_pairs
            results.append(item)
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as outputs:
            json.dump(results, outputs)
    return results


def char_idx_to_token_idx(tokenized: Encoding, start, end):
    r"""
    Convert char indexing to token indexing.
    :param tokenized:
    :param start:
    :param end:
    :return:
    """
    token_ix_set = set()
    for char_ix in range(start, end):
        token_ix = tokenized.char_to_token(char_ix)
        # white spaces have no token and will return None
        if token_ix is not None:
            token_ix_set.add(token_ix)
        ixs = sorted(token_ix_set)
        # continuous indices
        assert len(ixs) == ixs[-1] - ixs[0] + 1
    return ixs


class BIOLabelSet:
    def __init__(self, labels: List[str]):
        self.label_to_id = dict()
        self.id_to_label = dict()
        self.label_to_id["O"] = 0
        self.id_to_label[0] = "O"
        # Writing BILU will give us incremental ids for the labels
        for _num, (label, s) in enumerate(itertools.product(labels, "BILU")):
            num = _num + 1  # skip 0
            bilu_label = f"{s}-{label}"
            self.label_to_id[bilu_label] = num
            self.id_to_label[num] = bilu_label

    @classmethod
    def align_annotations_with_bio(cls, tokenized: Encoding, annotations,
                                   add_implicit_tokens=False):
        r"""
        Align tokens' labeling to BILOU scheme: B for beginning tokens,
            I for inside tokens, L for last tokens, and U for single tokens.
            And convert char indexing to token indexing; note that a tokenizer may
            add special tokens, e.g., [CLS], [SEP], etc.
        :param tokenized:
        :param annotations:
        :param add_implicit_tokens:
        :return: tokens with aligned BILOU tags.
        """
        # Make a list to store our labels the same length as our tokens
        length = len(tokenized.tokens)
        aligned_labels = ['O'] * length
        if add_implicit_tokens:
            # implicit aspect
            aligned_labels[length - 3] = 'U-Aspect'
            # implicit opinion
            aligned_labels[length - 2] = 'U-Opinion'
            # aligned_labels[length - 1] for [SEP]

        for anno in annotations:
            # A set that stores the token indices of the annotation
            token_list = char_idx_to_token_idx(tokenized, anno['start'], anno['end'])
            if len(token_list) == 1:
                # If there is only one token
                token_ix = token_list[0]
                # This annotation spans one token so is prefixed with U for unique
                prefix = 'U'
                aligned_labels[token_ix] = f"{prefix}-{anno['tag']}"
            else:
                last_token_in_anno_ix = len(token_list) - 1
                for ix, token_ix in enumerate(token_list):
                    if ix == 0:
                        prefix = 'B'
                    elif ix == last_token_in_anno_ix:
                        # It's the last token
                        prefix = 'L'
                    else:
                        # We're inside a multi token annotation
                        prefix = 'I'
                    aligned_labels[token_ix] = f"{prefix}-{anno['tag']}"
        return aligned_labels

    def get_aligned_label_ids_from_annotations(self, tokenized: Encoding,
                                               annotations,
                                               add_implicit_tokens=False):
        r"""
        Convert char indexing in annotations to token indexing of tokenized_text.
        Note that a tokenizer may add special tokens, e.g., [CLS], [SEP], etc.
        :param tokenized: tokenized encoding.
        :param annotations: labeled data.
        :param add_implicit_tokens:
        :return: tokens with aligned label ids.
        """
        raw_labels = self.align_annotations_with_bio(
            tokenized, annotations, add_implicit_tokens)
        label_list = list(map(self.label_to_id.get, raw_labels))
        # -100 is a special token for padding of labels
        # see torch.nn.NLLLoss(ignore_index=-100)
        # set the label to -100 for [CLS]
        if tokenized.special_tokens_mask[0]:
            label_list[0] = -100
        # set the label to -100 for [SEP]
        last = len(tokenized.tokens) - 1
        if tokenized.special_tokens_mask[last]:
            label_list[last] = -100
        return label_list


class CategorySentimentLabelSet:
    def __init__(self, category_names: List[str]):
        r"""
        Note that the sentiment has been tagged with 0, 1 and 2
        which represent negative, neutral and positive, respectively.
        :param category_names: A list of category names.
        """
        self.label_to_id = dict()
        self.id_to_label = dict()
        for ix, label in enumerate(category_names):
            self.label_to_id[label] = ix
            self.id_to_label[ix] = label

    def get_category_sentiment_pairs(self,
                                     tokenized: Encoding,
                                     annotation_pairs,
                                     text,
                                     neg_sampling=True):
        r"""
        Get category and sentiment spans, each with a unique label.
        :param tokenized:
        :param annotation_pairs:
        :param text:
        :param neg_sampling:
        :return:
        """
        category_idx = []
        category_labels = []
        sentiment_idx = []
        sentiment_labels = []
        cat_sent_idx = []
        cat_sent_labels = []
        # for testing: [([aspect idx], [opinion idx], category, sentiment), ...]
        pair_labels = []
        # Removing redundant aspects or opinions,
        # which affect negative sampling
        aspect_set = set()
        opinion_set = set()
        pair_set = set()
        word_num = len(tokenized)
        implicit_both_count = 0
        # for statistics of multi-labels
        cat_dict = {}
        sent_dict = {}
        for anno_pair in annotation_pairs:
            aspect_list = []
            opinion_list = []

            if 'aspect_start' in anno_pair:
                aspect_list = char_idx_to_token_idx(tokenized,
                                                    anno_pair['aspect_start'],
                                                    anno_pair['aspect_end'])

            if 'opinion_start' in anno_pair:
                opinion_list = char_idx_to_token_idx(tokenized,
                                                     anno_pair['opinion_start'],
                                                     anno_pair['opinion_end'])

            if len(aspect_list) == 0 and len(opinion_list) == 0:
                implicit_both_count += 1

            category = self.label_to_id[anno_pair['aspect_category']]
            sentiment = anno_pair['opinion_sentiment']

            # for statistics of multi-labels
            for aid in aspect_list:
                add_dict_set(cat_dict, aid, category)
            for oid in opinion_list:
                add_dict_set(sent_dict, oid, sentiment)

            # implicit aspect: use the [CLS] and the paired
            # opinion for classification on categories.
            # Note that [CLS] is at the start position [0].
            aspect = aspect_list if aspect_list else [0] + opinion_list
            aspect_key = '_'.join(map(str, aspect))
            # remove redundant aspects
            if aspect_key not in aspect_set:
                aspect_set.add(aspect_key)
                category_idx.append(aspect)
                category_labels.append(category)
            elif len(aspect_list) == 0:
                print(f'WARN: An opinion has multiple implicit aspects:')
                print(text)

            # implicit opinion: use the [CLS] and the paired
            # aspect for classification on sentiments.
            # Note that [CLS] is at the start position.
            opinion = opinion_list if opinion_list else [0] + aspect_list
            opinion_key = '_'.join(map(str, opinion))
            # remove redundant opinions
            if opinion_key not in opinion_set:
                opinion_set.add(opinion_key)
                sentiment_idx.append(opinion)
                sentiment_labels.append(sentiment)
            elif len(opinion_list) == 0:
                print(f'WARN: An aspect has multiple implicit opinions:')
                print(text)

            # the pair classification needs to consider the whole sentence [CLS]
            # here aspect and opinion do not contain the same token idx
            pair_list = sorted([0] + aspect_list + opinion_list)
            pair_key = '_'.join(map(str, pair_list))
            if pair_key not in pair_set:
                pair_set.add(pair_key)
                # implicit aspect or opinion
                # if not aspect_list or not opinion_list:
                #    pair_list = [0] + pair_list
                cat_sent_idx.append(pair_list)
                # positive category-sentiment pair,
                # need to negative sampling later
                cat_sent_labels.append(1)
                pair_aspect = aspect_list if aspect_list else [0]
                pair_opinion = opinion_list if opinion_list else [0]
                # allow one aspect/opinion has more than one category/sentiment.
                pair_labels.append((pair_aspect, pair_opinion, category, sentiment))

        if implicit_both_count > 1:
            print(f'WARN: Multiple implicit aspect-opinion pairs: {implicit_both_count}')

        # for statistics of multi-labels
        cat_multi_labels = np.array(list(map(len, cat_dict.values())))
        cat_repeat = sum(cat_multi_labels > 1)
        if cat_repeat > 0:
            print(f'WARN: category multi-labels: {cat_repeat}/{len(cat_multi_labels)}, {text}')
        sent_multi_labels = np.array(list(map(len, sent_dict.values())))
        sent_repeat = sum(sent_multi_labels > 1)
        if sent_repeat > 0:
            print(f'WARN: sentiment multi-labels: {sent_repeat}/{len(sent_multi_labels)}, {text}')

        # training and validation need negative sampling
        # testing does not need negative sampling
        if neg_sampling:
            # negative sampling
            pair_count = len(cat_sent_idx)
            negative_count = 0
            # try the first method for negative sampling:
            # by the combination of existing categories and sentiments
            if pair_count > 1:
                negative_pairs = list(itertools.product(
                    range(len(category_idx)), range(len(sentiment_idx))))
                negative_idx = np.random.permutation(len(negative_pairs))
                for idx in negative_idx:
                    a_idx, o_idx = negative_pairs[idx]
                    # consider one negative pair candidate
                    # note that aspect and opinion may contain the same token idx
                    # and the pair classification needs to consider the whole sentence [CLS]
                    negative_candidate = sorted(set([0] + category_idx[a_idx] + sentiment_idx[o_idx]))
                    negative_key = '_'.join(map(str, negative_candidate))
                    if negative_key not in pair_set:
                        pair_set.add(negative_key)
                        cat_sent_idx.append(negative_candidate)
                        cat_sent_labels.append(0)
                        negative_count += 1
                        # at most the same number of negative pairs as positive pairs
                        if negative_count == pair_count:
                            break
            # if the negative samples are not enough,
            # try the second method for negative sampling:
            # by assigning exiting categories or sentiments with random words
            if negative_count < pair_count:
                # get non-category and non-sentiment words
                words = set([idx for lst in category_idx for idx in lst])
                words = words.union([idx for lst in sentiment_idx for idx in lst])
                # exclude [0] for [CLS] and the last one for [SEP]
                words_set = set(range(1, word_num - 1)) - words
                words_list = sorted(words_set)
                cat_or_sent = category_idx + sentiment_idx

                while negative_count < pair_count:
                    # the word idx of exiting categories or sentiments
                    exist_ix = np.random.randint(len(cat_or_sent))
                    exist_word = cat_or_sent[exist_ix]
                    rand_word = None
                    # without random words
                    if len(words_list) == 0:
                        # exist_ix points to category,
                        # so get negative sample from category
                        if exist_ix < len(category_idx):
                            rand_word = category_idx[np.random.randint(len(category_idx))]
                        # exist_ix points to sentiment
                        # so get negative sample from sentiment
                        else:
                            rand_word = sentiment_idx[np.random.randint(len(sentiment_idx))]
                    else:
                        # get negative sample from random words
                        # the random word's start idx
                        rand_idx = words_list[np.random.randint(len(words_list))]
                        # the random word's length: from 1 to 3
                        rand_len = np.random.randint(1, 4)
                        rand_word = [rand_idx]
                        for i in range(1, rand_len):
                            if rand_idx + i in words_set:
                                rand_word.append(rand_idx + i)
                            else:
                                # a rand word must have continuous idx
                                break
                    # the pair classification needs to consider the whole sentence [CLS]
                    negative_candidate = sorted(set([0] + exist_word + rand_word))
                    negative_key = '_'.join(map(str, negative_candidate))
                    if negative_key not in pair_set:
                        pair_set.add(negative_key)
                        cat_sent_idx.append(negative_candidate)
                        cat_sent_labels.append(0)
                        negative_count += 1
                    else:
                        print('INFO: conflict on negative sampling!')

        return category_idx, category_labels, sentiment_idx, sentiment_labels, \
            cat_sent_idx, cat_sent_labels, pair_labels

    def get_category_sentiment_pairs_multi_labels(self,
                                                  tokenized: Encoding,
                                                  annotation_pairs,
                                                  add_implicit_tokens=False,
                                                  cls_position=[0]):
        r"""
        Get category and sentiment spans, with multi-labels.
        :param tokenized:
        :param annotation_pairs:
        :param add_implicit_tokens:
        :param cls_position: [0] or []
        :return:
        """
        category_idx = []
        category_labels = []
        sentiment_idx = []
        sentiment_labels = []
        cat_sent_idx = []
        cat_sent_labels = []
        # for testing: [([aspect idx], [opinion idx], category, sentiment), ...]
        pair_labels = []
        key_mappings = {}

        aspect_dict = dict()
        opinion_dict = dict()
        pair_dict = dict()

        length = len(tokenized.tokens)
        for anno_pair in annotation_pairs:
            category = self.label_to_id[anno_pair['aspect_category']]
            sentiment = anno_pair['opinion_sentiment']
            # implicit aspect
            aspect_list = [length - 3] if add_implicit_tokens else []
            # implicit opinion
            opinion_list = [length - 2] if add_implicit_tokens else []
            if 'aspect_start' in anno_pair:
                aspect_list = char_idx_to_token_idx(tokenized,
                                                    anno_pair['aspect_start'],
                                                    anno_pair['aspect_end'])
            if 'opinion_start' in anno_pair:
                opinion_list = char_idx_to_token_idx(tokenized,
                                                     anno_pair['opinion_start'],
                                                     anno_pair['opinion_end'])

            # implicit aspect: use the [CLS] and the paired
            # opinion for classification on categories.
            # Note that [CLS] is at the start position [0].
            aspect = cls_position + aspect_list if aspect_list else cls_position + opinion_list
            aspect_key = '_'.join(map(str, aspect))
            add_dict_set(aspect_dict, aspect_key, category)

            # implicit opinion: use the [CLS] and the paired
            # aspect for classification on sentiments.
            # Note that [CLS] is at the start position.
            opinion = cls_position + opinion_list if opinion_list else cls_position + aspect_list
            opinion_key = '_'.join(map(str, opinion))
            add_dict_set(opinion_dict, opinion_key, sentiment)

            # the pair classification needs to consider the whole sentence [CLS]
            # here aspect and opinion do not contain the same token idx
            pair_list = sorted(cls_position + aspect_list + opinion_list)
            pair_key = '_'.join(map(str, pair_list))
            # convert cartesian product of category-sentiment to labels
            comb_label_id = category * 3 + sentiment
            add_dict_set(pair_dict, pair_key, comb_label_id)

            pair_aspect = aspect_list if aspect_list else cls_position
            pair_opinion = opinion_list if opinion_list else cls_position
            # allow one aspect/opinion has more than one category/sentiment.
            pair_labels.append((pair_aspect, pair_opinion, comb_label_id))
            key_mappings[pair_key] = (aspect_key, opinion_key)

        aspect_key_offsets = {}
        for ix, (k, v) in enumerate(aspect_dict.items()):
            aspect_key_offsets[k] = ix
            category_idx.append(sorted(map(int, k.split('_'))))
            category_labels.append(sorted(v))

        opinion_key_offsets = {}
        for ix, (k, v) in enumerate(opinion_dict.items()):
            opinion_key_offsets[k] = ix
            sentiment_idx.append(sorted(map(int, k.split('_'))))
            sentiment_labels.append(sorted(v))

        # store the relation between pairs and categories/sentiments,
        # used for deriving the pair probability from categories and sentiments
        cat_sent_offsets = []
        for k, v in pair_dict.items():
            cat_sent_idx.append(sorted(map(int, k.split('_'))))
            cat_sent_labels.append(sorted(v))
            aspect_key, opinion_key = key_mappings[k]
            cat_sent_offsets.append([aspect_key_offsets[aspect_key],
                                     opinion_key_offsets[opinion_key]])
        positive_samples = (aspect_dict.keys(), opinion_dict.keys(), pair_dict.keys())

        return category_idx, category_labels, sentiment_idx, sentiment_labels, \
            cat_sent_idx, cat_sent_labels, pair_labels, cat_sent_offsets, positive_samples


@dataclass
class TrainingExample:
    input_ids: IntList
    attention_masks: IntList
    bio_labels: IntList
    category_idx: IntListList
    category_labels: IntList
    sentiment_idx: IntListList
    sentiment_labels: IntList
    cat_sent_idx: IntListList
    cat_sent_labels: IntList
    pair_labels: IntList
    cat_sent_offsets: IntListList
    positive_samples: ()


class TrainingDataset(Dataset):
    def __init__(self, data: Any, bio_label_set: BIOLabelSet,
                 cat_sent_label_set: CategorySentimentLabelSet,
                 tokenizer: PreTrainedTokenizerFast,
                 multi_labels=False, add_implicit_tokens=False,
                 cls_position=[0]):
        r"""

        :param data:
        :param bio_label_set:
        :param cat_sent_label_set:
        :param tokenizer:
        :param multi_labels:
        :param add_implicit_tokens:
        :param cls_position: [0] or []
        """
        self.bio_label_set = bio_label_set
        self.cat_sent_label_set = cat_sent_label_set
        self.tokenizer = tokenizer
        self.texts = []
        self.annotations = []
        self.annotation_pairs = []
        self.bio_class_distribution = torch.zeros(
            len(bio_label_set.id_to_label.values()))

        for example in data:
            self.texts.append(example["content"])
            self.annotations.append(example["annotations"])
            self.annotation_pairs.append(example["annotation_pairs"])

        # TOKENIZE All THE DATA
        tokenized_batch = self.tokenizer(self.texts)

        # MAKE A LIST OF TRAINING EXAMPLES.
        # ALIGN LABELS ONE EXAMPLE AT A TIME
        self.examples: List[TrainingExample] = []
        for ix in range(len(tokenized_batch.encodings)):
            encoding = tokenized_batch.encodings[ix]
            raw_annotations = self.annotations[ix]
            aligned_labels = bio_label_set. \
                get_aligned_label_ids_from_annotations(encoding,
                                                       raw_annotations,
                                                       add_implicit_tokens)
            for bio_id in aligned_labels[1:-1]:
                self.bio_class_distribution[bio_id] += 1
            if multi_labels:
                pair_results = cat_sent_label_set.get_category_sentiment_pairs_multi_labels(
                    encoding, self.annotation_pairs[ix], add_implicit_tokens, cls_position)
            else:
                pair_results = cat_sent_label_set.get_category_sentiment_pairs(
                    encoding, self.annotation_pairs[ix], self.texts[ix])
            example = TrainingExample(input_ids=encoding.ids,
                                      attention_masks=encoding.attention_mask,
                                      bio_labels=aligned_labels,
                                      category_idx=pair_results[0],
                                      category_labels=pair_results[1],
                                      sentiment_idx=pair_results[2],
                                      sentiment_labels=pair_results[3],
                                      cat_sent_idx=pair_results[4],
                                      cat_sent_labels=pair_results[5],
                                      pair_labels=pair_results[6],
                                      cat_sent_offsets=pair_results[7],
                                      positive_samples=pair_results[8])
            self.examples.append(example)
        self.bio_class_weight = self.get_bio_class_weight()

    def get_bio_class_weight(self):
        weight = torch.ones(len(self.bio_class_distribution))
        # the number of 'O' samples
        v, i = self.bio_class_distribution.max(0)
        # the number of all samples
        s = self.bio_class_distribution.sum()
        # set the weight of non-'O' classes to balance their effect
        weight[:i] = v / (s - v)
        weight[i + 1:] = v / (s - v)
        return weight

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> TrainingExample:
        return self.examples[idx]


class TrainingBatch:
    def __init__(self,
                 examples: List[TrainingExample],
                 tokenizer: PreTrainedTokenizerFast,
                 num_cat_labels=0, num_sent_labels=0):
        r"""
        examples already has the same sequence length
        :param examples:
        :param tokenizer:
        :param num_cat_labels:
        :param num_sent_labels:
        """
        input_ids: IntListList = []
        masks: IntListList = []
        labels: IntListList = []

        category_idx: IntListList = []
        category_masks: IntListList = []
        category_labels: IntList = []
        category_nums: IntList = []

        sentiment_idx: IntListList = []
        sentiment_masks: IntListList = []
        sentiment_labels: IntList = []
        sentiment_nums: IntList = []

        cat_sent_idx: IntListList = []
        cat_sent_masks: IntListList = []
        cat_sent_labels: IntList = []
        cat_sent_nums: IntList = []
        pair_labels: IntList = []
        cat_sent_offsets: IntListList = []
        positive_samples_list = []

        # find the max length for different tasks
        bio_seq_max_len = 0
        category_seq_max_len = 0
        sentiment_seq_max_len = 0
        cat_sent_seq_max_len = 0
        for ex in examples:
            bio_seq_max_len = max(bio_seq_max_len,
                                  len(ex.input_ids))
            category_seq_max_len = max(category_seq_max_len,
                                       (max(map(len, ex.category_idx))))
            sentiment_seq_max_len = max(sentiment_seq_max_len,
                                        (max(map(len, ex.sentiment_idx))))
            cat_sent_seq_max_len = max(cat_sent_seq_max_len,
                                       (max(map(len, ex.cat_sent_idx))))

        # padding for different tasks
        for ex in examples:
            # BIO task
            length = len(ex.input_ids)
            pad_num = bio_seq_max_len - length
            # NOTE: do not update ex.input_ids, generate new one
            input_ids.append(ex.input_ids + pad_num * [tokenizer.pad_token_id])
            masks.append(ex.attention_masks + pad_num * [0])
            labels.append(ex.bio_labels + pad_num * [0])

            # aspect's category
            cat_num = len(ex.category_labels)
            category_nums.append(cat_num)
            cat_offset = len(category_idx)
            for idx in range(cat_num):
                length = len(ex.category_idx[idx])
                pad_num = category_seq_max_len - length
                category_idx.append(ex.category_idx[idx] + pad_num * [0])
                category_masks.append(length * [1] + pad_num * [0])
                category_label = ex.category_labels[idx]
                if num_cat_labels > 0:
                    # multi-labels
                    assert isinstance(category_label, list)
                    cat_labels = np.array([0] * num_cat_labels)
                    cat_labels[category_label] = 1
                    category_labels.append(cat_labels.tolist())
                else:
                    assert isinstance(category_label, int)
                    category_labels.append(category_label)

            # opinion's sentiment
            sent_num = len(ex.sentiment_labels)
            sentiment_nums.append(sent_num)
            sent_offset = len(sentiment_idx)
            for idx in range(sent_num):
                length = len(ex.sentiment_idx[idx])
                pad_num = sentiment_seq_max_len - length
                sentiment_idx.append(ex.sentiment_idx[idx] + pad_num * [0])
                sentiment_masks.append(length * [1] + pad_num * [0])
                sentiment_label = ex.sentiment_labels[idx]
                if num_sent_labels > 0:
                    # multi-labels
                    assert isinstance(sentiment_label, list)
                    sent_labels = np.array([0] * num_sent_labels)
                    sent_labels[sentiment_label] = 1
                    sentiment_labels.append(sent_labels.tolist())
                else:
                    assert isinstance(sentiment_label, int)
                    sentiment_labels.append(sentiment_label)

            # aspect-opinion pair
            pair_num = len(ex.cat_sent_labels)
            cat_sent_nums.append(pair_num)
            offset = np.array([[cat_offset, sent_offset]]).repeat(pair_num, axis=0)
            cat_sent_offsets.extend((np.array(ex.cat_sent_offsets) + offset).tolist())
            for idx in range(pair_num):
                length = len(ex.cat_sent_idx[idx])
                pad_num = cat_sent_seq_max_len - length
                cat_sent_idx.append(ex.cat_sent_idx[idx] + pad_num * [0])
                cat_sent_masks.append(length * [1] + pad_num * [0])
                cat_sent_label = ex.cat_sent_labels[idx]
                if num_cat_labels > 0 and num_sent_labels > 0:
                    # multi-labels
                    assert isinstance(cat_sent_label, list)
                    p_labels = np.array([0] * (num_cat_labels * num_sent_labels))
                    p_labels[cat_sent_label] = 1
                    cat_sent_labels.append(p_labels.tolist())
                else:
                    assert isinstance(cat_sent_label, int)
                    cat_sent_labels.append(cat_sent_label)

            # for testing, no need padding
            pair_labels.extend(ex.pair_labels)

            # for negative sampling
            positive_samples_list.append(ex.positive_samples)

        self.input_ids = torch.LongTensor(input_ids)
        self.attention_masks = torch.LongTensor(masks)
        self.bio_labels = torch.LongTensor(labels)

        # for different types of loss function
        label_type = torch.float if num_cat_labels > 0 \
            and num_sent_labels > 0 else torch.long

        # aspect's category
        self.category_idx = torch.LongTensor(category_idx)
        self.category_masks = torch.LongTensor(category_masks)
        self.category_labels = torch.tensor(category_labels, dtype=label_type)
        self.category_nums = torch.LongTensor(category_nums)

        # opinion's sentiment
        self.sentiment_idx = torch.LongTensor(sentiment_idx)
        self.sentiment_masks = torch.LongTensor(sentiment_masks)
        self.sentiment_labels = torch.tensor(sentiment_labels, dtype=label_type)
        self.sentiment_nums = torch.LongTensor(sentiment_nums)

        # aspect-opinion pair
        self.cat_sent_idx = torch.LongTensor(cat_sent_idx)
        self.cat_sent_masks = torch.LongTensor(cat_sent_masks)
        self.cat_sent_labels = torch.tensor(cat_sent_labels, dtype=label_type)
        self.cat_sent_nums = torch.LongTensor(cat_sent_nums)
        self.cat_sent_offsets = torch.LongTensor(cat_sent_offsets)
        self.pair_labels = pair_labels
        self.positive_samples_list = positive_samples_list

    def __getitem__(self, item):
        return getattr(self, item)


def extract_aspect_opinion(pred, bio_mask, bio_label_set: BIOLabelSet):
    r"""
    Extract aspects and opinions from BIO results for a sentence,
    which are saved into category_id and sentiment_id, respectively.
    :param pred: BIO ids
    :param bio_mask: the actual length of a sentence
    :param bio_label_set: convert BIO ids to labels
    :return: the aspects and opinions in this sentence.
    """
    # save word indices of aspects
    category_id: IntListList = []
    # save word indices of opinions
    sentiment_id: IntListList = []
    cur_aspect = []
    cur_opinion = []
    # DO NOT consider the first one [CLS] and the last one [SEP]
    for ix in range(1, bio_mask.sum() - 1):
        pred_label = bio_label_set.id_to_label[pred[ix].item()]
        if pred_label == 'O':
            if cur_aspect:
                category_id.append(cur_aspect)
                cur_aspect = []
            if cur_opinion:
                sentiment_id.append(cur_opinion)
                cur_opinion = []
        elif pred_label.endswith('Aspect'):
            if cur_opinion:
                sentiment_id.append(cur_opinion)
                cur_opinion = []
            if pred_label.startswith('B'):
                if cur_aspect:
                    category_id.append(cur_aspect)
                    cur_aspect = []
                cur_aspect.append(ix)
            elif pred_label.startswith('I') or pred_label.startswith('L'):
                # if len(cur_aspect) == 0:
                #     print('Warning: I-Aspect or L-Aspect at the beginning!!!')
                cur_aspect.append(ix)
            elif pred_label.startswith('U'):
                if cur_aspect:
                    category_id.append(cur_aspect)
                    cur_aspect = []
                category_id.append([ix])
        elif pred_label.endswith('Opinion'):
            if cur_aspect:
                category_id.append(cur_aspect)
                cur_aspect = []
            if pred_label.startswith('B'):
                if cur_opinion:
                    sentiment_id.append(cur_opinion)
                    cur_opinion = []
                cur_opinion.append(ix)
            elif pred_label.startswith('I') or pred_label.startswith('L'):
                # if len(cur_opinion) == 0:
                #     print('Warning: I-Opinion or L-Opinion at the beginning!!!')
                cur_opinion.append(ix)
            elif pred_label.startswith('U'):
                if cur_opinion:
                    sentiment_id.append(cur_opinion)
                    cur_opinion = []
                sentiment_id.append([ix])
    # last aspect and opinion
    if cur_aspect:
        category_id.append(cur_aspect)
    if cur_opinion:
        sentiment_id.append(cur_opinion)
    return category_id, sentiment_id


def padding(batch_data: IntListList):
    r"""
    Pad the data with seq_var_length to seq_max_len.
    :param batch_data: [batch_size, seq_var_length]
    :return: masks with [batch_size, seq_max_len]
    """
    masks: IntListList = []
    seq_max_len = max(map(len, batch_data))
    for example in batch_data:
        length = len(example)
        pad_num = seq_max_len - length
        # NOTE: update example
        example.extend(pad_num * [0])
        masks.append(length * [1] + pad_num * [0])
    return masks


def filter_truth(candidate_list: IntListList, truth_set, num):
    for i in sorted(range(len(candidate_list)), reverse=True):
        key = '_'.join(map(str, candidate_list[i]))
        if key in truth_set:
            candidate_list.pop(i)
            if i < num:
                num -= 1
    return num


class CategorySentimentPairs:
    def __init__(self, bio_logits, bio_masks, bio_label_set: BIOLabelSet,
                 device, add_implicit_tokens=False, cls_position=[0],
                 truth_filters=None):
        r"""
        Generate the category-sentiment candidates based on the cartesian product
        of aspects and opinions which are extracted from BIO labeling results.
        :param bio_logits: [batch_size, seq_length, num_labels] or [batch_size, seq_length] when use CRF.
        :param bio_masks: [batch_size, seq_length]
        :param bio_label_set:
        :param device:
        :param add_implicit_tokens:
        :param cls_position: [0] or []
        :param truth_filters: for negative sampling
        """
        category_idx: IntListList = []
        category_masks: IntListList
        category_nums: IntList = []
        explicit_category_nums: IntList = []

        sentiment_idx: IntListList = []
        sentiment_masks: IntListList
        sentiment_nums: IntList = []
        explicit_sentiment_nums: IntList = []

        cat_sent_idx: IntListList = []
        cat_sent_masks: IntListList
        cat_sent_nums: IntList = []
        pair_candidates: IntList = []

        # bio_logits with [batch_size, seq_length, num_labels]
        # or [batch_size, seq_length] when use CRF.
        # get the label indices with the max values
        pred = bio_logits if bio_logits.dim() == bio_masks.dim() else bio_logits.max(-1)[1]
        batch_size, _ = pred.size()
        for bix in range(batch_size):
            # when add_implicit_tokens is True,
            # category_id should include implicit aspect,
            # and sentiment_id should include implicit opinion.
            category_id, sentiment_id = extract_aspect_opinion(
                pred[bix], bio_masks[bix], bio_label_set)
            length = sum(bio_masks[bix])
            if add_implicit_tokens:
                # NOTE: [length -1] for [SEP]
                ia_token_id = [length - 3]
                for asp in category_id:
                    if asp == ia_token_id:
                        break
                else:
                    category_id.append(ia_token_id)

                io_token_id = [length - 2]
                for opi in sentiment_id:
                    if opi == io_token_id:
                        break
                else:
                    sentiment_id.append(io_token_id)

            aspect_num = len(category_id)
            opinion_num = len(sentiment_id)
            # pair candidates: cartesian product
            # the pair classification needs to consider the whole sentence [CLS]
            cat_sent_id = [sorted(cls_position + pair[0] + pair[1]) for pair in
                           itertools.product(category_id, sentiment_id)]
            # need to deepcopy due to padding later
            pair_candidate = [(pair[0], pair[1]) for pair in
                              itertools.product(copy.deepcopy(category_id),
                                                copy.deepcopy(sentiment_id))]

            # implicit aspects only
            if opinion_num and (add_implicit_tokens is False):
                implicit_aspects = [sorted(pair[0] + pair[1]) for pair in
                                    itertools.product([cls_position], sentiment_id)]
                # for implicit aspects, the following both are the same
                category_id.extend(implicit_aspects)
                # need to deepcopy due to padding later
                cat_sent_id.extend(copy.deepcopy(implicit_aspects))
                # for test: aspect before opinion
                pair_candidate.extend([(pair[0], pair[1]) for pair in
                                       itertools.product([cls_position],
                                                         copy.deepcopy(sentiment_id))])
            # implicit opinions only
            if aspect_num and (add_implicit_tokens is False):
                # NOTE: because category_id may be extended,
                # so we must use category_id[:aspect_num]
                implicit_opinions = [sorted(pair[0] + pair[1]) for pair in
                                     itertools.product([cls_position], category_id[:aspect_num])]
                # for implicit opinions, the following both are the same
                sentiment_id.extend(implicit_opinions)
                # need to deepcopy due to padding later
                cat_sent_id.extend(copy.deepcopy(implicit_opinions))
                # for test: aspect before opinion
                pair_candidate.extend([(pair[1], pair[0]) for pair in
                                       itertools.product([cls_position],
                                                         copy.deepcopy(category_id[:aspect_num]))])
            # implicit both aspects and opinions
            if add_implicit_tokens is False:
                category_id.append([0])
                sentiment_id.append([0])
                cat_sent_id.append([0])
                pair_candidate.append(([0], [0]))

            category_id_temp = []
            for aspect in category_id[:aspect_num]:
                category_id_temp.append(cls_position + aspect)
            category_id_temp.extend(category_id[aspect_num:])

            sentiment_id_temp = []
            for opinion in sentiment_id[:opinion_num]:
                sentiment_id_temp.append(cls_position + opinion)
            sentiment_id_temp.extend(sentiment_id[opinion_num:])

            # filer out positive samples for negative sampling
            if truth_filters is not None:
                aspect_set, sentiment_set, pair_set = truth_filters[bix]
                aspect_num = filter_truth(category_id_temp, aspect_set, aspect_num)
                opinion_num = filter_truth(sentiment_id_temp, sentiment_set, opinion_num)
                for i in sorted(range(len(cat_sent_id)), reverse=True):
                    key = '_'.join(map(str, cat_sent_id[i]))
                    if key in pair_set:
                        cat_sent_id.pop(i)
                        pair_candidate.pop(i)

            # accumulate aspects
            # len: aspect_num + opinion_num + 1
            category_idx.extend(category_id_temp)
            category_nums.append(len(category_id_temp))
            explicit_category_nums.append(aspect_num)

            # accumulate opinions
            # len: opinion_num + aspect_num + 1
            sentiment_idx.extend(sentiment_id_temp)
            sentiment_nums.append(len(sentiment_id_temp))
            explicit_sentiment_nums.append(opinion_num)

            # accumulate pairs
            cat_sent_idx.extend(cat_sent_id)
            # len: aspect_num * opinion_num + opinion_num + aspect_num + 1
            cat_sent_nums.append(len(cat_sent_id))
            # for test
            pair_candidates.extend(pair_candidate)

        # padding for a batch
        category_masks = padding(category_idx) if category_idx else []
        sentiment_masks = padding(sentiment_idx) if sentiment_idx else []
        cat_sent_masks = padding(cat_sent_idx) if cat_sent_idx else []
        # for aspect's category
        self.category_idx = torch.LongTensor(category_idx).to(device)
        self.category_masks = torch.LongTensor(category_masks).to(device)
        self.category_nums = torch.LongTensor(category_nums).to(device)
        self.explicit_category_nums = torch.LongTensor(explicit_category_nums).to(device)
        # for opinion's sentiment
        self.sentiment_idx = torch.LongTensor(sentiment_idx).to(device)
        self.sentiment_masks = torch.LongTensor(sentiment_masks).to(device)
        self.sentiment_nums = torch.LongTensor(sentiment_nums).to(device)
        self.explicit_sentiment_nums = torch.LongTensor(explicit_sentiment_nums).to(device)
        # for aspect-opinion pair
        self.cat_sent_idx = torch.LongTensor(cat_sent_idx).to(device)
        self.cat_sent_masks = torch.LongTensor(cat_sent_masks).to(device)
        self.cat_sent_nums = torch.LongTensor(cat_sent_nums).to(device)
        self.pair_candidates = pair_candidates

    def decode_pair_prob(self, cat_prob, sent_prob):
        r"""
        NOTE: the batch size must be one.
        :param cat_prob:
        :param sent_prob:
        :return:
        """
        c1 = cat_prob[:self.explicit_category_nums].unsqueeze(1) * \
             sent_prob[:self.explicit_sentiment_nums].unsqueeze(0)
        c2 = cat_prob[-1 - self.explicit_sentiment_nums:-1] * \
             sent_prob[:self.explicit_sentiment_nums]
        c3 = cat_prob[:self.explicit_category_nums] * \
             sent_prob[-1 - self.explicit_category_nums:-1]
        c4 = cat_prob[-1:] * sent_prob[-1:]
        # len: aspect_num * opinion_num + opinion_num + aspect_num + 1
        return torch.cat((c1.view(-1), c2, c3, c4))

    def __getitem__(self, item):
        return getattr(self, item)


def get_loss(model, bio_loss_fn, loss_fn,
             bio_logits, bio_labels,
             cat_logits, category_labels,
             sent_logits, sentiment_labels,
             pair_logits, cat_sent_labels,
             neg_fn=None):
    def neg_expand(logits, labels):
        b1, d1 = logits.size()
        b2, d2 = labels.size()
        assert d1 == d2
        if b1 > b2:
            labels.resize_(b1, d1)
            labels[b2:b1] = 0.0
        return labels

    # bio_logits is the loss when it is a scalar
    bio_loss = bio_loss_fn(bio_logits.view(-1, model.num_labels),
                           bio_labels.view(-1)) if bio_logits.dim() > 0 else bio_logits
    if neg_fn is not None:
        category_labels = neg_expand(cat_logits, category_labels)
        sentiment_labels = neg_expand(sent_logits, sentiment_labels)
        cat_sent_labels = neg_expand(pair_logits, cat_sent_labels)
    cat_loss = loss_fn(cat_logits, category_labels)
    sent_loss = loss_fn(sent_logits, sentiment_labels)
    pair_loss = loss_fn(pair_logits, cat_sent_labels)
    return bio_loss, cat_loss, sent_loss, pair_loss


def get_metrics(pred, labels):
    r"""
    Get hit counts.
    :param pred: logits [batch_size, ..., num_labels] or [batch_size, ...]
    :param labels: [batch_size, ...], which has the same size as logits except for the last dim.
    :return:
    """
    # get the label indices with the max values
    if pred.dim() == labels.dim() + 1:
        pred = pred.max(-1)[1]
    # -100 is a special token for padding of labels
    # see torch.nn.NLLLoss(ignore_index=-100)
    label_masks = ~labels.eq(-100)
    total_num = label_masks.sum().item()
    right_num = pred.eq(labels).mul(label_masks).sum().item()
    return total_num, right_num


def get_metrics_multi_labels(logits, labels):
    r"""
    Get hit counts.
    :param logits: logits [batch_size, num_labels].
    :param labels: [batch_size, num_labels] , which has the same size as logits.
    :return:
    """
    pred = (logits.sigmoid() > 0.5).type(torch.float)
    label_masks = labels.eq(1.0)
    total_num = label_masks.sum().item()
    right_num = pred.eq(labels).mul(label_masks).sum().item()
    return total_num, right_num


def get_pred_mapping(pred, idx, masks):
    r"""
    Get the predicting mapping for a text with multiple results.
    :param pred: [result_num] that is 'aspect_num + opinion_num + 1'.
    :param idx: [result_num, seq_length]
    :param masks: [result_num, seq_length]
    :return:
    """
    # NOTE: ONLY for a text
    pred_dict = {}
    # explicit aspects and explicit opinions
    for i, idx in enumerate(idx):
        for ix in idx[:sum(masks[i])]:
            # NOTE: only 'ix==0' will duplicate,
            # the later one will replace its previous one,
            # the last one with implicit aspect and implicit opinion
            pred_dict[ix.item()] = pred[i].item()
    return pred_dict


def get_hit_count(pair_labels, cat_label, sent_label,
                  aspect, cat_dict, opinion, sent_dict,
                  iou_min=1.0):
    r"""
    Calculate the hit by comparing aspect ids, opinion ids,
    and their category and sentiment.
    :param pair_labels: [([aspect ids], [opinion ids], category, sentiment), ...]
    :param cat_label:
    :param sent_label:
    :param aspect:
    :param cat_dict:
    :param opinion:
    :param sent_dict:
    :param iou_min:
    :return:
    """
    # BIO alignment performance,
    # find the maximum matching
    iou_max = -1.0
    for pair_label in pair_labels:
        aspect_iou = (len(set(pair_label[0]).intersection(set(aspect))) /
                      len(set(pair_label[0]).union(set(aspect))))
        opinion_iou = (len(set(pair_label[1]).intersection(set(opinion))) /
                       len(set(pair_label[1]).union(set(opinion))))
        iou_max = max(iou_max, min(aspect_iou, opinion_iou))

    if iou_max < iou_min:
        return 0
    # classification performance
    # NOTE: all ids in aspect/opinion have the same
    # category/sentiment in cat_dict/sent_dict.
    category = cat_dict[aspect[0]]
    sentiment = sent_dict[opinion[0]]
    # when iou_min is 1.0, here cat_set and sent_set has only one element
    cat_set = set()
    for a in aspect:
        if a in cat_label:
            cat_set.add(cat_label[a])
    sent_set = set()
    for o in opinion:
        if o in sent_label:
            sent_set.add(sent_label[o])
    if category in cat_set and sentiment in sent_set:
        if min(1.0 / len(cat_set), 1.0 / len(sent_set)) >= iou_min:
            return 1
    return 0


def get_hit_count_multi_labels(pair_labels, aspect, opinion, comb_label_id, iou_min):
    r"""
    Calculate the hit by comparing aspect ids, opinion ids,
    and their category and sentiment.
    :param pair_labels: [([aspect ids], [opinion ids], category_sentiment_id), ...]
    :param aspect:
    :param opinion:
    :param comb_label_id:
    :param iou_min:
    :return:
    """
    # BIO alignment performance,
    # find the maximum matching
    iou_max = -1.0
    for pair_label in pair_labels:
        # compare the combination id of category and sentiment
        if pair_label[2] != comb_label_id:
            continue
        # compare aspect ids and opinion ids
        aspect_iou = (len(set(pair_label[0]).intersection(set(aspect))) /
                      len(set(pair_label[0]).union(set(aspect))))
        opinion_iou = (len(set(pair_label[1]).intersection(set(opinion))) /
                       len(set(pair_label[1]).union(set(opinion))))
        iou_max = max(iou_max, min(aspect_iou, opinion_iou))
    return 0 if iou_max < iou_min else 1


def get_phrase(words, offsets, delimiter=' '):
    r"""
    Get a phrase form words based on offsets.
    :param words:
    :param offsets:
    :param delimiter:
    :return:
    """
    return delimiter.join([words[i] for i in offsets])


def decode_pairs(tokenizer, df, input_ids, pair_labels,
                 predictions, hit_count, cat_sent_label_set):
    r"""
    Decode predicted pairs to readable labels.
    :param tokenizer: convert ids to tokens
    :param df: dataframe of panda for output
    :param input_ids:
    :param pair_labels:
    :param predictions:
    :param hit_count:
    :param cat_sent_label_set:
    """
    # NOTE: batch size must be one
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    text = get_phrase(tokens, range(len(tokens)))

    pair_list = []
    for pair_label in pair_labels:
        aspect_true = get_phrase(tokens, pair_label[0])
        opinion_true = get_phrase(tokens, pair_label[1])
        cat_true = cat_sent_label_set.id_to_label[pair_label[2] // 3]
        sent_true = pair_label[2] % 3
        pair_list.append((aspect_true, cat_true, opinion_true, sent_true))

    pred_list = []
    for aspect, opinion, comb_label_id in predictions:
        aspect_pred = get_phrase(tokens, aspect)
        opinion_pred = get_phrase(tokens, opinion)
        cat_pred = cat_sent_label_set.id_to_label[comb_label_id // 3]
        sent_pred = comb_label_id % 3
        pred_list.append((aspect_pred, cat_pred, opinion_pred, sent_pred))

    df.loc[len(df)] = [text, str(pair_list), str(pred_list), hit_count]


def train(data_loader, model, neg_fn, bio_loss_fn, loss_fn,
          optimizer, device, global_step, writer: SummaryWriter,
          use_crf=False, multi_tasks=True, cross_view=False, print_period=1):
    r"""
    Train the model.
    :param data_loader:
    :param model:
    :param neg_fn: negative sampling function
    :param bio_loss_fn:
    :param loss_fn:
    :param optimizer:
    :param device:
    :param global_step: count the current number of trained batches.
    :param writer:
    :param use_crf:
    :param multi_tasks:
    :param cross_view:
    :param print_period:
    :return:
    """
    model.train()
    total_loss = 0
    batch_idx = 0
    crf_decode = use_crf and neg_fn is not None
    for batch in data_loader:
        batch_idx += 1
        global_step += 1
        # bio_logits with size(batch_num, seq_length, num_labels)
        # other logits with size(examples, *_num_labels)
        bio_logits, cat_logits, sent_logits, pair_logits \
            = model(batch, device, crf_decode, neg_fn)
        bio_logits = bio_logits[0] if crf_decode else bio_logits
        losses = get_loss(model, bio_loss_fn, loss_fn,
                          bio_logits, batch.bio_labels.to(device),
                          cat_logits, batch.category_labels.to(device),
                          sent_logits, batch.sentiment_labels.to(device),
                          pair_logits, batch.cat_sent_labels.to(device),
                          neg_fn)
        loss = sum(losses) if multi_tasks else losses[0] + 0*losses[1] + 0*losses[2] + losses[3]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # train cross view loss
        # make pair prediction close to both category and sentiment prediction
        if cross_view and isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
            # NOTE: here no need to do negative sampling,
            # because we cannot align pairs with categories and sentiments.
            _, cat_logits, sent_logits, pair_logits \
                = model(batch, device, False, None)
            cat_sent_offsets = batch.cat_sent_offsets.to(device)
            cat_prob = cat_logits.sigmoid()
            sent_prob = sent_logits.sigmoid()
            pair_prob = pair_logits.sigmoid()
            cat_prob_ex = cat_prob[cat_sent_offsets[:, 0]].unsqueeze(2)
            sent_prob_ex = sent_prob[cat_sent_offsets[:, 1]].unsqueeze(1)
            cat_sent_prob = (cat_prob_ex * sent_prob_ex).view([len(cat_sent_offsets), -1])
            cv_loss = torch.nn.functional.binary_cross_entropy(cat_sent_prob, pair_prob)

            set_grad(model.cat_sent_pair, False)
            optimizer.zero_grad()
            cv_loss.backward()
            optimizer.step()
            set_grad(model.cat_sent_pair, True)

            if global_step % print_period == 0:
                print(f'step: {global_step}, cv loss: {cv_loss.item()}')

        # logging
        loss = loss.item()
        total_loss += loss
        writer.add_scalar('Train/loss/step', loss, global_step)
        if global_step % print_period == 0:
            print(f'step: {global_step}, loss: {loss}')

        if np.isnan(loss):
            for param in model.named_parameters():
                if np.isnan(torch.sum(param[1].grad).item()):
                    print(param[0], 'grad:', param[1].grad)
            break
    return global_step, total_loss / batch_idx


def validate(data_loader, model, bio_loss_fn, loss_fn, device,
             use_crf=False, multi_labels=False, multi_tasks=True):
    r"""
    Validate the model to get performance metrics.
    :param data_loader:
    :param model:
    :param bio_loss_fn:
    :param loss_fn:
    :param device:
    :param use_crf:
    :param multi_labels:
    :param multi_tasks:
    :return:
    """
    model.eval()
    total_num = np.zeros(4)
    right_num = np.zeros(4)
    total_loss = 0
    batch_idx = 0
    for batch in data_loader:
        batch_idx += 1
        # bio_logits with size(batch_num, seq_length, num_labels)
        # other logits with size(examples, *_num_labels)
        # If the use_crf is True, then set crf_decode to True
        logits = model(batch, device, use_crf)
        labels = (batch.bio_labels.to(device),
                  batch.category_labels.to(device),
                  batch.sentiment_labels.to(device),
                  batch.cat_sent_labels.to(device))
        bio_logits = logits[0][0] if use_crf else logits[0]
        losses = get_loss(model, bio_loss_fn, loss_fn,
                          bio_logits, labels[0],
                          logits[1], labels[1],
                          logits[2], labels[2],
                          logits[3], labels[3])
        # note that: total_loss += sum(losses) causes out-of-memory,
        # i.e., "+=" of tensor in loop
        # which may result in the explosion of computation graph.
        delta_loss = sum(losses) if multi_tasks else losses[0] + losses[-1]
        total_loss += delta_loss.item()
        # for the BIO task
        bio_pred = logits[0][1].to(device) if use_crf else logits[0]
        total, right = get_metrics(bio_pred, labels[0])
        total_num[0] += total
        right_num[0] += right
        # for other tasks
        for i in range(1, 4):
            if multi_labels:
                total, right = get_metrics_multi_labels(logits[i], labels[i])
            else:
                total, right = get_metrics(logits[i], labels[i])
            total_num[i] += total
            right_num[i] += right
        if np.isnan(total_loss):
            break
    accuracy = right_num / total_num
    return total_loss / batch_idx, accuracy


def test(data_loader, model,
         bio_label_set, device,
         iou_min=1.0, top_k_pairs: int = 0,
         joint_prob=False, multi_labels=False,
         tokenizer=None, df=None, cat_sent_label_set=None,
         add_implicit_tokens=False, cls_position=[0]):
    r"""
    Test the model to get performance metrics.
    NOTE: the batch size must be one.
    :param data_loader:
    :param model:
    :param device:
    :param iou_min:
    :param top_k_pairs: 0 or less has no effect
    :param joint_prob: calculate the joint prob of categories and sentiments.
    :param multi_labels:
    :param tokenizer: convert ids to tokens
    :param df: dataframe of panda for output
    :param cat_sent_label_set:
    :param add_implicit_tokens:
    :param cls_position: [0] or []
    :return:
    """
    def is_implicit(token, add_implicit, cls_token, implicit_token):
        if add_implicit:
            if token == implicit_token:
                return True
        else:
            if token == cls_token:
                return True
        return False

    model.eval()
    total_count = 0
    hit_count = 0
    propose_count = 0
    batch_idx = 0
    threshold = 0.5
    # detail_results[metric type, aspect type, opinion type]
    detail_results = np.zeros([3, 2, 2])
    if joint_prob and not multi_labels:
        threshold = threshold ** 3
    for batch in data_loader:
        batch_idx += 1
        # bio_logits with size(batch_num, seq_length, num_labels)
        bio_masks = batch.attention_masks.to(device)
        assert len(bio_masks) == 1
        length = sum(bio_masks[0])
        seq_features = model.get_pretrained_features(
            batch.input_ids.to(device), bio_masks)
        # Always set crf_decode to True
        bio_logits = model.predict_bio(seq_features, bio_masks, None, True)
        # to(device) for CRF layer
        csp = CategorySentimentPairs(bio_logits.to(device), bio_masks,
                                     bio_label_set, device, add_implicit_tokens,
                                     cls_position)

        # aspect-opinion pairs
        pair_logits = model.predict_pair(seq_features,
                                         csp.cat_sent_nums,
                                         csp.cat_sent_idx,
                                         csp.cat_sent_masks)

        if multi_labels:
            # pair_logits: [num_examples, num_cat_labels * num_sent_labels]
            comb_dim = pair_logits.size()[1]
            # binary probabilities to a vector
            pair_pred = pair_logits.sigmoid().view(-1)
        else:
            # [num_examples, 2]
            pair_pred = pair_logits.softmax(dim=1)

            # other logits with size(examples, *_num_labels)
            # aspect's category
            cat_logits = model.predict_category(seq_features,
                                                csp.category_nums,
                                                csp.category_idx,
                                                csp.category_masks)
            cat_prob, cat_pred = cat_logits.softmax(dim=1).max(-1)
            cat_dict = get_pred_mapping(cat_pred,
                                        csp.category_idx,
                                        csp.category_masks)
            cat_label = get_pred_mapping(batch.category_labels,
                                         batch.category_idx,
                                         batch.category_masks)

            # opinion's sentiment
            sent_logits = model.predict_sentiment(seq_features,
                                                  csp.sentiment_nums,
                                                  csp.sentiment_idx,
                                                  csp.sentiment_masks)
            sent_prob, sent_pred = sent_logits.softmax(dim=1).max(-1)
            sent_dict = get_pred_mapping(sent_pred,
                                         csp.sentiment_idx,
                                         csp.sentiment_masks)
            sent_label = get_pred_mapping(batch.sentiment_labels,
                                          batch.sentiment_idx,
                                          batch.sentiment_masks)

            if joint_prob:
                pair_pred = pair_pred * csp.decode_pair_prob(cat_prob, sent_prob).unsqueeze(1)
            pair_pred = pair_pred[:, 1]

        positive_count = sum(pair_pred > threshold)
        # sort according to the positive probability
        indices = pair_pred.sort(0, descending=True)[1]
        actual_k = top_k_pairs if top_k_pairs > 0 else positive_count
        total_count += len(batch.pair_labels)
        predictions = []
        hit_count_start = hit_count
        # the batch size must be one
        for pair_label in batch.pair_labels:
            at = int(is_implicit(pair_label[0], add_implicit_tokens, cls_position, [length - 3]))
            ot = int(is_implicit(pair_label[1], add_implicit_tokens, cls_position, [length - 2]))
            detail_results[0, at, ot] += 1

        for ix in range(actual_k):
            propose_count += 1
            if multi_labels:
                ixx = torch.div(indices[ix], comb_dim, rounding_mode='trunc')
                aspect, opinion = csp.pair_candidates[ixx]
                comb_label_id = indices[ix] % comb_dim
                diff_count = get_hit_count_multi_labels(
                    batch.pair_labels, aspect, opinion, comb_label_id, iou_min)
                predictions.append((aspect, opinion, comb_label_id.item()))
            else:
                aspect, opinion = csp.pair_candidates[indices[ix]]
                diff_count = get_hit_count(batch.pair_labels, cat_label, sent_label,
                                           aspect, cat_dict, opinion, sent_dict, iou_min)
            hit_count += diff_count
            at = int(is_implicit(aspect, add_implicit_tokens, cls_position, [length - 3]))
            ot = int(is_implicit(opinion, add_implicit_tokens, cls_position, [length - 2]))
            detail_results[1, at, ot] += 1
            detail_results[2, at, ot] += diff_count

        if tokenizer:
            hit_diff = hit_count - hit_count_start
            decode_pairs(tokenizer, df, batch.input_ids, batch.pair_labels,
                         predictions, hit_diff, cat_sent_label_set)

    precision = hit_count / propose_count if propose_count else 0
    recall = hit_count / total_count
    f1 = 2 * precision * recall / (precision + recall) if hit_count else 0
    detail_pre = detail_results[2] / (detail_results[1] + 1e-10)
    detail_recall = detail_results[2] / (detail_results[0] + 1e-10)
    detail_f1 = 2 * detail_pre * detail_recall / (detail_pre + detail_recall + 1e-10)
    print(f'total_count: {total_count}, propose_count: {propose_count}, hit_count: {hit_count}')
    return precision, recall, f1, detail_f1
