#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import math
from typing import Optional, Tuple, Union
import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from transformers import GPT2Model, GPT2PreTrainedModel
from torchcrf import CRF


class ACOS:
    r"""
    A class for sequence labeling using pre-trained models, e.g., BERT and GPT.
    """

    def __init__(self, model, config, num_cat_labels, num_sent_labels,
                 use_crf=False, query=0, multi_heads=0, cat_input_ids=None,
                 cat_attention_mask=None, polarity_ids=None,
                 multi_labels=False, neg_num=False, num_layers=2):
        self.pretrained_model = model
        self.use_crf = use_crf
        self.multi_labels = multi_labels
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.num_cat_labels = num_cat_labels
        self.num_sent_labels = num_sent_labels
        self.neg_num = neg_num
        self.neg_max = abs(self.neg_num)
        if not hasattr(config, 'hidden_dropout_prob'):
            config.hidden_dropout_prob = 0.1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # BIO task
        self.bio_classifier = nn.Linear(config.hidden_size, config.num_labels)
        if self.use_crf:
            self.crf = CRF(config.num_labels, batch_first=True)

        self.att_first = True if multi_heads > 0 else False
        self.multi_heads = abs(multi_heads)

        # aspect's category
        self.multi_head_cat = nn.MultiheadAttention(config.hidden_size, self.multi_heads,
                                                    dropout=config.hidden_dropout_prob,
                                                    batch_first=True) if self.multi_heads else None
        self.aspect_category = nn.Linear(config.hidden_size, num_cat_labels)
        self.cat_input_ids = cat_input_ids
        self.cat_attention_mask = cat_attention_mask
        self.cat_query = None if query == 0 else nn.init.kaiming_uniform_(
            torch.nn.Parameter(torch.empty([1, config.hidden_size])), a=math.sqrt(5))

        # opinion's sentiment
        self.multi_head_sent = nn.MultiheadAttention(config.hidden_size, self.multi_heads,
                                                     dropout=config.hidden_dropout_prob,
                                                     batch_first=True) if self.multi_heads else None
        self.opinion_sentiment = nn.Linear(config.hidden_size, num_sent_labels)
        self.polarity_ids = polarity_ids
        self.sent_query = None if query == 0 else nn.init.kaiming_uniform_(
            torch.nn.Parameter(torch.empty([1, config.hidden_size])), a=math.sqrt(5))

        # aspect-opinion pair
        num_pairs = num_cat_labels * num_sent_labels if self.multi_labels else 2
        self.multi_head_pair = nn.MultiheadAttention(config.hidden_size, self.multi_heads,
                                                     dropout=config.hidden_dropout_prob,
                                                     batch_first=True) if self.multi_heads else None
        self.pair_query = nn.init.kaiming_uniform_(torch.nn.Parameter(
            torch.empty([1, config.hidden_size])), a=math.sqrt(5)) if self.multi_heads else None
        self.cat_sent_pair = nn.Sequential()
        for layer in range(1, num_layers + 1):
            if layer < num_layers:
                self.cat_sent_pair.append(
                    nn.Linear(int(config.hidden_size / (2 ** (layer - 1))),
                              int(config.hidden_size / (2 ** layer))))
                self.cat_sent_pair.append(nn.ReLU())
            else:
                self.cat_sent_pair.append(
                    nn.Linear(int(config.hidden_size / (2 ** (layer - 1))),
                              num_pairs))

    def get_pretrained_features(self, input_ids, attention_mask):
        r"""
        Get the BERT output features of a given text.
        :param input_ids:
        :param attention_mask:
        :return: output features with [batch_size, seq_length, hidden_size]
        """
        # get the output features of a given text
        outputs = self.pretrained_model(input_ids, attention_mask=attention_mask)
        # the input and output: [batch_size, seq_length, hidden_size]
        return self.dropout(outputs[0])

    def predict_bio(self, seq_features, attention_mask, bio_labels, crf_decode):
        r"""
        BIO sequence labeling.
        :param seq_features:
        :param attention_mask:
        :param bio_labels:
        :param crf_decode: effect only when use crf for bio.
        :return:
        """
        # bio_logits: [batch_size, seq_length, num_labels]
        bio_logits = self.bio_classifier(seq_features)
        if not self.use_crf:
            return bio_logits

        # use crf
        # remove the first one for [CLS]
        # and convert the mask to boolean
        crf_mask = attention_mask[:, 1:] > 0
        # reset the last one for [SEP]
        seq_len = crf_mask.sum(1)
        # NOTE: cannot use ':', instead use range to generate list
        crf_mask[range(len(seq_len)), seq_len - 1] = False
        crf_result = []
        if bio_labels is not None:
            crf_labels = copy.deepcopy(bio_labels[:, 1:])
            crf_labels[:, seq_len - 1] = 0
            # NOTE: crf returns 'log likelihood' without negative
            # bio loss: scalar
            bio_loss = -self.crf(bio_logits[:, 1:], crf_labels, crf_mask)
            crf_result.append(bio_loss)
        if crf_decode:
            bio_pred = torch.zeros(bio_logits.size()[0:2], dtype=torch.long)
            # crf prediction: [batch_size, seq_length-2] with different lengths
            crf_pred = self.crf.decode(bio_logits[:, 1:], crf_mask)
            for ix, sample in enumerate(crf_pred):
                # 0 for [CLS]
                bio_pred[ix, 1:1 + len(sample)] = torch.LongTensor(sample)
            crf_result.append(bio_pred)

        return crf_result[0] if len(crf_result) == 1 else crf_result

    def predict_category(self, seq_features, category_nums,
                         category_idx, category_masks):
        r"""
        Predict the category of aspects.
        :param seq_features:
        :param category_nums:
        :param category_idx:
        :param category_masks: [example_num, seq_len]
        :return: category logits
        """
        seq_features_cat = self.get_features(seq_features,
                                             category_nums,
                                             category_idx,
                                             category_masks)
        # cat_logits
        if self.multi_head_cat is None:
            return self.aspect_category(seq_features_cat[0])
        if self.cat_input_ids is None:
            cat_cls = None
        else:
            outputs = self.pretrained_model(self.cat_input_ids, self.cat_attention_mask)
            cat_cls = outputs[0][:, 0]
        if self.cat_query is not None:
            batch_cat_queries = self.cat_query.repeat(len(category_masks), 1, 1)
            # NOTE: torch and transforms define masks reversely.
            key_padding_mask = (category_masks == 0)
            attn_output_cat, _ = self.multi_head_cat(batch_cat_queries,
                                                     seq_features_cat[1],
                                                     seq_features_cat[1],
                                                     key_padding_mask=key_padding_mask)
            # attn_output_cat with size [batch_size, 1, hidden_size]
            attn_cat = attn_output_cat.squeeze(1)
            cat_x = seq_features_cat[0] + attn_cat if self.att_first else attn_cat
            if cat_cls is None:
                return self.aspect_category(cat_x)
            else:
                return self.aspect_category(cat_x) + attn_cat.mm(cat_cls.transpose(0, 1))
        else:
            batch_size = len(seq_features_cat[1])
            cat_cls_tensor = cat_cls.unsqueeze(0).repeat(batch_size, 1, 1)
            if self.att_first:
                # NOTE: attn_mask contains a row with all True (masked) cause nan,
                # so mask attention outputs rather than inputs.
                # example_num, seq_len = category_masks.size()
                # mask = category_masks.unsqueeze(2).\
                #     repeat(1, 1, self.num_cat_labels).\
                #     repeat(1, self.multi_heads, 1).view(example_num*self.multi_heads,
                #                                         seq_len, self.num_cat_labels)
                # NOTE: torch and transforms define masks reversely.
                # attn_mask = (mask == 0)
                attn_output_cat, _ = self.multi_head_cat(seq_features_cat[1],
                                                         cat_cls_tensor,
                                                         cat_cls_tensor
                                                         # attn_mask= attn_mask
                                                         )
                # if np.isnan(torch.sum(attn_output_cat).item()):
                #     print('seq_features_sent', np.isnan(torch.sum(seq_features_cat[1]).item()))
                #     print('polarity_tensor', np.isnan(torch.sum(cat_cls_tensor).item()))
                # for param in self.multi_head_cat.named_parameters():
                #     print('cat_att', param)
                return self.aspect_category(attn_output_cat.mul(category_masks.unsqueeze(2)).sum(1) /
                                            category_masks.sum(1, True))
            else:
                attn_output_cat, _ = self.multi_head_cat(seq_features_cat[0].unsqueeze(1),
                                                         cat_cls_tensor,
                                                         cat_cls_tensor)
                return self.aspect_category(attn_output_cat.squeeze(1))

    def predict_sentiment(self, seq_features, sentiment_nums,
                          sentiment_idx, sentiment_masks):
        r"""
        Predict the sentiment of opinions.
        :param seq_features:
        :param sentiment_nums:
        :param sentiment_idx:
        :param sentiment_masks:
        :return: sentiment logits
        """
        seq_features_sent = self.get_features(seq_features,
                                              sentiment_nums,
                                              sentiment_idx,
                                              sentiment_masks)
        # sent_logits
        if self.multi_head_sent is None:
            return self.opinion_sentiment(seq_features_sent[0])

        polarity_embeddings = None if self.polarity_ids is None else \
            self.pretrained_model.embeddings.word_embeddings(self.polarity_ids)
        if self.sent_query is not None:
            batch_sent_queries = self.sent_query.repeat(len(sentiment_masks), 1, 1)
            # NOTE: torch and transforms define masks reversely.
            key_padding_mask = (sentiment_masks == 0)
            attn_output_sent, _ = self.multi_head_sent(batch_sent_queries,
                                                       seq_features_sent[1],
                                                       seq_features_sent[1],
                                                       key_padding_mask=key_padding_mask)
            # attn_output_sent with size [batch_size, 1, hidden_size]
            attn_sent = attn_output_sent.squeeze(1)
            sent_x = seq_features_sent[0] + attn_sent if self.att_first else attn_sent
            if polarity_embeddings is None:
                return self.opinion_sentiment(sent_x)
            else:
                return self.opinion_sentiment(sent_x) + attn_sent.mm(polarity_embeddings.transpose(0, 1))
        else:
            batch_size = len(seq_features_sent[1])
            polarity_tensor = polarity_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
            if self.att_first:
                attn_output_sent, _ = self.multi_head_sent(seq_features_sent[1],
                                                           polarity_tensor,
                                                           polarity_tensor)
                # if np.isnan(torch.sum(attn_output_sent).item()):
                #     print('seq_features_sent', np.isnan(torch.sum(seq_features_sent[1]).item()))
                #     print('polarity_tensor', np.isnan(torch.sum(polarity_tensor).item()))
                # for param in self.multi_head_sent.named_parameters():
                #     print('sent_att', param)
                return self.opinion_sentiment(attn_output_sent.mul(sentiment_masks.unsqueeze(2)).sum(1) /
                                              sentiment_masks.sum(1, True))
            else:
                attn_output_sent, _ = self.multi_head_sent(seq_features_sent[0].unsqueeze(1),
                                                           polarity_tensor,
                                                           polarity_tensor)
                return self.opinion_sentiment(attn_output_sent.squeeze(1))

    def predict_pair(self, seq_features, cat_sent_nums,
                     cat_sent_idx, cat_sent_masks):
        r"""
        Binary classification on aspect-opinion pairs.
        :param seq_features:
        :param cat_sent_nums:
        :param cat_sent_idx:
        :param cat_sent_masks:
        :return: pair logits
        """
        seq_features_pair = self.get_features(seq_features,
                                              cat_sent_nums,
                                              cat_sent_idx,
                                              cat_sent_masks)
        # pair_logits
        if self.multi_head_pair is None:
            return self.cat_sent_pair(seq_features_pair[0])

        batch_pair_queries = self.pair_query.repeat(len(cat_sent_masks), 1, 1)
        # NOTE: torch and transforms define masks reversely.
        key_padding_mask = (cat_sent_masks == 0)
        attn_output_pair, _ = self.multi_head_pair(batch_pair_queries,
                                                   seq_features_pair[1],
                                                   seq_features_pair[1],
                                                   key_padding_mask=key_padding_mask)
        # attn_output_pair with size [batch_size, 1, hidden_size]
        pair_x = seq_features_pair[0] + attn_output_pair.squeeze(1) if \
            self.att_first else attn_output_pair.squeeze(1)
        return self.cat_sent_pair(pair_x)

    def predict(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            bio_labels: Optional[torch.Tensor] = None,
            category_idx: Optional[torch.Tensor] = None,
            category_masks: Optional[torch.Tensor] = None,
            category_nums: Optional[torch.Tensor] = None,
            sentiment_idx: Optional[torch.Tensor] = None,
            sentiment_masks: Optional[torch.Tensor] = None,
            sentiment_nums: Optional[torch.Tensor] = None,
            cat_sent_idx: Optional[torch.Tensor] = None,
            cat_sent_masks: Optional[torch.Tensor] = None,
            cat_sent_nums: Optional[torch.Tensor] = None,
            positive_samples_list=None,
            crf_decode=False,
            neg_fn=None):
        r"""
        Multi-tasks: extract aspect-opinion (i.e., BIO),
        classify the category of aspects and sentiment of opinions,
        and whether a category-sentiment pair is valid.
        :param input_ids:
        :param attention_mask:
        :param bio_labels:
        :param category_idx:
        :param category_masks:
        :param category_nums:
        :param sentiment_idx:
        :param sentiment_masks:
        :param sentiment_nums:
        :param cat_sent_idx:
        :param cat_sent_masks:
        :param cat_sent_nums:
        :param positive_samples_list:
        :param crf_decode: effect only when use crf for bio.
        :param neg_fn: sampling negative function.
        :return:
        """
        # get the bert features of a given text
        # output: [batch_size, seq_length, hidden_size]
        seq_features = self.get_pretrained_features(input_ids, attention_mask)

        # BIO task
        bio_logits = self.predict_bio(
            seq_features, attention_mask, bio_labels, crf_decode)

        # aspect's category
        cat_logits = self.predict_category(
            seq_features, category_nums, category_idx, category_masks)

        # opinion's sentiment
        sent_logits = self.predict_sentiment(
            seq_features, sentiment_nums, sentiment_idx, sentiment_masks)

        # aspect-opinion pair
        pair_logits = self.predict_pair(
            seq_features, cat_sent_nums, cat_sent_idx, cat_sent_masks)

        if neg_fn is not None:
            assert self.neg_num != 0
            if isinstance(bio_logits, list):
                assert len(bio_logits) == 2
                if self.neg_num < 0:
                    # if negative sampling with random logits
                    bio_logits_x = torch.randn(list(bio_logits[1].size()) + [self.num_labels])
                    bio_logits_neg = bio_logits_x.max(-1)[1]
                else:
                    bio_logits_neg = bio_logits[1]
            else:
                bio_logits_neg = torch.randn(bio_logits.size()) if self.neg_num < 0 else bio_logits
            csp = neg_fn(bio_logits_neg, attention_mask, truth_filters=positive_samples_list)
            if len(csp.category_idx) > 0:
                neg_cat_logits = self.predict_category(
                    seq_features, csp.category_nums, csp.category_idx, csp.category_masks)
                cat_logits = torch.cat((cat_logits, neg_cat_logits))
            if len(csp.sentiment_idx) > 0:
                neg_sent_logits = self.predict_sentiment(
                    seq_features, csp.sentiment_nums, csp.sentiment_idx, csp.sentiment_masks)
                sent_logits = torch.cat((sent_logits, neg_sent_logits))
            neg_pair_size = len(csp.cat_sent_idx)
            if neg_pair_size > 0:
                # remove some negative samples to avoid out-of-memory
                if neg_pair_size > self.neg_max:
                    print('Too many random samples:', neg_pair_size)
                    bool_ix = torch.tensor([True]*self.neg_max + [False]*(neg_pair_size-self.neg_max))
                    bool_ix = bool_ix[torch.randperm(neg_pair_size)]
                    offset = 0
                    for i in range(len(csp.cat_sent_nums)):
                        new_num_x = sum(bool_ix[offset:offset + csp.cat_sent_nums[i]])
                        # 'csp.cat_sent_nums[i] is zero' will cause 'int'
                        new_num = new_num_x if isinstance(new_num_x, int) else new_num_x.item()
                        offset += csp.cat_sent_nums[i]
                        csp.cat_sent_nums[i] = new_num
                    assert sum(csp.cat_sent_nums).item() == self.neg_max
                    csp.cat_sent_idx = csp.cat_sent_idx[bool_ix]
                    csp.cat_sent_masks = csp.cat_sent_masks[bool_ix]
                neg_pair_logits = self.predict_pair(
                    seq_features, csp.cat_sent_nums, csp.cat_sent_idx, csp.cat_sent_masks)
                pair_logits = torch.cat((pair_logits, neg_pair_logits))

        # bio_logits with size(batch_num, seq_length, num_labels)
        # other logits with size(examples, *_num_labels)
        return bio_logits, cat_logits, sent_logits, pair_logits

    def get_features(self, seq_features, repeat_nums, idx, masks):
        r"""
        Extract features from BERT's outputs (i.e., seq_features) based
        :param seq_features: [batch_size, seq_length, hidden_size]
        :param repeat_nums: [batch_size], which specifies the repeating numbers of each example.
        :param idx: [examples, sequences], which specifies the sparse indices in seq_features.
        :param masks: [examples, sequences], which masks the paddings in idx.
        :return: [examples, hidden_size]
        """
        # batch_size <= examples due to repeat
        # seq_features_ext: [examples, seq_length, hidden_size]
        seq_features_ext = seq_features.repeat_interleave(
            repeat_nums, dim=0,
            output_size=repeat_nums.sum().item()
        )
        # idx: [examples, sequences]
        # index: [examples, sequences, 1]
        index = idx.unsqueeze(2)
        # sequences << seq_length
        # index: [examples, sequences, hidden_size]
        index = index.repeat(1, 1, self.hidden_size)
        # masks: [examples, sequences]
        # seq_features_red: [examples, sequences, hidden_size]
        seq_features_red = seq_features_ext.gather(1, index)
        # seq_features_mean: [examples, hidden_size]
        seq_features_mean = seq_features_red.mul(
            masks.unsqueeze(2)).sum(1) / masks.sum(1, True)
        return seq_features_mean, seq_features_red

    def forward(self, batch, device, crf_decode=False, neg_fn=None):
        r"""
        Shared by train and validate.
        :param batch:
        :param device:
        :param crf_decode: effect only when use crf for bio.
        :param neg_fn: sampling negative function.
        :return:
        """
        return self.predict(batch.input_ids.to(device),
                            batch.attention_masks.to(device),
                            batch.bio_labels.to(device),
                            batch.category_idx.to(device),
                            batch.category_masks.to(device),
                            batch.category_nums.to(device),
                            batch.sentiment_idx.to(device),
                            batch.sentiment_masks.to(device),
                            batch.sentiment_nums.to(device),
                            batch.cat_sent_idx.to(device),
                            batch.cat_sent_masks.to(device),
                            batch.cat_sent_nums.to(device),
                            batch.positive_samples_list,
                            crf_decode,
                            neg_fn)

    def _reorder_cache(self, past, beam_idx):
        pass

    def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
        pass

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        pass


class BertForACOS(ACOS, BertPreTrainedModel):
    r"""
    A class for sequence labeling using pre-trained bert.
    NOTE: two superclasses, ACOS must be before BertPreTrainedModel.
    """

    def __init__(self, config, num_cat_labels, num_sent_labels,
                 use_crf=False, frozen=False, query=0, multi_heads=0,
                 cat_input_ids=None, cat_attention_mask=None, polarity_ids=None,
                 multi_labels=False, neg_num=False, num_layers=2):
        r"""

        :param config:
        :param num_cat_labels:
        :param num_sent_labels:
        :param use_crf:
        :param frozen:
        :param query:
        :param multi_heads: 0 for no attention,
        >0 for attention before aggregation,
        <0 for attention after aggregation.
        :param cat_input_ids:
        :param cat_attention_mask:
        :param polarity_ids:
        :param multi_labels:
        :param neg_num:
        :param num_layers:
        """
        BertPreTrainedModel.__init__(self, config)
        self.bert = BertModel(config, add_pooling_layer=False)
        if frozen:
            for param in self.bert.parameters():
                param.requires_grad = False

        ACOS.__init__(self, self.bert, config, num_cat_labels, num_sent_labels,
                      use_crf, query, multi_heads, cat_input_ids, cat_attention_mask,
                      polarity_ids, multi_labels, neg_num, num_layers)

        # Initialize weights and apply final processing
        self.post_init()


class GPTForACOS(ACOS, GPT2PreTrainedModel):
    r"""
    A class for sequence labeling using pre-trained gpt.
    NOTE: two superclasses, ACOS must be before GPT2PreTrainedModel.
    """

    def __init__(self, config, num_cat_labels, num_sent_labels,
                 use_crf=False, frozen=False, query=0, multi_heads=0,
                 cat_input_ids=None, cat_attention_mask=None, polarity_ids=None,
                 multi_labels=False, neg_num=False, num_layers=2):
        r"""

        :param config:
        :param num_cat_labels:
        :param num_sent_labels:
        :param use_crf:
        :param frozen:
        :param query:
        :param multi_heads: 0 for no attention,
        >0 for attention before aggregation,
        <0 for attention after aggregation.
        :param cat_input_ids:
        :param cat_attention_mask:
        :param polarity_ids:
        :param multi_labels:
        :param neg_num:
        :param num_layers:
        """
        GPT2PreTrainedModel.__init__(self, config)
        gpt = GPT2Model(config)
        self.base_model_prefix = 'pretrained_model'
        if frozen:
            for param in gpt.parameters():
                param.requires_grad = False

        ACOS.__init__(self, gpt, config, num_cat_labels, num_sent_labels,
                      use_crf, query, multi_heads, cat_input_ids, cat_attention_mask,
                      polarity_ids, multi_labels, neg_num, num_layers)

        # Initialize weights and apply final processing
        self.post_init()
        