#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import re
import argparse
from datetime import datetime
import pandas as pd
from functools import partial
from torch.utils.data.dataloader import DataLoader
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import AdamW
from modelx.bertx import BertForACOS, GPTForACOS
from acos import *

parser = argparse.ArgumentParser(description='Hyper-parameters')
parser.add_argument('-exp', type=str, help='experiment name', default='')
parser.add_argument('-seed', type=int, help='random', default=0)
parser.add_argument('-ad', type=int, help='whether add dev data for training', default=0)
parser.add_argument('-data', type=str, help='dataset: laptop or rest16', default='laptop')
parser.add_argument('-pmn', type=str, help='pretrained model name', default='bert-large-uncased')
parser.add_argument('-epoch', type=int, help='epoch number', default=500)
parser.add_argument('-batch', type=int, help='batch size', default=32)
parser.add_argument('-lr', type=float, help='learning rate', default=1e-5)
parser.add_argument('-wd', type=float, help='weight decay', default=1e-2)
parser.add_argument('-crf', type=bool, help='add crf layer', default=False)
parser.add_argument('-frozen', type=bool, help='frozen bert', default=False)
parser.add_argument('-query', type=int, help='add queries for attention', default=1)
parser.add_argument('-mh', type=int, help='multi-heads', default=8)
parser.add_argument('-le', type=bool, help='label embedding for attention', default=True)
parser.add_argument('-ml', type=bool, help='multi-labels', default=True)
parser.add_argument('-mt', type=bool, help='multi-tasks', default=True)
parser.add_argument('-cv', type=bool, help='cross view', default=False)
parser.add_argument('-imp', type=bool, help='add implicit tokens', default=True)
parser.add_argument('-cls', type=bool, help='add cls token', default=False)
parser.add_argument('-ng', type=int, default=4096, help='negative sampling: 0 for None, neg-value for random sampling')
parser.add_argument('-nl', type=int, help='number of layers for pair classification', default=1)

# testing-related only
parser.add_argument('-iou', type=float, help='minimum iou', default=1.0)
parser.add_argument('-top', type=int, help='recommend top k pairs', default=0)
parser.add_argument('-jp', type=bool, help='joint probability, effect only when no multi-labels', default=False)
parser.add_argument('-path', type=str, help='the model file')

args = parser.parse_args()

seed = args.seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False

laptop_categories = ['MULTIMEDIA_DEVICES#PRICE', 'OS#QUALITY', 'SHIPPING#QUALITY', 'GRAPHICS#OPERATION_PERFORMANCE',
                     'CPU#OPERATION_PERFORMANCE',
                     'COMPANY#DESIGN_FEATURES', 'MEMORY#OPERATION_PERFORMANCE', 'SHIPPING#PRICE',
                     'POWER_SUPPLY#CONNECTIVITY', 'SOFTWARE#USABILITY',
                     'FANS&COOLING#GENERAL', 'GRAPHICS#DESIGN_FEATURES', 'BATTERY#GENERAL', 'HARD_DISC#USABILITY',
                     'FANS&COOLING#DESIGN_FEATURES',
                     'MEMORY#DESIGN_FEATURES', 'MOUSE#USABILITY', 'CPU#GENERAL', 'LAPTOP#QUALITY',
                     'POWER_SUPPLY#GENERAL', 'PORTS#QUALITY',
                     'KEYBOARD#PORTABILITY', 'SUPPORT#DESIGN_FEATURES', 'MULTIMEDIA_DEVICES#USABILITY',
                     'MOUSE#GENERAL', 'KEYBOARD#MISCELLANEOUS',
                     'MULTIMEDIA_DEVICES#DESIGN_FEATURES', 'OS#MISCELLANEOUS', 'LAPTOP#MISCELLANEOUS',
                     'SOFTWARE#PRICE', 'FANS&COOLING#OPERATION_PERFORMANCE',
                     'MEMORY#QUALITY', 'OPTICAL_DRIVES#OPERATION_PERFORMANCE', 'HARD_DISC#GENERAL',
                     'MEMORY#GENERAL', 'DISPLAY#OPERATION_PERFORMANCE',
                     'MULTIMEDIA_DEVICES#GENERAL', 'LAPTOP#GENERAL', 'MOTHERBOARD#QUALITY', 'LAPTOP#PORTABILITY',
                     'KEYBOARD#PRICE', 'SUPPORT#OPERATION_PERFORMANCE',
                     'GRAPHICS#GENERAL', 'MOTHERBOARD#OPERATION_PERFORMANCE', 'DISPLAY#GENERAL', 'BATTERY#QUALITY',
                     'LAPTOP#USABILITY', 'LAPTOP#DESIGN_FEATURES',
                     'PORTS#CONNECTIVITY', 'HARDWARE#QUALITY', 'SUPPORT#GENERAL', 'MOTHERBOARD#GENERAL',
                     'PORTS#USABILITY', 'KEYBOARD#QUALITY', 'GRAPHICS#USABILITY',
                     'HARD_DISC#PRICE', 'OPTICAL_DRIVES#USABILITY', 'MULTIMEDIA_DEVICES#CONNECTIVITY',
                     'HARDWARE#DESIGN_FEATURES', 'MEMORY#USABILITY',
                     'SHIPPING#GENERAL', 'CPU#PRICE', 'Out_Of_Scope#DESIGN_FEATURES', 'MULTIMEDIA_DEVICES#QUALITY',
                     'OS#PRICE', 'SUPPORT#QUALITY',
                     'OPTICAL_DRIVES#GENERAL', 'HARDWARE#USABILITY', 'DISPLAY#DESIGN_FEATURES', 'PORTS#GENERAL',
                     'COMPANY#OPERATION_PERFORMANCE',
                     'COMPANY#GENERAL', 'Out_Of_Scope#GENERAL', 'KEYBOARD#DESIGN_FEATURES',
                     'Out_Of_Scope#OPERATION_PERFORMANCE',
                     'OPTICAL_DRIVES#DESIGN_FEATURES', 'LAPTOP#OPERATION_PERFORMANCE', 'KEYBOARD#USABILITY',
                     'DISPLAY#USABILITY', 'POWER_SUPPLY#QUALITY',
                     'HARD_DISC#DESIGN_FEATURES', 'DISPLAY#QUALITY', 'MOUSE#DESIGN_FEATURES', 'COMPANY#QUALITY',
                     'HARDWARE#GENERAL', 'COMPANY#PRICE',
                     'MULTIMEDIA_DEVICES#OPERATION_PERFORMANCE', 'KEYBOARD#OPERATION_PERFORMANCE',
                     'SOFTWARE#PORTABILITY', 'HARD_DISC#OPERATION_PERFORMANCE',
                     'BATTERY#DESIGN_FEATURES', 'CPU#QUALITY', 'WARRANTY#GENERAL', 'OS#DESIGN_FEATURES',
                     'OS#OPERATION_PERFORMANCE', 'OS#USABILITY',
                     'SOFTWARE#GENERAL', 'SUPPORT#PRICE', 'SHIPPING#OPERATION_PERFORMANCE', 'DISPLAY#PRICE',
                     'LAPTOP#PRICE', 'OS#GENERAL', 'HARDWARE#PRICE',
                     'SOFTWARE#DESIGN_FEATURES', 'HARD_DISC#MISCELLANEOUS', 'PORTS#PORTABILITY',
                     'FANS&COOLING#QUALITY', 'BATTERY#OPERATION_PERFORMANCE',
                     'CPU#DESIGN_FEATURES', 'PORTS#OPERATION_PERFORMANCE', 'SOFTWARE#OPERATION_PERFORMANCE',
                     'KEYBOARD#GENERAL', 'SOFTWARE#QUALITY',
                     'LAPTOP#CONNECTIVITY', 'POWER_SUPPLY#DESIGN_FEATURES', 'HARDWARE#OPERATION_PERFORMANCE',
                     'WARRANTY#QUALITY', 'HARD_DISC#QUALITY',
                     'POWER_SUPPLY#OPERATION_PERFORMANCE', 'PORTS#DESIGN_FEATURES', 'Out_Of_Scope#USABILITY']
rest16_categories = ['RESTAURANT#GENERAL', 'SERVICE#GENERAL', 'FOOD#GENERAL', 'FOOD#QUALITY', 'FOOD#STYLE_OPTIONS',
                     'DRINKS#STYLE_OPTIONS', 'DRINKS#PRICES',
                     'AMBIENCE#GENERAL', 'RESTAURANT#PRICES', 'FOOD#PRICES', 'RESTAURANT#MISCELLANEOUS',
                     'DRINKS#QUALITY', 'LOCATION#GENERAL']


def training():
    # hyper-parameters
    # training-related
    data = args.data
    model_name = args.pmn
    epoch_num = args.epoch
    batch_size = args.batch
    learning_rate = args.lr
    weight_decay = args.wd
    use_crf = args.crf
    frozen = args.frozen
    multi_heads = args.mh
    multi_labels = args.ml
    multi_tasks = args.mt
    cross_view = args.cv
    neg_sampling = args.ng
    add_implicit_tokens = args.imp
    cls_position = [0] if args.cls else []
    # both cannot be true at the same time
    assert (frozen and add_implicit_tokens) is False
    assert (cls_position == [0]) or (cls_position == [])
    # must add implicit tokens or cls position
    assert add_implicit_tokens or (cls_position == [0])
    assert multi_labels or (neg_sampling == 0)
    assert multi_labels or (cross_view is False)
    # args.le  affects only when args.query is True
    assert args.query or args.le

    # testing-related
    iou_min = args.iou
    top_k_pairs = args.top
    joint_prob = args.jp

    cats = laptop_categories if data == 'laptop' else rest16_categories
    data_folder = 'Laptop' if data == 'laptop' else 'Restaurant'

    # process dataset
    bio_label_set = BIOLabelSet(labels=["Aspect", "Opinion"])
    cat_sent_label_set = CategorySentimentLabelSet(cats)
    train_raw = acos_data_to_ddi_format(f'./data/{data_folder}-ACOS/{data}_quad_train.tsv',
                                        add_implicit_tokens)
    dev_raw = acos_data_to_ddi_format(f'./data/{data_folder}-ACOS/{data}_quad_dev.tsv',
                                      add_implicit_tokens)
    test_raw = acos_data_to_ddi_format(f'./data/{data_folder}-ACOS/{data}_quad_test.tsv',
                                       add_implicit_tokens)
    if args.ad:
        train_raw.extend(dev_raw)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    if add_implicit_tokens:
        tokenizer.add_tokens(['[IMPLICIT_ASPECT]', '[IMPLICIT_OPINION]'], special_tokens=True)
    if args.le:
        cat_text = [cat.replace('#', ' ') for cat in cats]
        cat_encoded_input = tokenizer(cat_text, padding=True, return_tensors='pt')
        polarity_tokens = ['negative', 'neutral', 'positive']
        polarity_ids = torch.tensor(tokenizer.convert_tokens_to_ids(polarity_tokens))
    else:
        cat_encoded_input = None
        polarity_ids = None

    train_ds = TrainingDataset(data=train_raw, tokenizer=tokenizer,
                               bio_label_set=bio_label_set,
                               cat_sent_label_set=cat_sent_label_set,
                               multi_labels=multi_labels,
                               add_implicit_tokens=add_implicit_tokens,
                               cls_position=cls_position)
    dev_ds = TrainingDataset(data=dev_raw, tokenizer=tokenizer,
                             bio_label_set=bio_label_set,
                             cat_sent_label_set=cat_sent_label_set,
                             multi_labels=multi_labels,
                             add_implicit_tokens=add_implicit_tokens,
                             cls_position=cls_position)
    test_ds = TrainingDataset(data=test_raw, tokenizer=tokenizer,
                              bio_label_set=bio_label_set,
                              cat_sent_label_set=cat_sent_label_set,
                              multi_labels=multi_labels,
                              add_implicit_tokens=add_implicit_tokens,
                              cls_position=cls_position)

    num_cat_labels = len(cat_sent_label_set.id_to_label) \
        if multi_labels else 0
    num_sent_labels = 3 if multi_labels else 0

    batch_fn = partial(TrainingBatch,
                       tokenizer=tokenizer,
                       num_cat_labels=num_cat_labels,
                       num_sent_labels=num_sent_labels)
    train_loader = DataLoader(
        train_ds,
        collate_fn=batch_fn,
        batch_size=batch_size,
        shuffle=True,
    )
    dev_loader = DataLoader(
        dev_ds,
        collate_fn=batch_fn,
        batch_size=1,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        collate_fn=batch_fn,
        batch_size=1,
        shuffle=False,
    )

    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device', device)
    cat_input_ids = cat_encoded_input.input_ids.to(device) if cat_encoded_input else None
    cat_attention_mask = cat_encoded_input.attention_mask.to(device) if cat_encoded_input else None
    polarity_ids = polarity_ids.to(device) if polarity_ids is not None else None

    if args.path:
        # incremental training
        model = torch.load(args.path).to(device)
    else:
        if model_name.startswith('bert'):
            class_name = 'BertForACOS'
        elif model_name.startswith('gpt'):
            class_name = 'GPTForACOS'
        else:
            print('Please input the correct pretrained model name!')
            exit(1)

        model = eval(class_name).from_pretrained(
            model_name,
            num_labels=len(bio_label_set.id_to_label),
            num_cat_labels=len(cat_sent_label_set.id_to_label),
            num_sent_labels=3,
            use_crf=use_crf,
            frozen=frozen,
            query=args.query,
            multi_heads=multi_heads,
            cat_input_ids=cat_input_ids,
            cat_attention_mask=cat_attention_mask,
            polarity_ids=polarity_ids,
            multi_labels=multi_labels,
            neg_num=neg_sampling,
            num_layers=args.nl
        )
        if add_implicit_tokens:
            model.resize_token_embeddings(len(tokenizer))
        model = model.to(device)

    neg_fn = partial(CategorySentimentPairs,
                     bio_label_set=bio_label_set,
                     device=device,
                     add_implicit_tokens=add_implicit_tokens,
                     cls_position=cls_position) if neg_sampling else None

    # define loss and optimizer
    bio_loss_fn = CrossEntropyLoss(weight=train_ds.bio_class_weight.to(device))
    loss_fn = BCEWithLogitsLoss() if multi_labels else CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # train and test
    if args.path:
        log_dir, pt_name = os.path.split(args.path)
        start_epoch = int(pt_name.split('_')[1].split('.')[0]) + 1
    else:
        start_epoch = 1
        path_rstr = r"[\/\\\:\*\?\"\<\>\|\n\t ]"
        comment_suffix = f'{model_name}_{data}_en{epoch_num}_ad{args.ad}_bs{batch_size}' \
                         f'_lr{learning_rate}_wd{weight_decay}_ng{neg_sampling}' \
                         f'_crf{int(use_crf)}_fz{int(frozen)}_q{int(args.query)}_mh{multi_heads}' \
                         f'_le_{int(args.le)}_ml{int(multi_labels)}_mt{int(multi_tasks)}_cv{int(cross_view)}' \
                         f'_imp{int(add_implicit_tokens)}_cls{int(args.cls)}_nl{args.nl}'
        comment_suffix = re.sub(path_rstr, '', comment_suffix)
        dt = datetime.today().strftime('d%m%dt%H%M%S')
        log_dir = f'runs/{dt}{args.exp}_{seed}_{comment_suffix}'
    writer = SummaryWriter(log_dir)
    df = pd.DataFrame(columns=['epoch', 'train_loss', 'dev_pre', 'dev_rec',
                               'dev_f1', 'precision', 'recall', 'f1',
                               'f1-00', 'f1-01', 'f1-10', 'f1-11'])
    global_step = 0
    f1_dict = {}
    print_period = 1 if str(device) == 'cpu' else 10
    for epoch in range(start_epoch, epoch_num + 1):
        global_step, train_loss = train(
            train_loader, model, neg_fn, bio_loss_fn, loss_fn, optimizer,
            device, global_step, writer, use_crf, multi_tasks, cross_view, print_period)
        writer.add_scalar('Train/loss/epoch', train_loss, epoch)
        print(f'epoch: {epoch}, train_loss: {train_loss}')
        # dev_loss, accuracy = validate(dev_loader, model, bio_loss_fn, loss_fn, device,
        #                               use_crf, multi_labels, multi_tasks)
        dev_pre, dev_rec, dev_f1, _ = test(dev_loader, model, bio_label_set, device,
                                           iou_min, top_k_pairs, joint_prob, multi_labels,
                                           None, None, None, add_implicit_tokens, cls_position)

        precision, recall, f1, f1x = test(test_loader, model, bio_label_set, device,
                                          iou_min, top_k_pairs, joint_prob, multi_labels,
                                          None, None, None, add_implicit_tokens, cls_position)
        f1_dict[epoch] = f1
        writer.add_scalar('Dev/precision/epoch', dev_pre, epoch)
        writer.add_scalar('Dev/recall/epoch', dev_rec, epoch)
        writer.add_scalar('Dev/f1/epoch', dev_f1, epoch)
        writer.add_scalar('Test/precision/epoch', precision, epoch)
        writer.add_scalar('Test/recall/epoch', recall, epoch)
        writer.add_scalar('Test/f1/epoch', f1, epoch)
        print(f'epoch: {epoch}, dev_pre: {dev_pre}, dev_rec: {dev_rec}, dev_f1: {dev_f1}')
        print(f'epoch: {epoch}, precision: {precision}, recall: {recall}, f1: {f1}')
        df = df.append({'epoch': epoch,
                        'train_loss': train_loss,
                        'dev_pre': round(dev_pre, 4),
                        'dev_rec': round(dev_rec, 4),
                        'dev_f1': round(dev_f1, 4),
                        'precision': round(precision, 4),
                        'recall': round(recall, 4),
                        'f1': round(f1, 4),
                        'f1-00': round(f1x[0, 0], 4),
                        'f1-01': round(f1x[0, 1], 4),
                        'f1-10': round(f1x[1, 0], 4),
                        'f1-11': round(f1x[1, 1], 4)},
                       ignore_index=True)
        # only save the top-3 models according to accuracy
        if len(f1_dict) > 3:
            min_epoch = min(f1_dict, key=f1_dict.get)
            f1_dict.pop(min_epoch)
            if min_epoch != epoch:
                # delete the old model
                os.remove(f'{writer.log_dir}/model_{min_epoch}.pt')
                # save the new model
                torch.save(model, f'{writer.log_dir}/model_{epoch}.pt')
            else:
                # no need to save or delete
                pass
        else:
            # less than 3 models, so save anyway
            torch.save(model, f'{writer.log_dir}/model_{epoch}.pt')
        if np.isnan(train_loss):
            break
        # rewrite for each epoch
        df['epoch'] = df['epoch'].astype(np.int)
        df.to_csv(f'{writer.log_dir}/metrics.csv', index=False)
    print('top-3 f1:', f1_dict)
    writer.close()


if __name__ == '__main__':
    training()
