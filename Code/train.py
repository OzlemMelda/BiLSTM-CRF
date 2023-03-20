import os
import numpy as np
import torch
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader
import subprocess

eval_path = "./Data/evaluation"
eval_script = './conlleval.py'


if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_path):
    os.makedirs(eval_path)


def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def evaluating(model, datas: DataLoader, best_F: float, use_gpu: bool, id_to_tag: Dict) -> Tuple[float, float, bool]:
    """

    This function will evaluate the model on the given dataset.

    Args:
        model (_type_): 
        datas (DataLoader): the batched dataset to evaluate on
        best_F (float): _description_
        use_gpu (bool): this is a flag to indicate whether to use GPU or not
        id_to_tag (Dict): Dictionary that maps tag ids to tag names

    Returns:
        Tuple[float, float, bool]: (new_F, best_F, save)
    """
    prediction = []
    save = False
    new_F = 0.0
    for batch in datas:
        word_ids, words, tag_ids, tags, seq_len, valid_mask = batch
        if use_gpu:
            val, out = model.inference(word_ids.cuda())
        else:
            val, out = model.inference(word_ids)

        predicted_id = out.cpu()
        
        flat_tags = [item for sublist in tags for item in sublist]
        flat_words = [item for sublist in word_ids.tolist() for item in sublist]
        
        masked_predicted_id = torch.masked_select(predicted_id, valid_mask).tolist()
        predicted_tags = [id_to_tag[p_id] for p_id in masked_predicted_id]

        output = [str(w) + ' ' + i + ' ' + j for w, i, j in zip(flat_words, flat_tags, predicted_tags)]
        prediction = prediction + output
      
  
    predf = eval_path + '/pred'
    scoref = eval_path + '/score'
    

    with open(predf, 'w', encoding='utf-8') as f:
        f.write('\n'.join(prediction))

    subprocess.call('python %s < %s > %s' % (eval_script, predf, scoref), shell=True)
    
    eval_lines = [l.rstrip() for l in open(scoref, 'r', encoding='utf8')]

    for i, line in enumerate(eval_lines):
        print(line)
        if i == 1:
            new_F = float(line.strip().split()[-1])
            if new_F > best_F:
                best_F = new_F
                save = True
                print('the best F is ', new_F)
    return best_F, new_F, save


def train(model, epochs: int, train_data: DataLoader, dev_data: DataLoader, test_data: DataLoader, use_gpu: bool, id_to_tag: Dict) -> None:
    """

    This function will train the model for the given number of epochs.

    Args:
        model (Any): This is the model to train
        epochs (int): Number of epochs to train for
        train_data (DataLoader): the batched training dataset
        dev_data (DataLoader): the batched development dataset
        test_data (DataLoader): the batched test dataset
        use_gpu (bool): this is a flag to indicate whether to use GPU or not
        id_to_tag (Dict): Dictionary that maps tag ids to tag names

    Returns:
        None
    """
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    loss = 0.0
    best_dev_F = -1.0
    best_test_F = -1.0
    all_F = [[0, 0]]
    plot_every = 200
    eval_every = 200
    count = 0
    sys.stdout.flush()

    model.train(True)

    for epoch in range(1, epochs):
        
        for iter, batch in tqdm(enumerate(train_data)):
            word_ids, word, tag_ids, tag, seq_len, mask_list = batch
            model.zero_grad()
            count += 1
            
            if use_gpu:
                neg_log_likelihood = model(
                    word_ids.cuda(),
                    tag_ids.cuda(),
                )
            else:
                neg_log_likelihood = model(word_ids, tag_ids)
            loss += neg_log_likelihood.data.item()
            neg_log_likelihood.backward()
            optimizer.step()

            if count % plot_every == 0:
                loss /= plot_every
                losses.append(loss)
                print(loss)

            if (count % (eval_every) == 0 and count > (eval_every * 20) or count % (eval_every * 4) == 0 and count <
                    (eval_every * 20)):
                model.train(False)
                best_test_F, new_test_F, _ = evaluating(
                    model, test_data, best_test_F, use_gpu, id_to_tag)
                best_dev_F, new_dev_F, save = evaluating(
                    model, dev_data, best_dev_F, use_gpu, id_to_tag)
                if save:
                    torch.save(model, './Data/evaluation/model.pt')
                sys.stdout.flush()

                all_F.append([new_dev_F, new_test_F])
                model.train(True)

            if count % len(train_data) == 0:
                adjust_learning_rate(
                    optimizer, lr=learning_rate / (1 + 0.05 * count / len(train_data)))

    plt.plot(losses)
    plt.show()
