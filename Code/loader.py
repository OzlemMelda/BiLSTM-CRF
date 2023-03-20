import model as model
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import torch


def load_sentences(path):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in open(path, 'r', encoding='utf-8'):
        line = line.rstrip()
        if not line and len(sentence) > 0:
            sentences.append(sentence)
            sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences


def word_mapping(train, test=[], dev=[]):
    """
    Create a mapping of words.
    """

    word2id = {}
    word2id[model.START_TAG] = len(word2id)
    word2id[model.STOP_TAG] = len(word2id)
    word2id['<PAD>'] = len(word2id)
    word2id['<UNK>'] = len(word2id)

    sentences = train + test + dev

    words = [x[0].lower() for s in sentences for x in s]
    for word in words:
        if word not in word2id:
            word2id[word] = len(word2id)

    id2word = {v: k for k, v in word2id.items()}

    return word2id, id2word


def tag_mapping(train, test=[], dev=[]):
    """
    Create a mapping of tags.
    """
    sentences = train + test + dev
    tags = [word[-1] for s in sentences for word in s]

    tag2id = {}
    tag2id['<PAD>'] = len(tag2id)
    tag2id[model.START_TAG] = len(tag2id)
    tag2id[model.STOP_TAG] = len(tag2id)
    for tag in tags:
        if tag not in tag2id:
            tag2id[tag] = len(tag2id)
    id2tag = {v: k for k, v in tag2id.items()}

    return tag2id, id2tag


class Instance:
    """
    This class represents a single instance of a sentence.
    It contains the word ids, the word strings, the tag ids, the tag strings and the sequence length.

    """

    def __init__(self, word_ids, words, tag_ids, tags, seq_len):
        self.word_ids = word_ids
        self.words = words
        self.tag_ids = tag_ids
        self.tags = tags
        self.seq_len = seq_len


class NERDataset(Dataset):

    """
    This class represents a dataset of sentences.
    This will be used by the DataLoader to create batches.
    """

    def __init__(self, data, word2id, tag2id):
        self.word2id = word2id
        self.tag2id = tag2id
        self._create_instances(data)

    def _create_instances(self, data) -> List[Instance]:
        """
        This function will create instances from the data.

        Args:
            data (List[List[str, str]]): It is a list of sentences. For each sentence, the first element is the word and the last element is the tag.

        Returns:

            List[Instance]: It is a list of instances.
        """
        self.instances = []
        for sentence in data:
            words = [x[0] for x in sentence]
            tags = [x[-1] for x in sentence]
            seq_len = len(words)
            word_ids = [self.word2id[model.START_TAG]] + [self.word2id.get(word.lower(
            ), self.word2id['<UNK>']) for word in words] + [self.word2id[model.STOP_TAG]]
            tag_ids = [self.tag2id[model.START_TAG]] + [self.tag2id[tag]
                                                        for tag in tags] + [self.tag2id[model.STOP_TAG]]
            self.instances.append(
                Instance(word_ids, words, tag_ids, tags, seq_len))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        return instance.word_ids, instance.words, instance.tag_ids, instance.tags, instance.seq_len

    def collate_fn(self, batch: List[Dict]) -> Tuple[torch.Tensor, List, torch.Tensor, List, List, torch.Tensor]:
        """
        This function will help the bached data to have the same length. It will pad the data with 0s. In addition, for those padded positions, it will create masks.

        Args:
            batch (List[Tuple]): each tuple contains the word ids, the word strings, the tag ids, the tag strings and the sequence length.

        Returns:
            Tuple[torch.Tensor, List, torch.Tensor, List, List, torch.Tensor]: It returns a tuple of 6 elements. The first element is a tensor of word ids. The second element is a list of word strings. The third element is a tensor of tag ids. The fourth element is a list of tag strings. The fifth element is a list of sequence lengths. The sixth element is a tensor of masks.
        """
        length_list = [instance[4] for instance in batch]
        max_len = max(length_list)
        word_ids_list = []
        tag_ids_list = []
        mask_list = []
        word_list = [instance[1] for instance in batch]
        tag_list = [instance[3] for instance in batch]
        for i, instance in enumerate(batch):
            pad_len = max_len - instance[4]
            word_ids = torch.tensor(
                instance[0] + [self.word2id['<PAD>']] * pad_len, dtype=torch.long)
            word_ids_list.append(word_ids)
            tag_ids = torch.tensor(
                instance[2] + [self.tag2id['<PAD>']] * pad_len, dtype=torch.long)
            tag_ids_list.append(tag_ids)
            mask_list.append(torch.ByteTensor(
                [0] + [1] * instance[4] + [0] + [0] * pad_len))
        return torch.stack(word_ids_list), word_list, torch.stack(tag_ids_list), tag_list, length_list, torch.stack(mask_list)
