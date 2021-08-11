from __future__ import annotations

import json
import pickle
import random
from itertools import product

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab

from util import doc_pipeline, query_pipeline, load_processed_data


class RankNetLSTM(nn.Module):
    """RankNet model using LSTM encoders and GLoVe embeddings.

    RankNetLSTM is based on RankNet, which is trained with pairwise data.

    During inference, given a document and a query, RankNetLSTM will first embed them   \\
    using GLoVe embeddings before passing them to their respective LSTM encoders: one   \\
    for documents and one for queries. The encoded representations of the query and     \\
    document are then concatenated and passed through a single feedforward layer to     \\
    generate the relevance score for the document and query.

    RankNetLSTM uses pairwise data for training. Each training sample should contain a  \\
    relevant document, irrelevant document, and a query. During training, the relevance \\
    scores of both documents with respect to the query are computed and passed into the \\
    RankNet loss function. This function calculates the loss for the training sample    \\
    based on the difference of the two scores, which can be backpropagated across the   \\
    entire network to train the model.

    Arguments:
        vocab: Vocab from the preprocessed data

    ----------
    DISCLAIMER
    ----------
    This model does not work very well with pseudo-labelled small datasets. Based on my \\
    experience training it on a dataset containing 1000 samples with pseudo-labels from \\
    BM25, the model always produced the same scores for each document regardless of the \\
    query, which suggests overfitting to the relevant documents in the training dataset \\
    and inability to generalize. Engineers who want to experiment with this approach in \\
    the future might want to use a larger, human-labelled dataset.
    """

    def __init__(self, vocab: Vocab):
        super(RankNetLSTM, self).__init__()
        vocab.load_vectors('glove.6B.50d')
        self.embed = nn.Embedding.from_pretrained(
            vocab.vectors, freeze=True, padding_idx=vocab.stoi['<pad>'])
        self.query_lstm = nn.LSTM(
            input_size=50, hidden_size=50, bias=True, batch_first=True)
        self.doc_lstm = nn.LSTM(
            input_size=50, hidden_size=50, bias=True, batch_first=True)
        self.linear = nn.Linear(in_features=100, out_features=1, bias=True)

    def forward(self, query, doc):
        query = self.query_lstm(self.embed(query))[0].mean(dim=1)
        doc = self.doc_lstm(self.embed(doc))[0].mean(dim=1)
        query_doc = torch.cat([query, doc], dim=1)
        return self.linear(query_doc)


def rank_net_loss(pos_score, neg_score, gamma):
    return torch.log(1 + torch.exp(-gamma * (pos_score - neg_score))).mean()


def main(
    labelled_data_path: str = 'files/test_data_labelled.json',
    processed_data_path: str = 'files/test_data_processed.pickle',
    output_path: str = 'files/ranknet_lstm.pt',
    query_len: int = 50,
    doc_len: int = 200,
    batch_size: int = 128,
    num_epochs: int = 5,
    gamma: float = 1.0,
    learning_rate: float = 0.01,
):
    """Train a RankNetLSTM model.

    Arguments:
        labelled_data_path: path to load labelled data
        processed_data_path: path to load processed data
        output_path: path to save RankNetLSTM model state
        query_len: query length to trim/pad to
        doc_len: document length to trim/pad to
        batch_size: batch size
        num_epochs: number of training epochs
        gamma: diff factor for rank net loss
        learning_rate: learning rate for optimizer
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    data = load_processed_data(processed_data_path)
    vocab = data['vocab']
    docs_map = {doc.id: doc for doc in data['docs']}
    with open(labelled_data_path, 'r') as fp:
        train_data = json.load(fp)

    # Split data
    train_data, test_data = train_test_split(train_data, test_size=1 / 10)
    train_data, valid_data = train_test_split(train_data, test_size=1 / 9)

    # Collate function
    def collate_batch(batch):
        collated = []
        for query, relevant_ids, irrelevant_ids in batch:
            query = query_pipeline(query, vocab, query_len)
            for pos_id, neg_id in product(relevant_ids, irrelevant_ids):
                pos_doc = doc_pipeline(docs_map[pos_id], vocab, doc_len)
                neg_doc = doc_pipeline(docs_map[neg_id], vocab, doc_len)
                collated.append((query, pos_doc, neg_doc))

        # Shuffle samples
        random.shuffle(collated)
        query, pos_doc, neg_doc = zip(*collated)

        # Load into tensors and send to GPU
        query = torch.stack(query).to(device)
        pos_doc = torch.stack(pos_doc).to(device)
        neg_doc = torch.stack(neg_doc).to(device)
        return query, pos_doc, neg_doc

    # Dataloader
    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size,
                                  shuffle=False, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_data, batch_size=batch_size,
                                 shuffle=False, collate_fn=collate_batch)

    # Instantiate model
    model = RankNetLSTM(vocab).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}.', end='')
        train_loss = 0.0
        model.train()
        for idx, (query, pos_doc, neg_doc) in enumerate(train_dataloader):
            optimizer.zero_grad()
            pos_score = model(query, pos_doc)
            neg_score = model(query, neg_doc)
            loss = rank_net_loss(pos_score, neg_score, gamma)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
            print(f'\rEpoch {epoch+1}/{num_epochs}. '
                  f'Batch {idx+1}/{len(train_dataloader)}. '
                  f'Training loss: {train_loss / (idx + 1)}', end='')

        # Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for query, pos_doc, neg_doc in valid_dataloader:
                pos_score = model(query, pos_doc)
                neg_score = model(query, neg_doc)
                loss = rank_net_loss(pos_score, neg_score, gamma)
                val_loss += loss.detach().item()
        print(f'\nEpoch {epoch+1}/{num_epochs}. '
              f'Validation loss: {val_loss / len(valid_dataloader)}')

    # Testing
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for query, pos_doc, neg_doc in test_dataloader:
            pos_score = model(query, pos_doc)
            neg_score = model(query, neg_doc)
            loss = rank_net_loss(pos_score, neg_score, gamma)
            test_loss += loss.detach().item()
    print(f'Test loss: {test_loss / len(test_dataloader)}')

    # Save model
    torch.save(model.state_dict(), output_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train a RankNetLSTM model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-i', '--in', default='files/test_data_labelled.json',
        help='path to load labelled data',
        metavar="PATH", dest='labelled_data_path')
    parser.add_argument(
        '-d', '--data', default='files/test_data_processed.pickle',
        help='path to load processed data',
        metavar="PATH", dest='processed_data_path')
    parser.add_argument(
        '-o', '--out', default='files/ranknet_lstm.pt',
        help='path to save RankNetLSTM model state',
        metavar="PATH", dest='output_path')
    parser.add_argument(
        '-q', '--query-len', default=50, type=int,
        help='query length to trim/pad to',
        metavar='Q', dest='query_len')
    parser.add_argument(
        '-D', '--doc-len', default=200, type=int,
        help='document length to trim/pad to',
        metavar='D', dest='doc_len')
    parser.add_argument(
        '-b', '--batch', default=128, type=int,
        help='batch size',
        metavar='B', dest='batch_size')
    parser.add_argument(
        '-e', '--epoch', default=5, type=int,
        help='number of training epochs',
        metavar='E', dest='num_epochs')
    parser.add_argument(
        '-g', '--gamma', default=1.0, type=float,
        help='diff factor for rank net loss',
        metavar='G', dest='gamma')
    parser.add_argument(
        '-l', '--lr', default=0.01, type=float,
        help='learning rate for optimizer',
        metavar='L', dest='learning_rate')
    args = parser.parse_args()
    main(**vars(args))
