import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from model import HierarchialAttentionNetwork
from utils import *
from datasets import HANDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

log_name = 'chong'
if not os.path.exists(log_name):
    os.mkdir(log_name)

writer = {
    'train_loss': SummaryWriter(f"./{log_name}/epoch_loss"),
    'train_acc': SummaryWriter(f"./{log_name}/epoch_acc"),
    'batch_loss': SummaryWriter(f"./{log_name}/batch_loss"),
    'batch_acc': SummaryWriter(f"./{log_name}/batch_acc"),
    'val_loss': SummaryWriter(f"./{log_name}/val_loss"),
    'val_acc': SummaryWriter(f"./{log_name}/val_acc"),
    'batch_lr': SummaryWriter(f"./{log_name}/batch_lr"),
    'graph': SummaryWriter(f"./{log_name}/graph")
}


# Data parameters
data_folder = 'dataset'
# path to pre-trained word2vec embeddings
word2vec_file = os.path.join(data_folder, 'word2vec_model')
with open(os.path.join(data_folder, 'word_map.json'), 'r') as j:
    word_map = json.load(j)

# Model parameters
n_classes = len(label_map)
word_rnn_size = 50  # word RNN size
sentence_rnn_size = 50  # character RNN size
word_rnn_layers = 1  # number of layers in character RNN
sentence_rnn_layers = 1  # number of layers in word RNN
# size of the word-level attention layer (also the size of the word context vector)
word_att_size = 100
# size of the sentence-level attention layer (also the size of the sentence context vector)
sentence_att_size = 100
dropout = 0.3  # dropout
fine_tune_word_embeddings = True  # fine-tune word embeddings?

# Training parameters
start_epoch = 0  # start at this epoch
batch_size = 256  # batch size
lr = 1e-3  # learning rate
momentum = 0.9  # momentum
workers = 6  # number of workers for loading data in the DataLoader
epochs = 1  # number of epochs to run
grad_clip = None  # clip gradients at this value
print_freq = 10  # print training or validation status every __ batches
checkpoint = None  # path to model checkpoint, None if none

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# set to true only if inputs to model are fixed size; otherwise lot of computational overhead
cudnn.benchmark = True


def main():
    """
    Training and validation.
    """
    global checkpoint, start_epoch, word_map

    max_acc = -float('inf')
    # Initialize model or load checkpoint
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        word_map = checkpoint['word_map']
        start_epoch = checkpoint['epoch'] + 1
        print(
            '\nLoaded checkpoint from epoch %d.\n' % (start_epoch - 1))
    else:
        embeddings, emb_size = load_word2vec_embeddings(
            word2vec_file, word_map)  # load pre-trained word2vec embeddings

        model = HierarchialAttentionNetwork(n_classes=n_classes,
                                            vocab_size=len(word_map),
                                            emb_size=emb_size,
                                            word_rnn_size=word_rnn_size,
                                            sentence_rnn_size=sentence_rnn_size,
                                            word_rnn_layers=word_rnn_layers,
                                            sentence_rnn_layers=sentence_rnn_layers,
                                            word_att_size=word_att_size,
                                            sentence_att_size=sentence_att_size,
                                            dropout=dropout)
        model.sentence_attention.word_attention.init_embeddings(
            embeddings)  # initialize embedding layer with pre-trained embeddings
        model.sentence_attention.word_attention.fine_tune_embeddings(
            fine_tune_word_embeddings)  # fine-tune
        optimizer = optim.Adam(params=filter(
            lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Loss functions
    criterion = nn.CrossEntropyLoss()

    # Move to device
    model = model.to(device)
    criterion = criterion.to(device)

    # DataLoaders
    train_ratio = 0.6
    val_ratio = 0.3
    test_ratio = 0.1

    # 划分数据集大小
    dataset_size = len(HANDataset(data_folder, 'train'))
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    # 建立划分索引
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, dataset_size))

    # 划分数据
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # 对应的数据加载
    train_loader = DataLoader(
        dataset=HANDataset(data_folder, 'train'),
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset=HANDataset(data_folder, 'train'),
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=HANDataset(data_folder, 'train'),
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=workers,
        pin_memory=True
    )
    # train_loader = torch.utils.data.DataLoader(HANDataset(data_folder, 'train'), batch_size=batch_size, shuffle=True,                                         num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        train_acc, train_loss = train(train_loader=train_loader,
                                      model=model,
                                      criterion=criterion,
                                      optimizer=optimizer,
                                      epoch=epoch)

        # Decay learning rate every epoch
        adjust_learning_rate(optimizer, 0.5)

        val_acc, val_loss = evaluate(
            val_loader=val_loader, model=model, criterion=criterion)
        
        print(f'epoch {epoch} val acc: {val_acc}')

        if val_acc > max_acc:
            save_checkpoint(epoch, model, optimizer, word_map)
            max_acc = val_acc
        # Save pic
        writer['train_acc'].add_scalar(
            "train acc/epoch", train_acc, epoch+1)
        writer['train_loss'].add_scalar(
            "train loss/epoch", train_loss, epoch+1)
        writer['val_acc'].add_scalar(
            "val acc/epoch", val_acc, epoch+1)
        writer['val_loss'].add_scalar(
            "val loss/epoch", val_loss, epoch+1)
    test(test_loader=test_loader)


def train(train_loader, model, criterion, optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: cross entropy loss layer
    :param optimizer: optimizer
    :param epoch: epoch number
    """

    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time per batch
    data_time = AverageMeter()  # data loading time per batch
    losses = AverageMeter()  # cross entropy loss
    accs = AverageMeter()  # accuracies

    start = time.time()

    # Batches
    for i, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(train_loader):

        data_time.update(time.time() - start)

        # (batch_size, sentence_limit, word_limit)
        documents = documents.to(device)
        sentences_per_document = sentences_per_document.squeeze(
            1).to(device)  # (batch_size)
        words_per_sentence = words_per_sentence.to(
            device)  # (batch_size, sentence_limit)
        labels = labels.squeeze(1).to(device)  # (batch_size)

        # Forward prop.
        scores, word_alphas, sentence_alphas = model(documents, sentences_per_document,
                                                     words_per_sentence)  # (n_documents, n_classes), (n_documents, max_doc_len_in_batch, max_sent_len_in_batch), (n_documents, max_doc_len_in_batch)

        # Loss
        loss = criterion(scores, labels)  # scalar

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update
        optimizer.step()

        # Find accuracy
        _, predictions = scores.max(dim=1)  # (n_documents)
        correct_predictions = torch.eq(predictions, labels).sum().item()
        accuracy = correct_predictions / labels.size(0)

        # Keep track of metrics
        losses.update(loss.item(), labels.size(0))
        batch_time.update(time.time() - start)
        accs.update(accuracy, labels.size(0))

        start = time.time()

        # Print training status
        if i % print_freq == 0:
            writer['batch_loss'].add_scalar(
                "loss/10 batch", loss.item(), epoch*len(train_loader)+i)
            writer['batch_acc'].add_scalar(
                "acc/10 batch", accuracy, epoch*len(train_loader)+i)
            writer['batch_lr'].add_scalar(
                "lr/100 batch", optimizer.state_dict()['param_groups'][0]['lr'], epoch*len(train_loader)+i)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses,
                                                                  acc=accs))
    return accs.avg, losses.avg


def evaluate(val_loader, model, criterion):
    """
    Performs one epoch's valuating.

    :param val_loader: DataLoader for training data
    :param model: model
    :param criterion: cross entropy loss layer
    :param epoch: epoch number
    """
    model.eval()
    # Batches
    with torch.no_grad():

        batch_time = AverageMeter()  # forward prop. + back prop. time per batch
        data_time = AverageMeter()  # data loading time per batch
        losses = AverageMeter()  # cross entropy loss
        accs = AverageMeter()  # accuracies

        start = time.time()
        for i, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(tqdm(val_loader, desc='Evaluating')):

            data_time.update(time.time() - start)

            # (batch_size, sentence_limit, word_limit)
            documents = documents.to(device)
            sentences_per_document = sentences_per_document.squeeze(
                1).to(device)  # (batch_size)
            words_per_sentence = words_per_sentence.to(
                device)  # (batch_size, sentence_limit)
            labels = labels.squeeze(1).to(device)  # (batch_size)

            # Forward prop.
            scores, word_alphas, sentence_alphas = model(documents, sentences_per_document,
                                                         words_per_sentence)  # (n_documents, n_classes), (n_documents, max_doc_len_in_batch, max_sent_len_in_batch), (n_documents, max_doc_len_in_batch)
            # Loss
            loss = criterion(scores, labels)  # scalar

            # Find accuracy
            _, predictions = scores.max(dim=1)  # (n_documents)
            correct_predictions = torch.eq(predictions, labels).sum().item()
            accuracy = correct_predictions / labels.size(0)

            # Keep track of metrics
            losses.update(loss.item(), labels.size(0))
            batch_time.update(time.time() - start)
            accs.update(accuracy, labels.size(0))

            start = time.time()

    return accs.avg, losses.avg


def test(test_loader):
    checkpoint = torch.load('/data2/lwc/PythonProjects/exp3/checkpoint_han.pth.tar')
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    # Track metrics
    accs = AverageMeter()  # accuracies
    
    with torch.no_grad():
        # Evaluate in batches
        for i, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(
                tqdm(test_loader, desc='Testing')):
            # (batch_size, sentence_limit, word_limit)
            documents = documents.to(device)
            sentences_per_document = sentences_per_document.squeeze(
                1).to(device)  # (batch_size)
            words_per_sentence = words_per_sentence.to(
                device)  # (batch_size, sentence_limit)
            labels = labels.squeeze(1).to(device)  # (batch_size)

            scores, word_alphas, sentence_alphas = model(documents, sentences_per_document,
                                                        words_per_sentence)  # (n_documents, n_classes), (n_documents, max_doc_len_in_batch, max_sent_len_in_batch), (n_documents, max_doc_len_in_batch)

            # Find accuracy
            _, predictions = scores.max(dim=1)  # (n_documents)
            correct_predictions = torch.eq(predictions, labels).sum().item()
            accuracy = correct_predictions / labels.size(0)

            # Keep track of metrics
            accs.update(accuracy, labels.size(0))

            start = time.time()

        # Print final result
        print('\n * TEST ACCURACY - %.1f per cent\n' % (accs.avg * 100))


if __name__ == '__main__':
    main()
