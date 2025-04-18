import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from app.model import SentimentAnalysis
from utils import YelpReviewPolarityDatasetLoader

device = torch.device("cpu")
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")

print("Using device: {}".format(device))

def train(dataloader):
    model.train()

    # Train the model
    train_loss = 0
    train_acc = 0
    for text, offsets, label in dataloader:
        # START TODO:
        # TODO complete the training code. The inputs of the model are text and offsets
        text, offsets, label = text.to(device), offsets.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(text, offsets)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(output)
        train_acc += (output.argmax(1) == label).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(dataloader.dataset), train_acc / len(dataloader.dataset)

@torch.no_grad() # decorator: avoid computing gradients
def test(dataloader: DataLoader):
    model.eval()

    loss = 0
    acc = 0
    # TODO complete the evaluation code. The inputs of the model are text and offsets
    for text, offsets, label in dataloader:
        text, offsets, label = text.to(device), offsets.to(device), label.to(device)
        output = model(text, offsets)
        loss_val = criterion(output, label)

        loss += loss_val.item() * len(output)
        acc += (output.argmax(1) == label).sum().item()

    return loss / len(dataloader.dataset), acc / len(dataloader.dataset)


if __name__ == "__main__":

    # Hyperparameters
    NGRAMS = 1  # 2 or 3 will be better but slower.
    BATCH_SIZE = 16
    EMBED_DIM = 32
    N_EPOCHS = 2  # 5 would be ideal, but slower.

    # Load the dataset
    yelp_loader = YelpReviewPolarityDatasetLoader(NGRAMS, BATCH_SIZE, device=device)

    # Retrieve train, validation and test datasets
    train_val_dataset = yelp_loader.get_train_val_dataset()
    test_dataset = yelp_loader.get_test_dataset()

    # Retrieve vocabulary size and number of classes
    VOCAB_SIZE = yelp_loader.get_vocab_size()
    NUM_CLASS = yelp_loader.get_num_classes()

    # Load the model
    # START TODO / END TODO: load the model
    model = SentimentAnalysis(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)

    # We will use CrossEntropyLoss even though we are doing binary classification
    # because the code is ready to also work for many classes
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Setup optimizer and LR scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    # Split train and val datasets
    # TODO split `train_val_dataset` in `train_dataset` and `valid_dataset`. The size of train dataset should be 95%

    train_dataset, valid_dataset = torch.utils.data.random_split(train_val_dataset, [0.95, 0.05])

    # DataLoader needs an special function to generate the batches.
    # Since we will have inputs of varying size, we will concatenate
    # all the inputs in a single vector and create a vector with the "offsets" between inputs.
    # You can check the `generate_batch` function for more info.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=yelp_loader.generate_batch)
    val_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=yelp_loader.generate_batch)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=yelp_loader.generate_batch)

    print("Data loaded")

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_loss, train_acc = train(train_loader)
        valid_loss, valid_acc = test(val_loader)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print(f"Epoch: {epoch + 1},  | time in {mins} minutes, {secs} seconds")
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

    print("Training finished")

    test_loss, test_acc = test(test_loader)
    print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')

    # Now save the artifacts of the training
    savedir = "app/state_dict.pt"
    print(f"Saving checkpoint to {savedir}...")
    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab_word2idx": yelp_loader.vocab.word2idx,
        "vocab_idx2word": yelp_loader.vocab.idx2word,
        "ngrams": NGRAMS,
        "embed_dim": EMBED_DIM,
        "num_class": NUM_CLASS,
    }
    torch.save(checkpoint, savedir)