import torch
from torch import optim, nn
from torch.nn import functional as F

from src.architecture.feature_extractors.image_feature_extractor import Head
from src.architecture.ptr_network import PointerNetwork

EPSILON = 1e-7  # for numerical stability of log in case of probabilities reaching 0


def cross_entropy(overall_probabilities: torch.Tensor,
                  target: torch.Tensor) -> torch.Tensor:
    flat_target = target.flatten()
    flat_probabilities = overall_probabilities.reshape(-1, overall_probabilities.shape[-1])
    return F.nll_loss(torch.log(flat_probabilities + EPSILON), flat_target, reduction='mean')


def sample(min_length=5, max_length=12):
    """
    Generates a single example for a pointer network. The example consist in a tuple of two
    elements. First element is an unsorted array and the second element
    is the result of applying argsort on the first element
    """
    array_len = torch.randint(low=min_length,
                              high=max_length + 1,
                              size=(1,))
    x = torch.randint(high=array_len.item(), size=(array_len,))
    return x, x.argsort()


def batch(batch_size=32, min_len=5, max_len=12):
    array_len = torch.randint(low=min_len,
                              high=max_len + 1,
                              size=(1,))

    x = torch.randint(high=10, size=(batch_size, array_len))
    return x, x.argsort(dim=1)


def train(model, optimizer, epoch, clip=1.):
    """Train single epoch"""
    print('Epoch [{}] -- Train'.format(epoch))
    for step in range(500):
        optimizer.zero_grad()

        # Forward
        x, y = batch()
        x = x.unsqueeze(-1).to(torch.float32)
        probabilities, peak_indices = model(x)
        loss = cross_entropy(overall_probabilities=probabilities, target=y)

        # Backward
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if (step + 1) % 100 == 0:
            print('Epoch [{}] loss: {}'.format(epoch, loss.item()))


@torch.no_grad()
def evaluate(model, epoch):
    """Evaluate after a train epoch"""
    print('Epoch [{}] -- Evaluate'.format(epoch))

    x_val, y_val = batch(4)
    x_val = x_val.unsqueeze(-1).to(torch.float32)

    probabilities, peak_indices = model(x_val)
    loss = cross_entropy(overall_probabilities=probabilities, target=y_val)

    x_val = x_val.squeeze(-1)

    print('Validation loss: {}'.format(loss.item()))
    for i in range(peak_indices.size(0)):
        print('{} --> {} --> {}'.format(
            x_val[i],
            x_val[i].gather(0, peak_indices[i]),
            x_val[i].gather(0, y_val[i])
        ))


mlp_feature_extractor = Head(input_size=1,
                             hidden_size=20,
                             hidden_layers=2,
                             output_size=64)

ptr_network = PointerNetwork(feature_extractor=mlp_feature_extractor,
                             embedding_dim=64,
                             hidden_size=50,
                             max_seq_len=15,
                             only_uniques=True)

optimizer = optim.Adam(ptr_network.parameters())

for epoch in range(10):
    train(ptr_network, optimizer, epoch + 1)
    evaluate(ptr_network, epoch + 1)
