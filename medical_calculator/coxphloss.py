import torch
from torch import Tensor

def cox_ph_loss_sorted(log_h: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    """Requires the input to be sorted by descending duration time.
    See DatasetDurationSorted.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) (p.s. h_0exp(log_h) where log_h equivalent to beta*x) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma) # gamma and eps are just for numerical stability

    # print(events) # [0, 0, 0, 0, 0]
    # print(log_h) # [3.1497, -3.5484, -1.7658, -7.4636, -2.7555]
    # print(gamma)  # 3.1497
    # print(log_cumsum_h)  # [3.1497, 3.1509, 3.1582, 3.1583, 3.1610]
    # print(- log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum()+0.001))
    

    return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum()+0.001)  
    # changes were made here, otherwise with small batch size and if no positive events included, div(0) will give invalid result

    # note:  
    # log_h.sub(log_cumsum_h).mul(events).sum() is equivalent to log-partial likelihood (- is just a minus sign in front). 
    # ref: https://towardsdatascience.com/survival-analysis-optimize-the-partial-likelihood-of-the-cox-model-b56b8f112401
    # div is just used to normalise the loss by the number of events
    # this is correct if there is no tied event. If there are tied events, this assume a random order between them and only appear 
    # as a good approximation.

def cox_ph_loss(log_h: Tensor, durations: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    idx = durations.sort(descending=True)[1] # from long to short
    events = events[idx]
    log_h = log_h[idx]
    return cox_ph_loss_sorted(log_h, events, eps)

class CoxPHLoss(torch.nn.Module):
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    def forward(self, log_h: Tensor, durations: Tensor, events: Tensor) -> Tensor:
        return cox_ph_loss(log_h, durations, events)