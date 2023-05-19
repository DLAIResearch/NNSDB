import torch
from torch.optim import LBFGS
def create_FNNS_example(inputs_bd,identity_msg,opt,decoder):
    steps = 20000
    max_iter = 10
    alpha = 0.1
    eps = 0.3
    image = inputs_bd.unsqueeze(0)                 
    target = identity_msg
    model = decoder
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum').to(opt.device)
    target = target.to(opt.device)
    adv_image = image.clone().detach()
    for i in range(steps // max_iter):
        adv_image.requires_grad = True
        optimizer = LBFGS([adv_image], lr=alpha, max_iter=max_iter)

        def closure():
            adv_image_rgb = adv_image
            outputs = model(adv_image_rgb)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            return loss

        optimizer.step(closure)
        delta = torch.clamp(adv_image - image, min=-eps, max=eps)
        adv_image = torch.clamp(image + delta, min=0, max=1).detach()
        adv_image_rgb = adv_image
        err = len(torch.nonzero((model(adv_image_rgb)>0).float().view(-1) != target.view(-1))) / target.numel()
        if err< 0.00001: break
    return adv_image.squeeze()


