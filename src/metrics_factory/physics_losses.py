import torch






def lwr(model, x_unlabel, alpha=1):
    #get gradient
    x = torch.tensor(x_unlabel[:,0:1], requires_grad=True).float().to(model.device)
    t = torch.tensor(x_unlabel[:,1:2], requires_grad=True).float().to(model.device)
    rho, u = model.test(torch.cat((x, t), 1))
    q = rho * u

    # get the derivative
    drho_dt = torch.autograd.grad(rho, t, torch.ones([t.shape[0], 1]).to(model.device),
                                  retain_graph=True, create_graph=True)[0]
    dq_dx = torch.autograd.grad(q, x, torch.ones([x.shape[0], 1]).to(model.device),
                                  retain_graph=True, create_graph=True)[0]

    drho_dt = model





def raissi(loss_labeled, model, x_batch, alpha=1):
    '''
    :param loss_labeled: loss function for the data discrepancy
    :param model: flow model
    :param x_batch: x feature
    :param alpha: weights for the data discrepancy
    :return: alpha*loss_labeled + (1-alpha)*loss_labeled
    '''

