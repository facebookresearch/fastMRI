
import torch
import torch.optim

def load(p, args):
    if args.method == "adam":
        # Default is 0.9, 0.999, so we kind of immitate that
        if args.beta2 is None:
            beta2 = 1-((1-args.momentum)/100)
        else:
            beta2 = args.beta2
        optimizer = torch.optim.Adam(p, lr=args.lr, betas=(args.momentum, beta2),
            weight_decay=args.decay, eps=args.adam_eps)
    elif args.method == "sgd":
        optimizer = torch.optim.SGD(p, lr=args.lr, momentum=args.momentum,
            weight_decay=args.decay)
    elif args.method == "rmsprop":
        optimizer = torch.optim.RMSprop(p, lr=args.lr, momentum=args.momentum,
            weight_decay=args.decay)
    elif args.method == "lbfgs":
        optimizer = torch.optim.LBFGS(p, lr=args.lr)
    else:
        raise Exception(f"Unrecognised optimizer method name: {args.method}")

    return optimizer
