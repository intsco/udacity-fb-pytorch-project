from functools import partial

from flowers.fastai_tricks import *


def tta(model, inputs):
    ns, ncrops, c, h, w = inputs.size()
    outputs = model(inputs.view(-1, c, h, w))
    return outputs.view(ns, ncrops, -1).mean(1)


def create_model(arch, device, n_classes=102, class_to_idx=None):
    body = create_body(arch=arch, pretrained=True, cut=-2)
    body_out_ch = body(torch.zeros(1, 3, 224, 224)).shape[1]
    head = create_head(body_out_ch *2, n_classes, lin_ftrs=[512], dropout_ps=0.5, bn_final=False)
    model = nn.Sequential(body, head)
    init_module_rec(model[1], func=partial(init_weight_bias, func=nn.init.kaiming_normal_))
    model.to(device)

    if class_to_idx is not None:
        model.class_to_idx = class_to_idx
        model.idx_to_class =  {v: k for k ,v in model.class_to_idx.items()}
    return model


def load_model(arch, device, path):
    state = torch.load(open(path, 'rb'))
    model = create_model(arch, device, n_classes=state['n_classes'], class_to_idx=state.get('class_to_idx', None))
    model.load_state_dict(state['state_dict'])
    return model


def save_model(model, path, class_to_idx=None):
    state = {
        'state_dict': model.state_dict(),
        'n_classes': model[-1][-1].out_features, 'class_to_idx': class_to_idx
    }
    torch.save(state, open(path, 'wb'))
