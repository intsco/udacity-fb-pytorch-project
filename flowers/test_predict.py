from pathlib import Path

from flowers.model import *


def ensemble_predict_test(arch, device, test_dl, model_paths):
    cv_outputs = []
    for model_path in model_paths:
        model = load_model(arch, device, model_path)
        model.to(device)
        model.eval()
        with torch.set_grad_enabled(False):
            model_outputs = np.concatenate([tta(model, inputs.to(device)).detach().cpu().numpy()
                                            for inputs, _ in test_dl])
            cv_outputs.append(model_outputs)
    outputs_avg = np.stack(cv_outputs, axis=1).mean(1)
    return outputs_avg.argmax(axis=1)


def predict_test(test_dl, model, device):
    model.to(device)
    model.eval()
    test_preds = []
    for inputs, _ in test_dl:
        print('.', end='')
        inputs = inputs.to(device)

        with torch.set_grad_enabled(False):
            if len(inputs.size()) > 4:
                ns, ncrops, c, h, w = inputs.size()
                outputs = model(inputs.view(-1, c, h, w))
                outputs = outputs.view(ns, ncrops, -1).mean(1)
            else:
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_preds.append(preds.cpu().numpy())

    return np.concatenate(test_preds)


def create_submission(test_preds, test_dl, idx_to_class, fn='submission.csv'):
    preds = [idx_to_class[ci] for ci in test_preds]

    with open(fn, 'w') as f:
        f.write('file_name,id\n')
        for (img_path, _), pred_class in zip(test_dl.dataset.imgs, preds):
            img_name = Path(img_path).name
            f.write(f'{img_name},{pred_class}\n')
    print(f'Saved to file: {fn}')
