from collections import Counter
import numpy as np
from matplotlib import pyplot as plt


def imshow(ax, inp, imagenet_stats, title=None):
    """ Imshow for Tensor """
    inp = inp.transpose((1, 2, 0))
    inp = imagenet_stats['std'] * inp + imagenet_stats['mean']
    inp = np.clip(inp, 0, 1)
    ax.imshow(inp)
    if title is not None:
        plt.title(title)


def plot_inputs(rows, cols, input_iter):
    inputs, classes = [t.numpy() for t in next(input_iter)]
    most_common_classes = [ci for ci, _ in Counter(classes).most_common(3)]

    for c in most_common_classes:
        f, axes = plt.subplots(nrows=rows, ncols=cols, figsize=( 3 *cols, 3* rows))
        inputs_ = inputs[classes == c]

        if len(inputs_.shape) > 4:
            bs, ncrops, c, h, w = inputs_.shape
            inputs_ = inputs_.reshape(-1, c, h, w)

        for i in range(min(len(inputs_), rows * cols)):
            imshow(axes[i % rows, i % cols], inputs_[i])
        f.suptitle(f'class {c}', y=0.91);
        plt.show()


