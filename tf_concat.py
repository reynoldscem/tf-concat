import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def rand_tensor_indices(low=1, high=5):
    from numpy.random import randint
    n_dims = randint(low=low, high=high)
    return tuple(randint(low=low, high=high, size=n_dims))


def make_test_list(k=5):
    entries = (
        make_test_list,
        lambda: tf.random_uniform(rand_tensor_indices()),
        lambda: None
    )
    weights = (0.2, 0.4, 0.4)
    from numpy.random import choice
    return [entry() for entry in choice(entries, p=weights, size=k)]


def combine_weights(in_list):
    """
    Returns a 1D tensor of the input list of (nested lists of) tensors, useful
    for doing things like comparing current weights with old weights for EWC.

    1.) For all elements in input list, (ln 3)
          if a list combine it recursively
          else leave it alone
    2.) From resulting list, get all non-none elements and flatten them (ln 2)
    3.) If resulting list is empty return None (ln 1)
          else return concatenation of list
    ( All on one line :) )
    """

    return (lambda x: None if not x else tf.concat(x, axis=0))([
        tf.reshape(x, [-1])
        for x in [
            combine_weights(x) if isinstance(x, list) else x for x in in_list
        ] if x is not None
    ])


def combine_weights2a(in_list):
    accumulator = []
    stack = [item for item in in_list]

    while len(stack) > 0:
        entry = stack.pop()
        if isinstance(entry, tf.Tensor):
            accumulator.append(tf.reshape(entry, (-1,)))
        elif entry is None:
            pass
        elif isinstance(entry, list):
            for subentry in entry:
                stack.append(subentry)
        else:
            raise Exception

    return tf.concat(accumulator, axis=0)


def combine_weights2b(in_list):
    stack = [x for x in in_list]  # Copy to avoid messing with original list

    # One line :)
    return tf.concat(
        [
            tf.reshape(x, (-1,))
            for x in stack if x is not None and
            not (
                isinstance(x, list)  # We abuse lazy evaluation
                and [stack.append(subentry) for subentry in x]
            )
        ], axis=0)


def main():
    test_list = make_test_list()
    print(test_list)

    with tf.Session().as_default():
        print(combine_weights(test_list).get_shape().as_list())
        print(combine_weights2b(test_list).get_shape().as_list())


if __name__ == '__main__':
    main()
