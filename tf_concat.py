import tensorflow as tf


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

    from random import choices
    return [entry() for entry in choices(entries, weights, k=k)]


def main():
    test_list = make_test_list()
    print(test_list)

    accumulator = []
    stack = []

    for index in range(len(test_list)):
        stack.append((index,))

    while len(stack) > 0:
        visit_index = stack.pop()
        entry = test_list
        for sub_index in visit_index:
            entry = entry[sub_index]
        if isinstance(entry, tf.Tensor):
            accumulator.append(tf.reshape(entry, (-1,)))
        elif entry is None:
            pass
        elif isinstance(entry, list):
            for index in range(len(entry)):
                stack.append(visit_index + (index, ))
        else:
            raise Exception

    with tf.Session().as_default():
        print(tf.concat(accumulator, axis=0).eval())


if __name__ == '__main__':
    main()
