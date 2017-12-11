"""
This is a separate utility class for attacking YOLO detector specifically
"""

from parameters_yolo import *
from attack_detector.yad2k.models.keras_yolo import yolo_head


def gradient_yolo(yolo_model, anchors, op=0):

    _, _, output_conf, output_class = yolo_head(yolo_model.output, anchors, 80)

    target_conf = K.placeholder(shape=(1, 19, 19, 5, 1))
    target_class = K.placeholder(shape=(1, 19, 19, 5, 80))

    # We use MSE here but it can be changed to other appropriate loss
    loss_conf = keras.losses.mean_squared_error(target_conf, output_conf)
    loss_class = keras.losses.mean_squared_error(target_class, output_class)

    if op == 0:
        loss = loss_conf + loss_class
    elif op == 1:
        loss = loss_conf
    elif op == 2:
        loss = loss_class
    else:
        raise ValueError("Invalid argument: op")

    grad = K.gradients(loss, yolo_model.input)

    if op == 0:
        return K.function([yolo_model.input, target_conf, target_class,
                           K.learning_phase()], grad)
    elif op == 1:
        return K.function([yolo_model.input, target_conf, K.learning_phase()],
                          grad)
    elif op == 2:
        return K.function([yolo_model.input, target_class, K.learning_phase()],
                          grad)


def gradient_input(grad_fn, x, target_conf=None, target_class=None):
    """Wrapper function to use gradient function more cleanly"""

    if target_conf is not None and target_class is not None:
        return grad_fn([x.reshape(INPUT_SHAPE), target_conf,
                        target_class, 0])[0][0]
    elif target_conf is not None:
        return grad_fn([x.reshape(INPUT_SHAPE), target_conf, 0])[0][0]
    elif target_class is not None:
        return grad_fn([x.reshape(INPUT_SHAPE), target_class, 0])[0][0]
    else:
        return None


def fg_yolo(model, x, y, mag_list, anchors, target=True, mask=None):
    """
    Fast Gradient attack. Similar to iterative attack but only takes one step
    and then clip result afterward.

    Parameters
    ----------
    model    : Keras Model
               Model to attack
    x        : np.array, shape=(n_sample, height, width, n_channel)
               Benign samples to attack
    y        : np.array, shape=(n_sample, NUM_LABELS)
               Target label for each of the sample in x if target is True.
               Otherwise, corresponding labels of x. Must be one-hot encoded.
    mag_list : list of float
               List of perturbation magnitude to use in the attack
    target   : (optional) bool
               True, if targeted attack. False, otherwise.
    mask     : (optional) np.array of 0 or 1, shape=(n_sample, height, width)
               Mask to restrict gradient update on valid pixels

    Return
    ------
    x_adv    : np.array, shape=(n_mag, n_sample, height, width, n_channel)
               Adversarial examples
    """

    x_adv = np.zeros((len(mag_list), ) + x.shape, dtype=np.float32)
    grad_fn = gradient_yolo(model, anchors, op=1)
    start_time = time.time()

    for i, x_in in enumerate(x):

        # Retrieve gradient
        if target is not None:
            grad = -1 * gradient_input(grad_fn, x_in, target_conf=y[i])
        else:
            grad = gradient_input(grad_fn, x_in, target_conf=y[i])

        # Apply mask
        if mask is not None:
            mask_rep = np.repeat(mask[i, :, :, np.newaxis], N_CHANNEL, axis=2)
            grad *= mask_rep

        # Normalize gradient
        try:
            grad /= np.linalg.norm(grad)
        except ZeroDivisionError:
            raise

        for j, mag in enumerate(mag_list):
            x_adv[j, i] = x_in + grad * mag

        # Progress printing
        if (i % 1000 == 0) and (i > 0):
            elasped_time = time.time() - start_time
            print("Finished {} samples in {:.2f}s.".format(i, elasped_time))
            start_time = time.time()

    # Clip adversarial examples to stay in range [0, 1]
    x_adv = np.clip(x_adv, 0, 1)

    return x_adv


def iterative_yolo(model, x, y, anchors, n_step=20, step_size=0.05, target=True,
                   mask=None):
    """
    Iterative attack. Move a benign sample in the gradient direction one small
    step at a time for <n_step> times. Clip values after each step.

    Parameters
    ----------
    model     : Keras Model
                Model to attack
    x         : np.array, shape=(n_sample, height, width, n_channel)
                Benign samples to attack
    y         : np.array, shape=(n_sample, NUM_LABELS)
                Target label for each of the sample in x if target is True.
                Otherwise, corresponding labels of x. Must be one-hot encoded.
    n_step    : (optional) int
                Number of iteration to take
    step_size : (optional) float
                Magnitude of perturbation in each iteration
    target    : (optional) bool
                True, if targeted attack. False, otherwise.
    mask      : (optional) np.array of 0 or 1, shape=(n_sample, height, width)
                Mask to restrict gradient update on valid pixels

    Return
    ------
    x_adv    : np.array, shape=(n_mag, n_sample, height, width, n_channel)
               Adversarial examples
    """

    x_adv = np.zeros(x.shape, dtype=np.float32)
    grad_fn = gradient_yolo(model, anchors, op=1)
    start_time = time.time()

    for i, x_in in enumerate(x):

        x_cur = np.copy(x_in)
        # Get mask with the same shape as gradient
        if mask is not None:
            mask_rep = np.repeat(mask[i, :, :, np.newaxis], N_CHANNEL, axis=2)
        # Start update in steps
        for _ in range(n_step):
            if target is not None:
                grad = -1 * gradient_input(grad_fn, x_cur, target_conf=y[i])
            else:
                grad = gradient_input(grad_fn, x_cur, target_conf=y[i])

            # Apply mask
            if mask is not None:
                grad *= mask_rep

            # Normalize gradient
            try:
                grad /= np.linalg.norm(grad)
            except ZeroDivisionError:
                raise

            x_cur += grad * step_size
            # Clip to stay in range [0, 1]
            x_cur = np.clip(x_cur, 0, 1)

        x_adv[i] = np.copy(x_cur)

        # Progress printing
        if (i % 200 == 0) and (i > 0):
            elasped_time = time.time() - start_time
            print("Finished {} samples in {:.2f}s.".format(i, elasped_time))
            start_time = time.time()

    return x_adv
