from lib.utils import *


def fg(model, x, y, mag_list, target=True, mask=None):
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
    grad_fn = gradient_fn(model)
    start_time = time.time()

    for i, x_in in enumerate(x):

        # Retrieve gradient
        if target:
            grad = -1 * gradient_input(grad_fn, x_in, y[i])
        else:
            grad = gradient_input(grad_fn, x_in, y[i])
        print(np.linalg.norm(grad))

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


def iterative(model, x, y, norm="2", n_step=20, step_size=0.05, target=True,
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
    norm      : (optional) string
                "2" = L-2 norm (default)
                "inf" = L-infinity norm
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
    grad_fn = gradient_fn(model)
    start_time = time.time()

    for i, x_in in enumerate(x):

        x_cur = np.copy(x_in)
        # Get mask with the same shape as gradient
        if mask is not None:
            mask_rep = np.repeat(mask[i, :, :, np.newaxis], N_CHANNEL, axis=2)
        # Start update in steps
        for _ in range(n_step):
            if target is not None:
                grad = -1 * gradient_input(grad_fn, x_cur, y[i])
            else:
                grad = gradient_input(grad_fn, x_cur, y[i])

            if norm == "2":
                try:
                    grad /= np.linalg.norm(grad)
                except ZeroDivisionError:
                    raise
            elif norm == "inf":
                grad = np.sign(grad)
            else:
                raise ValueError("Invalid norm!")

            # Apply mask
            if mask is not None:
                grad *= mask_rep

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


def fg_transform(model, x, y, mag_list, target=True, mask=None,
                 batch_size=BATCH_SIZE):
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

    P_TRN = 1.0  # Probability of applying transformation
    P_ENH = 1.0  # Probability of applying enhancement
    INT_TRN = 0.3  # Intensity of randomness (for transform)
    INT_ENH = 0.4  # Intensity of randomness (for enhance)

    # Initialize random transformer
    seed = np.random.randint(1234)
    rnd_transform = RandomTransform(seed=seed, p=P_TRN, intensity=INT_TRN)
    rnd_enhance = RandomEnhance(seed=seed, p=P_ENH, intensity=INT_ENH)

    x_adv = np.zeros((len(mag_list),) + x.shape, dtype=np.float32)
    grad_fn = gradient_fn(model)
    start_time = time.time()

    for i, x_in in enumerate(x):

        # Retrieve gradient
        if target:
            grad = -1 * gradient_input(grad_fn, x_in, y[i])
            for j in range(batch_size - 1):
                x_trn = rnd_transform.transform(x_in)
                x_trn = rnd_enhance.enhance(x_trn)
                grad += -1 * gradient_input(grad_fn, x_trn, y[i])
            grad /= batch_size
        else:
            grad = gradient_input(grad_fn, x_in, y[i])
            for j in range(batch_size - 1):
                x_trn = rnd_transform.transform(x_in)
                x_trn = rnd_enhance.enhance(x_trn)
                grad += gradient_input(grad_fn, x_trn, y[i])
            grad /= batch_size

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


def iter_transform(model, x, y, norm="2", n_step=20, step_size=0.05,
                   target=True, mask=None, batch_size=BATCH_SIZE):

    P_TRN = 1.0  # Probability of applying transformation
    P_ENH = 1.0  # Probability of applying enhancement
    INT_TRN = 0.1  # Intensity of randomness (for transform)
    INT_ENH = 0.2  # Intensity of randomness (for enhance)

    # Initialize random transformer
    seed = np.random.randint(1234)
    rnd_transform = RandomTransform(seed=seed, p=P_TRN, intensity=INT_TRN)
    rnd_enhance = RandomEnhance(seed=seed, p=P_ENH, intensity=INT_ENH)

    grad_fn = gradient_fn(model)
    start_time = time.time()
    x_adv = np.copy(x)
    losses = []
    trans = []
    factors = []

    # Get a list of transformations
    for _ in range(batch_size - 1):
        _ = rnd_transform.transform(x_adv)
        _ = rnd_enhance.enhance(x_adv)
        trans.append(rnd_transform.get_last_transform())
        factors.append(rnd_enhance.get_last_factors())

    # Get mask with the same shape as gradient
    if mask is not None:
        mask_rep = np.repeat(mask[:, :, np.newaxis], N_CHANNEL, axis=2)
    # Start update in steps
    for _ in range(n_step):
        if target is not None:
            # Sum gradient over the entire batch of transformed images
            grad = -1 * gradient_input(grad_fn, x_adv, y)
            for i in range(batch_size - 1):
                x_trn = rnd_transform.apply_transform(x_adv, trans[i])
                x_trn = rnd_enhance.enhance_factors(x_trn, factors[i])
                grad += -1 * gradient_input(grad_fn, x_trn, y)
            # Average by batch size
            grad /= batch_size
        else:
            grad = gradient_input(grad_fn, x_adv, y)
            for i in range(batch_size - 1):
                x_trn = rnd_transform.apply_transform(x_adv, trans[i])
                x_trn = rnd_enhance.enhance_factors(x_trn, factors[i])
                grad += gradient_input(grad_fn, x_trn, y)
            grad /= batch_size

        # Normalize gradient by specified norm
        if norm == "2":
            try:
                grad /= np.linalg.norm(grad)
            except ZeroDivisionError:
                raise
        elif norm == "inf":
            grad = np.sign(grad)
        else:
            raise ValueError("Invalid norm!")

        # Apply mask
        if mask is not None:
            grad *= mask_rep

        x_adv += grad * step_size
        # Clip to stay in range [0, 1]
        x_adv = np.clip(x_adv, 0, 1)

        # Also save progress of loss
        loss = model.evaluate(x_adv.reshape(INPUT_SHAPE),
                              y.reshape(1, OUTPUT_DIM), verbose=0)[0]
        losses.append(loss)

    return x_adv, np.array(losses)


def rnd_pgd(model, x, y, norm="2", n_step=40, step_size=0.01, target=True,
            mask=None, init_rnd=0.1):
    """
    Projected gradient descent started with a random point in a ball centered
    around real data point.
    """

    x_rnd = np.zeros_like(x)

    for i, x_cur in enumerate(x):

        # Find a random point in a ball centered at given data point
        h, w, c = x_cur.shape
        epsilon = np.random.rand(h, w, c) - 0.5
        if norm == "2":
            try:
                epsilon /= np.linalg.norm(epsilon)
            except ZeroDivisionError:
                raise
        elif norm == "inf":
            epsilon = np.sign(epsilon)
        else:
            raise ValueError("Invalid norm!")

        x_rnd[i] = np.clip(x_cur + init_rnd * epsilon, 0, 1)

    return iterative(model, x_rnd, y, norm, n_step, step_size, target, mask)


def s_pgd(model, x, y, norm="2", n_step=40, step_size=0.01, target=True,
          mask=None, beta=0.1, early_stop=True, grad_fn=None):
    """
    Projected gradient descent with added randomness during each step
    """

    x_adv = np.zeros_like(x)
    if grad_fn is None:
        grad_fn = gradient_fn(model)
    start_time = time.time()

    for i, x_in in enumerate(x):

        x_cur = np.copy(x_in)
        # Get mask with the same shape as gradient
        if mask is not None:
            mask_rep = np.repeat(mask[i, :, :, np.newaxis], N_CHANNEL, axis=2)

        y_cls = np.argmax(y[i])
        # Start update in steps
        for _ in range(n_step):

            # Get gradient
            if target is not None:
                grad = -1 * gradient_input(grad_fn, x_cur, y[i])
            else:
                grad = gradient_input(grad_fn, x_cur, y[i])

            # Get uniformly random direction
            h, w, c = x_cur.shape
            epsilon = np.random.rand(h, w, c) - 0.5

            if norm == "2":
                try:
                    grad /= np.linalg.norm(grad)
                    epsilon /= np.linalg.norm(epsilon)
                except ZeroDivisionError:
                    raise
            elif norm == "inf":
                grad = np.sign(grad)
                epsilon = np.sign(epsilon)
            else:
                raise ValueError("Invalid norm!")

            # Apply mask
            if mask is not None:
                grad *= mask_rep
                epsilon += mask_rep

            x_cur += (grad * step_size + beta * epsilon * step_size)
            # Clip to stay in range [0, 1]
            x_cur = np.clip(x_cur, 0, 1)

            if early_stop:
                # Stop when sample becomes adversarial
                if target is not None:
                    if predict(model, x_cur) == y_cls:
                        break
                else:
                    if predict(model, x_cur) != y_cls:
                        break

        x_adv[i] = x_cur

        # Progress printing
        if (i % 200 == 0) and (i > 0):
            elasped_time = time.time() - start_time
            print("Finished {} samples in {:.2f}s.".format(i, elasped_time))
            start_time = time.time()

    return x_adv


def symbolic_fgs(x, grad, eps=0.3, clipping=True):
    """
    FGSM attack.
    """

    # signed gradient
    normed_grad = K.sign(grad)

    # Multiply by constant epsilon
    scaled_grad = eps * normed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = K.stop_gradient(x + scaled_grad)

    if clipping:
        adv_x = K.clip(adv_x, 0, 1)

    return adv_x


def symbolic_fg(x, grad, eps=0.3, clipping=True):
    """
    FG attack
    """

    # Unit vector in direction of gradient
    reduc_ind = list(range(1, len(x.get_shape())))
    normed_grad = grad / tf.sqrt(tf.reduce_sum(tf.square(grad),
                                               reduction_indices=reduc_ind,
                                               keep_dims=True))
    # Multiply by constant epsilon
    scaled_grad = eps * normed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = K.stop_gradient(x + scaled_grad)

    if clipping:
        adv_x = K.clip(adv_x, 0, 1)

    return adv_x


def symb_iter_fgs(model, x, y, steps, alpha, eps, clipping=True):
    """
    I-FGSM attack.
    """

    adv_x = x
    # iteratively apply the FGSM with small step size
    for i in range(steps):
        logits = model(adv_x)
        grad = gen_grad(adv_x, logits, y)

        adv_x = symbolic_fgs(adv_x, grad, alpha, True)
        r = adv_x - x
        r = K.clip(r, -eps, eps)
        adv_x = x + r

    if clipping:
        adv_x = K.clip(adv_x, 0, 1)

    return adv_x
