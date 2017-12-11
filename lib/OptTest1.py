import random

from lib.keras_utils import *
from lib.utils import *
from parameters import *
from skimage.transform import ProjectiveTransform

EPS = 1e-10   # Epsilon
MIN_CP = -2.  # Minimum power index of c
MAX_CP = 2.   # Maximum power index of c
SCORE_THRES = 0.99  # Softmax score threshold to consider success of attacks
PROG_PRINT_STEPS = 200  # Print progress every certain steps
EARLYSTOP_STEPS = 1000  # Early stopping if no improvement for certain steps
INT_TRN = 0.07  # Intensity of randomness (for transform)

DELTA_BRI = 0.15
THRES = 0.05


class OptTest1:
    """
    This class implements a generator for adversarial examples that are robust
    to certain transformations or variations. It is a modification from
    Carlini et al. (https://arxiv.org/abs/1608.04644) and Athalye et al.
    (https://arxiv.org/abs/1707.07397)
    """

    def _setup_opt(self):
        """Used to setup optimization when c is updated"""

        # obj_func = c * loss + l2-norm(d)
        self.f = self.c * self.loss + self.c_smooth * self.smooth + self.norm
        # Setup optimizer
        if self.use_bound:
            # Use Scipy optimizer with upper and lower bound [0, 1]
            self.optimizer = ScipyOptimizerInterface(
                self.f, var_list=self.var_list, var_to_bounds={
                    self.x_in: (0, 1)},
                method="L-BFGS-B")
        else:
            # Use learning rate decay
            global_step = tf.Variable(0, trainable=False)

            if self.decay:
                # lr = tf.train.exponential_decay(
                #     self.lr, global_step, 100, 0.95, staircase=False)
                lr = tf.train.inverse_time_decay(
                    self.lr, global_step, 50, 0.01, staircase=True)
            else:
                lr = self.lr
            # Use Adam optimizer
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
            # self.optimizer = tf.train.GradientDescentOptimizer(
            #     learning_rate=lr)
            self.opt = self.optimizer.minimize(
                self.f, var_list=self.var_list, global_step=global_step)

    def __init__(self, model, target=True, c=1, lr=0.01, init_scl=0.1,
                 use_bound=False, loss_op=0, k=5, var_change=True, p_norm="1",
                 l=0, use_mask=True, decay=True, batch_size=BATCH_SIZE,
                 sp_size=None, rnd_tran=INT_TRN, rnd_bri=DELTA_BRI, c_smooth=10):
        """
        Initialize the optimizer. Default values of the parameters are
        recommended and decided specifically for attacking traffic sign
        recognizer trained on GTSRB dataset.

        Parameters
        ----------
        model      : Keras model
                     Target model to attack
        target     : (optional) bool (default: True)
                     True if performing targeted attack; False, otherwise.
        c          : (optional) float (default: 1)
                     Constant balancing the objective function (f) between norm
                     of perturbation and loss (f = c * loss + norm). Larger c
                     means stronger attack but also more "visible" (stronger
                     perturbation).
        lr         : (optional) float (default: 0.01)
                     Learning rate of optimizer
        init_scl   : (optional) float (default: 0.1)
                     Standard deviation of Gaussian used to initialize objective
                     variable
        use_bound  : (optional) bool (default: False)
                     If True, optimizer with bounding box [0, 1] will be used.
                     Otherwise, Adam optimizer is used.
        loss_op    : (optional) int (default: 0)
                     Option for loss function to optimize over.
                     loss_op = 0: Carlini's l2-attack
                     loss_op = 1: cross-entropy loss
        k          : (optional) float (default: 5)
                     "Confidence threshold" used with loss_op = 0. Used to
                     control strength of attack. The higher the k the stronger
                     the attack.
        var_change : (optional) bool (default: True)
                     If True, objective variable will be changed according to
                     Carlini  et al. (which also gets rid of the need to use
                     any bounding) Otherwise, optimize directly on perturbation.
        use_mask   : (optional) bool (default: True)
                     if True, perturbation will be masked before applying to
                     the target sign. Mask must be specified when calling
                     optimize() and optimize_search().
        batch_size : (optional) int (default: BATCH_SIZE)
                     Define number of transformed images to use
        """

        self.model = model
        self.target = target
        self.c = c
        self.lr = lr
        self.use_bound = use_bound
        self.loss_op = loss_op
        self.k = k
        self.use_mask = use_mask
        self.decay = decay
        self.batch_size = batch_size
        self.var_change = var_change
        self.init_scl = init_scl
        self.rnd_tran = rnd_tran
        self.c_smooth = c_smooth

        # Initialize variables
        init_val = tf.random_uniform(INPUT_SHAPE, minval=-init_scl,
                                     maxval=init_scl, dtype=tf.float32)
        self.x = K.placeholder(name='x', dtype='float32', shape=INPUT_SHAPE)
        self.y = K.placeholder(name='y', dtype='float32',
                               shape=(1, OUTPUT_DIM))
        if self.use_mask:
            self.m = K.placeholder(
                name='m', dtype='float32', shape=INPUT_SHAPE)

        # If change of variable is specified
        if var_change:
            # Optimize on w instead of d
            self.w = tf.Variable(initial_value=init_val, trainable=True,
                                 dtype=tf.float32)
            x_full = (0.5 + EPS) * (tf.tanh(self.w) + 1)
            self.d = x_full - self.x
            if self.use_mask:
                self.d = tf.multiply(self.d, self.m)
            self.x_d = self.x + self.d
            self.var_list = [self.w]
        else:
            # Optimize directly on d (perturbation)
            self.d = tf.Variable(initial_value=init_val, trainable=True,
                                 dtype=tf.float32)
            if self.use_mask:
                dm = tf.multiply(self.d, self.m)
                self.x_in = self.x + dm
            else:
                self.x_in = self.x + self.d
            # Require clipping
            self.x_d = tf.clip_by_value(self.x_in, 0, 1)
            self.var_list = [self.d]

        # Get x_in by transforming x_d by given transformation matrices
        self.M = K.placeholder(name='M', dtype='float32',
                               shape=(self.batch_size, 8))
        x_repeat = tf.tile(self.x_d, [self.batch_size, 1, 1, 1])
        self.x_tran = tf.contrib.image.transform(x_repeat, self.M,
                                                 interpolation='BILINEAR')

        # Randomly adjust brightness
        b = np.multiply((2 * np.random.rand(batch_size, 1) - 1) * rnd_bri,
                        np.ones(shape=(batch_size, N_FEATURE)))
        b = b.reshape((batch_size,) + IMG_SHAPE)
        delta = tf.Variable(initial_value=b, trainable=False, dtype=tf.float32)
        self.x_b = tf.clip_by_value(self.x_tran + delta, 0, 1)

        # Upsample and downsample
        def resize(x):
            tmp = tf.image.resize_images(
                x[0], x[1], method=tf.image.ResizeMethod.BILINEAR)
            return tf.image.resize_images(
                tmp, [HEIGHT, WIDTH], method=tf.image.ResizeMethod.BILINEAR)

        # Set upsampling size to be 600x600 if not specified
        if sp_size is None:
            sp_size = np.zeros((batch_size, 2)) + 600
        up_size = tf.constant(sp_size, dtype=tf.int32)
        # Use map_fn to do random resampling for each image
        self.x_rs = tf.map_fn(resize, (self.x_b, up_size), dtype=tf.float32)

        self.x_in = self.x_rs
        model_output = self.model(self.x_in)
        self.model_output = model_output

        if loss_op == 0:
            # Carlini l2-attack's loss
            # Get 2 largest outputs
            output_2max = tf.nn.top_k(model_output, 2)[0]
            # Find z_i = max(Z[i != y])
            i_y = tf.argmax(self.y, axis=1, output_type=tf.int32)
            i_max = tf.to_int32(tf.argmax(model_output, axis=1))
            z_i = tf.where(tf.equal(i_y, i_max), output_2max[:, 1],
                           output_2max[:, 0])
            if self.target:
                # loss = max(max(Z[i != t]) - Z[t], -K)
                loss = tf.maximum(z_i - model_output[:, i_y[0]], -self.k)
            else:
                # loss = max(Z[y] - max(Z[i != y]), -K)
                loss = tf.maximum(model_output[:, i_y[0]] - z_i, -self.k)
            # Average across batch
            self.loss = tf.reduce_sum(loss) / self.batch_size
        elif loss_op == 1:
            # Cross entropy loss, loss = -log(F(x_t))
            loss_all = K.categorical_crossentropy(
                self.y, model_output, from_logits=True)
            self.loss = tf.reduce_sum(loss_all)
            self.loss /= self.batch_size
            if not self.target:
                # loss = log(F(x_y))
                self.loss *= -1
        else:
            raise ValueError("Invalid loss_op")

        # Regularization term with l2-norm
        if p_norm == "2":
            norm = tf.norm(self.d, ord='euclidean')
        elif p_norm == "1":
            #norm = tf.norm(self.d, ord=1)
            norm = tf.reduce_sum(tf.maximum(tf.abs(self.d) - THRES, 0))
        elif p_norm == "inf":
            norm = tf.norm(self.d, ord=np.inf)
        else:
            raise ValueError("Invalid norm_op")

        d_x = tf.concat([self.d[:, WIDTH - 1:, :],
                         self.d[:, :WIDTH - 1, :]], axis=1)
        d_y = tf.concat([self.d[HEIGHT - 1:, :, :],
                         self.d[:HEIGHT - 1, :, :]], axis=0)
        smooth_x = tf.reduce_sum(tf.square(self.d - d_x))
        smooth_y = tf.reduce_sum(tf.square(self.d - d_y))
        self.smooth = smooth_x + smooth_y

        # Encourage norm to be larger than some value
        self.norm = tf.maximum(norm, l)
        self._setup_opt()

    def _get_rand_transform_matrix(self, image_size, d, batch_size):

        M = np.zeros((batch_size, 8))

        for i in range(batch_size):
            tl_top = random.uniform(-d, d)     # Top left corner, top
            tl_left = random.uniform(-d, d)    # Top left corner, left
            bl_bottom = random.uniform(-d, d)  # Bot left corner, bot
            bl_left = random.uniform(-d, d)    # Bot left corner, left
            tr_top = random.uniform(-d, d)     # Top right corner, top
            tr_right = random.uniform(-d, d)   # Top right corner, right
            br_bottom = random.uniform(-d, d)  # Bot right corner, bot
            br_right = random.uniform(-d, d)   # Bot right corner, right

            transform = ProjectiveTransform()
            transform.estimate(np.array((
                (tl_left, tl_top),
                (bl_left, image_size - bl_bottom),
                (image_size - br_right, image_size - br_bottom),
                (image_size - tr_right, tr_top)
            )), np.array((
                (0, 0),
                (0, image_size),
                (image_size, image_size),
                (image_size, 0)
            )))

            M[i] = transform.params.flatten()[:8]

        return M

    def optimize(self, x, y, n_step=1000, prog=True, mask=None):
        """
        Run optimization attack, produce adversarial example from a batch of
        images transformed from a single sample, x.

        Parameters
        ----------
        x      : np.array
                 Original benign sample
        y      : np.array
                 One-hot encoded target label if <target> was set to True or
                 one-hot encoded true label, otherwise.
        n_step : (optional) int
                 Number of steps to run optimization
        prog   : (optional) bool
                 True if progress should be printed
        mask   : (optional) np.array of 0 or 1, shape=(n_sample, height, width)
                 Mask to restrict gradient update on valid pixels

        Returns
        -------
        x_adv : np.array, shape=INPUT_SHAPE
                Output adversarial example created from x
        """

        with tf.Session() as sess:

            # Create inputs to optimization
            x_ = np.copy(x).reshape(INPUT_SHAPE)
            y_ = np.copy(y).reshape((1, OUTPUT_DIM))

            # Generate a batch of transformed images
            M_ = self._get_rand_transform_matrix(
                WIDTH, np.floor(WIDTH * self.rnd_tran), self.batch_size)

            # Include mask in feed_dict if mask is used
            if self.use_mask:
                # Repeat mask for all channels
                m_ = np.repeat(
                    mask[np.newaxis, :, :, np.newaxis], N_CHANNEL, axis=3)
                feed_dict = {self.x: x_, self.y: y_, self.M: M_, self.m: m_,
                             K.learning_phase(): False}
            else:
                feed_dict = {self.x: x_, self.y: y_, self.M: M_,
                             K.learning_phase(): False}

            # Initialize variables and load weights
            sess.run(tf.global_variables_initializer())
            self.model.load_weights(WEIGTHS_PATH)
            if self.var_change:
                # Initialize w here to be within L1-ball center at rctanh(2x-1)
                init_rand = np.random.uniform(
                    -self.init_scl, self.init_scl, size=INPUT_SHAPE)
                # Clip values to remove numerical error atanh(1) or atanh(-1)
                tanhw = np.clip(x_ * 2 - 1, -1 + EPS, 1 - EPS)
                init_w = np.arctanh(tanhw) + init_rand
                self.w.load(init_w)
                # self.w.load(init_rand)

            # Set up some variables for early stopping
            min_norm = float("inf")
            min_d = None
            earlystop_count = 0

            # Start optimization
            for step in range(n_step):
                if self.use_bound:
                    self.optimizer.minimize(sess, feed_dict=feed_dict)
                else:
                    sess.run(self.opt, feed_dict=feed_dict)
                    #return sess.run(self.d_x, feed_dict=feed_dict)

                # Keep track of "best" solution
                if self.loss_op == 0:
                    norm = sess.run(self.norm, feed_dict=feed_dict)
                    loss = sess.run(self.loss, feed_dict=feed_dict)
                    # Save working adversarial example with smallest norm
                    if loss < -0.95 * self.k:
                        if norm < min_norm:
                            min_norm = norm
                            min_d = sess.run(self.d, feed_dict=feed_dict)
                            # Reset early stopping counter
                            earlystop_count = 0
                        else:
                            earlystop_count += 1
                            # Early stop if no improvement
                            if earlystop_count > EARLYSTOP_STEPS:
                                print(step, min_norm)
                                break

                # Print progress
                if (step % PROG_PRINT_STEPS == 0) and prog:
                    f = sess.run(self.f, feed_dict=feed_dict)
                    norm = sess.run(self.norm, feed_dict=feed_dict)
                    loss = sess.run(self.loss, feed_dict=feed_dict)
                    smooth = sess.run(self.smooth, feed_dict=feed_dict)
                    print("Step: {}, norm={:.3f}, loss={:.3f}, smooth={:.3f}, obj={:.3f}".format(
                        step, norm, loss, smooth, f))
                    # print(sess.run(self.model_output, feed_dict=feed_dict)[0])

            if min_d is not None:
                x_adv = (x_ + min_d).reshape(IMG_SHAPE)
                return x_adv, min_norm
            else:
                d = sess.run(self.d, feed_dict=feed_dict)
                norm = sess.run(self.norm, feed_dict=feed_dict)
                x_adv = (x_ + d).reshape(IMG_SHAPE)
                return x_adv, norm

    def optimize_search(self, x, y, n_step=1000, search_step=10, prog=True,
                        mask=None):
        """
        Run optimization attack, produce adversarial example from a batch of
        images transformed from a single sample, x. Does binary search on
        log_10(c) to find optimal value of c.

        Parameters
        ----------
        x      : np.array
                 Original benign sample
        y      : np.array, shape=(OUTPUT_DIM,)
                 One-hot encoded target label if <target> was set to True or
                 one-hot encoded true label, otherwise.
        n_step : (optional) int
                 Number of steps to run optimization
        search_step : (optional) int
                      Number of steps to search on c
        prog   : (optional) bool
                 True if progress should be printed
        mask   : (optional) np.array of 0 or 1, shape=(n_sample, height, width)
                 Mask to restrict gradient update on valid pixels

        Returns
        -------
        x_adv_suc : np.array, shape=INPUT_SHAPE
                    Successful adversarial example created from x. None if fail.
        norm_suc  : float
                    Perturbation magnitude of x_adv_suc. None if fail.
        """

        # Declare min-max of search line [1e-2, 1e2] for c = 1e(cp)
        cp_lo = MIN_CP
        cp_hi = MAX_CP

        x_adv_suc = None
        norm_suc = float("inf")
        start_time = time.time()

        # Binary search on cp
        for c_step in range(search_step):

            # Update c
            cp = (cp_lo + cp_hi) / 2
            self.c = 10 ** cp
            self._setup_opt()

            # Run optimization with new c
            x_adv, norm = self.optimize(
                x, y, n_step=n_step, prog=False, mask=mask)

            # Evaluate result
            y_pred = self.model.predict(x_adv.reshape(INPUT_SHAPE))[0]
            score = softmax(y_pred)[np.argmax(y)]
            if self.target:
                if score > SCORE_THRES:
                    # Attack succeeded, decrease cp to lower norm
                    cp_hi = cp
                    # Only save adv example if norm becomes smaller
                    if norm < norm_suc:
                        x_adv_suc = np.copy(x_adv)
                        norm_suc = norm
                else:
                    # Attack failed, increase cp for stronger attack
                    cp_lo = cp
            else:
                if score > 1 - SCORE_THRES:
                    # Attack failed, increase cp for stronger attack
                    cp_lo = cp
                else:
                    # Attack succeeded, decrease cp to lower norm
                    cp_hi = cp
                    # Only save adv example if norm becomes smaller
                    if norm < norm_suc:
                        x_adv_suc = np.copy(x_adv)
                        norm_suc = norm
            if prog:
                print("c_Step: {}, c={:.4f}, score={:.3f}, norm={:.3f}".format(
                    c_step, self.c, score, norm))
        print("Finished in {:.2f}s".format(time.time() - start_time))
        return x_adv_suc, norm_suc


