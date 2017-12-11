from lib.keras_utils import *
from lib.utils import *
from parameters_yolo import *
from attack_detector.yad2k.models.keras_yolo import yolo_head

EPS = 1e-10   # Epsilon
MIN_CP = -2.  # Minimum power index of c
MAX_CP = 2.   # Maximum power index of c
SCORE_THRES = 0.5  # Softmax score threshold to consider success of attacks
PROG_PRINT_STEPS = 50  # Print progress every certain steps
EARLYSTOP_STEPS = 5000  # Early stopping if no improvement for certain steps


class OptYolo:
    """
    This class implements adversarial examples generator modeled after Carlini
    et al. (https://arxiv.org/abs/1608.04644). It also includes options to use
    other losses, change of variable, optimizer with bounding box (L-BFGS-B,
    etc.) and mask.
    """

    def _setup_opt(self):
        """Used to setup optimization when c is updated"""

        # obj_func = c * loss + l2-norm(d)
        self.f = self.c * self.loss + self.norm
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
            # decay_lr = tf.train.exponential_decay(
            #   self.lr, global_step, 500, 0.95, staircase=True)
            decay_lr = tf.train.inverse_time_decay(
                self.lr, global_step, 10, 0.01)
            # Use Adam optimizer
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=decay_lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
            self.opt = self.optimizer.minimize(
                self.f, var_list=self.var_list, global_step=global_step)

    def __init__(self, model, anchors, target=True, c=1, lr=0.01, init_scl=0.1,
                 use_bound=False, loss_op=0, k=5, var_change=True,
                 use_mask=True):
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
        """

        self.model = model
        self.target = target
        self.c = c
        self.lr = lr
        self.use_bound = use_bound
        self.loss_op = loss_op
        self.k = k
        self.use_mask = use_mask

        # Initialize variables
        init_val = np.random.normal(scale=init_scl, size=NPUT_SHAPE)
        self.x = K.placeholder(dtype='float32', shape=INPUT_SHAPE)
        self.y = K.placeholder(dtype='float32', shape=(1, 19, 19, 5, 80))
        if self.use_mask:
            self.m = K.placeholder(dtype='float32', shape=INPUT_SHAPE)

        # If change of variable is specified
        if var_change:
            # Optimize on w instead of d
            self.w = tf.Variable(initial_value=init_val, trainable=True,
                                 dtype=tf.float32)
            x_full = (0.5 + EPS) * (tf.tanh(self.w) + 1)
            self.d = x_full - self.x
            if self.use_mask:
                self.d = tf.multiply(self.d, self.m)
            self.x_in = self.x + self.d
            self.var_list = [self.w]
        else:
            # Optimize directly on d (perturbation)
            self.d = tf.Variable(initial_value=init_val, trainable=True,
                                 dtype=tf.float32)
            if self.use_mask:
                self.d = tf.multiply(self.d, self.m)
            self.x_in = self.x + self.d
            self.var_list = [self.d]

        # model_output here is box_class_pred
        _, _, _, model_output = yolo_head(self.model.output, anchors, 80)

        if loss_op == 0:
            # Carlini l2-attack's loss
            # Get 2 largest outputs
            output_2max = tf.nn.top_k(model_output, 2)[0][0]
            # Find z_i = max(Z[i != y])
            i_y = tf.argmax(self.y, axis=1, output_type=tf.int32)[0]
            i_max = tf.to_int32(tf.argmax(model_output, axis=1)[0])
            z_i = tf.cond(tf.equal(i_y, i_max),
                          lambda: output_2max[1], lambda: output_2max[0])
            if self.target:
                # loss = max(max(Z[i != t]) - Z[t], -K)
                self.loss = tf.maximum(z_i - model_output[0, i_y], -self.k)
            else:
                # loss = max(Z[y] - max(Z[i != y]), -K)
                self.loss = tf.maximum(model_output[0, i_y] - z_i, -self.k)
        elif loss_op == 1:
            # Cross entropy loss, loss = -log(F(x_t))
            self.loss = K.categorical_crossentropy(
                self.y, model_output, from_logits=False)[0]
            if not self.target:
                # loss = log(F(x_y))
                self.loss *= -1
        else:
            raise ValueError("Invalid loss_op")

        # Regularization term with l2-norm
        self.norm = tf.norm(self.d, ord='euclidean')

        # Call a helper function to finalize and set up optimizer
        self._setup_opt()

    def optimize(self, x, y, n_step=1000, prog=True, mask=None):
        """
        Run optimization attack, produce adversarial example from a single
        sample, x.

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

            # Initialize variables and load weights
            sess.run(tf.global_variables_initializer())
            self.model.load_weights(WEIGTHS_PATH)

            # Ensure that shape is correct
            x_ = np.copy(x).reshape(INPUT_SHAPE)
            y_ = np.copy(y).reshape((1, 19, 19, 5, 80))

            # Include mask in feed_dict if mask is used
            if self.use_mask:
                m_ = np.repeat(
                    mask[np.newaxis, :, :, np.newaxis], N_CHANNEL, axis=3)
                feed_dict = {self.x: x_, self.y: y_,
                             self.m: m_, K.learning_phase(): False}
            else:
                feed_dict = {self.x: x_, self.y: y_, K.learning_phase(): False}

            # Set up some variables for early stopping for loss_op = 0
            min_norm = float("inf")
            min_d = None
            earlystop_count = 0

            # Start optimization
            for step in range(n_step):
                if self.use_bound:
                    self.optimizer.minimize(sess, feed_dict=feed_dict)
                else:
                    sess.run(self.opt, feed_dict=feed_dict)

                # Keep track of "best" solution
                if self.loss_op == 0:
                    norm = sess.run(self.norm, feed_dict=feed_dict)
                    loss = sess.run(self.loss, feed_dict=feed_dict)
                    # Save working adversarial example with smallest norm
                    if loss == -self.k:
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
                    print("Step: {}, norm={:.3f}, loss={:.3f}, obj={:.3f}".format(
                        step, norm, loss, f))

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
        Run optimization attack, produce adversarial example from a single
        sample, x. Does binary search on log_10(c) to find optimal value of c.

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
        norm_suc = None
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
                    x_adv_suc = np.copy(x_adv)
                    norm_suc = norm
                else:
                    # Attack failed, increase cp for stronger attack
                    cp_lo = cp
            else:
                if score > 1 - SCORE_THRES:
                    # Attack failed
                    cp_lo = cp
                else:
                    # Attack succeeded
                    cp_hi = cp
                    x_adv_suc = np.copy(x_adv)
                    norm_suc = norm
            if prog:
                print("c_Step: {}, c={:.4f}, score={:.3f}".format(
                    c_step, self.c, score))
        print("Finished in {:.2f}s".format(time.time() - start_time))
        return x_adv_suc, norm_suc
