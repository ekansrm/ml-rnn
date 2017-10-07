"""PTB LSTM model.

"""

import os
import time

DATA_PATH = os.path.abspath('../data/ptb/data')

import numpy as np
import tensorflow as tf
from enum import Enum

from data.ptb.interface.tensorflow import reader

from data.ptb.interface.tensorflow import util

from tensorflow.python.client import device_lib

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", 1,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")

FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name)


class PTBModel(object):
    """
    PTB 模型
    包括模型的构建
    一个模型包含流图, 张量成员, 这样我就可以用直接访问成员变量的方法去获取张量赫尔运算! 函数+闭包也行.
    """


    class Config(object):
        """
        模型配置:
        配置是在build之前就得配置好的

        """
        class Mode(Enum):
            BASIC = "basic"
            CUDNN = "cudnn"
            BLOCK = "block"

        init_scale = 0.1
        learning_rate = 1.0
        max_grad_norm = 5
        num_layers = 2
        num_steps = 20
        hidden_size = 200
        max_epoch = 4
        max_max_epoch = 13
        keep_prob = 1.0
        lr_decay = 0.5
        batch_size = 20
        vocab_size = 10000
        rnn_mode = Mode.BASIC

    def __init__(self, is_training: bool, config: Config, input_):
        """
        :type is_training: bool
        """
        self._is_training = is_training
        self._input = input_
        self._rnn_params = None
        self._cell = None
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps

        self.build(is_training=is_training, config=config, input_= input_)

    def config(self, config: Config):
        pass

    def _build_rnn_graph(self, inputs, config: Config, is_training: bool):
        if config.rnn_mode == CUDNN:
            return self._build_rnn_graph_cudnn(inputs, config, is_training)
        else:
            return self._build_rnn_graph_lstm(inputs, config, is_training)

    def _build_rnn_graph_cudnn(self, inputs, config: Config, is_training: bool):
        """Build the inference graph using CUDNN cell."""
        inputs = tf.transpose(inputs, [1, 0, 2])
        self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=config.num_layers,
            num_units=config.hidden_size,
            input_size=config.hidden_size,
            dropout=1 - config.keep_prob if is_training else 0)
        params_size_t = self._cell.params_size()
        self._rnn_params = tf.get_variable(
            "lstm_params",
            initializer=tf.random_uniform(
                [params_size_t], -config.init_scale, config.init_scale),
            validate_shape=False)
        c = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                     tf.float32)
        h = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                     tf.float32)
        self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
        outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.reshape(outputs, [-1, config.hidden_size])
        return outputs, (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

    def _build_rnn_graph_lstm(self, inputs, config: Config, is_trainnig: bool):
        """
        使用标准的 LSTM 单元构造流图
        :param inputs:
        :param config:
        :param is_trainnig:
        :return:
        """
        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.

        # 构造一个单元, 可以理解为一层
        lstm_cell = self._make_lstm_cell(config=config, is_training=is_trainnig)
        if is_trainnig and config.keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                cell=lstm_cell,
                output_keep_prob=config.keep_prob
            )

        lstm_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell] * config.num_layers
        )

        self._initial_state = lstm_cells.zero_state(config.batch_size, data_type())
        state = self._initial_state

        # 讲输入输入 LSTM的cell, 然后获取输出和最后状态
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if time_step > 0: # 如果不是第一次, 就把当前变量空间设置为 reuse
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = lstm_cells(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(values=outputs, axis=1), [-1, config.hidden_size])

        return output, state

    def _make_lstm_cell(self, config: Config, is_training: bool):
        """
        根据配置, 生产对应的 LSTM 单元. BASIC|BLOCK
        :param config:
        :param is_training:
        :return:
        """
        if config.rnn_mode == BASIC:
            return tf.contrib.rnn.BasicLSTMCell(
                num_units=config.hidden_size,
                forget_bias=0.0,
                state_is_tuple=True, # If True, accepted and returned states are 2-tuples of the c_state and m_state. If False, they are concatenated along the column axis. The latter behavior will soon be deprecated.
                reuse=not is_training
            )
        if config.rnn_mode == BLOCK:
            return tf.contrib.rnn.LSTMBlockCell(
                num_units=config.hidden_size,
                forget_bias=0.0,
            )
        raise ValueError("RNN 模式 'rnn_mode' 不支持 '%s' ", config.rnn_mode)

    def build(self, is_training: bool, input_, config: Config):
        self._is_training = is_training
        self._input = input_
        self._rnn_params = None
        self._cell = None
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size

        with tf.device("/cpu:0"):
            embedding = tf.get_variable(            # 词向量, 在词典的每个单词, 都对应一个词向量
                name="embedding",                   # - 一个批次的输入 [batch_size, nums_step]
                shape=[vocab_size, hidden_size],    # - 输出为 [batch_size, nums_step, hidden_size]
                dtype=data_type()
            )
            inputs = tf.nn.embedding_lookup(
                params=embedding,               # params会成为table, axis 0 成为id, 余下的张量为element
                ids=self._input.input_data      # ids里的每个指会被索引成params的element, 新的张量的shape为, [ *old_shape, *element_shape ]
            )

        # 如果是训练, 需要加 dropout 层
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(x=inputs, keep_prob=config.keep_prob)

        # 构建RNN
        output, state = self._build_rnn_graph(inputs, config, is_training=is_training)

        # 构建全连接层, 从 词向量获取词典每一个词成为下一个词的概率
        # 所以, 权重的shape [ 词向量长度, 词典大小 ]
        softmax_w = tf.get_variable(
            name="softmax_w",
            shape=[hidden_size, vocab_size],
            dtype=data_type()
        )
        softmax_b = tf.get_variable(
            name="softmax_b",
            shape=[vocab_size],
            dtype=data_type()
        )
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])


        # 使用tensorflow的函数计算序列交叉熵
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            input_.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
            average_across_timesteps=False,
            average_across_batch=True
        )

        # 每个batch的损失
        self._cost = tf.reduce_sum(loss)

        # 更新状态
        self._final_state = state

        # 只在训练模型时定义反向传播操作
        if not is_training:
            return

        # 梯度裁剪
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self._cost, tvars),
            config.max_grad_norm
        )

        # 定义训练操作
        self._lr = tf.Variable(0.0, trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        # 动态更新学习率
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def export_ops(self, name):
        """Exports ops to collections."""
        self._name = name
        ops = {util.with_prefix(self._name, "cost"): self._cost}
        if self._is_training:
            ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
            if self._rnn_params:
                ops.update(rnn_params=self._rnn_params)
        for name, op in ops.items():
            tf.add_to_collection(name, op)
        self._initial_state_name = util.with_prefix(self._name, "initial")
        self._final_state_name = util.with_prefix(self._name, "final")
        util.export_state_tuples(self._initial_state, self._initial_state_name)
        util.export_state_tuples(self._final_state, self._final_state_name)

    def import_ops(self):
        """Imports ops from collections."""
        if self._is_training:
            self._train_op = tf.get_collection_ref("train_op")[0]
            self._lr = tf.get_collection_ref("lr")[0]
            self._new_lr = tf.get_collection_ref("new_lr")[0]
            self._lr_update = tf.get_collection_ref("lr_update")[0]
            rnn_params = tf.get_collection_ref("rnn_params")
            if self._cell and rnn_params:
                params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
                    self._cell,
                    self._cell.params_to_canonical,
                    self._cell.canonical_to_params,
                    rnn_params,
                    base_variable_scope="Model/RNN")
                tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
        self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
        num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
        self._initial_state = util.import_state_tuples(
            self._initial_state, self._initial_state_name, num_replicas)
        self._final_state = util.import_state_tuples(
            self._final_state, self._final_state_name, num_replicas)

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def initial_state_name(self):
        return self._initial_state_name

    @property
    def final_state_name(self):
        return self._final_state_name


class SmallConfig(PTBModel.Config):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    rnn_mode = CUDNN


class MediumConfig(PTBModel.Config):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


class LargeConfig(PTBModel.Config):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


class TestConfig(PTBModel.Config):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
                   (time.time() - start_time)))

    return np.exp(costs / iters)


def get_config():
    """Get model config."""
    config = None
    if FLAGS.model == "small":
        config = SmallConfig()
    elif FLAGS.model == "medium":
        config = MediumConfig()
    elif FLAGS.model == "large":
        config = LargeConfig()
    elif FLAGS.model == "test":
        config = TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)
    if FLAGS.rnn_mode:
        config.rnn_mode = FLAGS.rnn_mode
    if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0" :
        config.rnn_mode = BASIC
    return config


def main(_):
    # if not FLAGS.data_path:
    #     raise ValueError("Must set --data_path to PTB data directory")
    gpus = [
        x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"
    ]
    if FLAGS.num_gpus > len(gpus):
        raise ValueError(
            "Your machine has only %d gpus "
            "which is less than the requested --num_gpus=%d."
            % (len(gpus), FLAGS.num_gpus))

    raw_data = reader.ptb_raw_data(DATA_PATH)
    train_data, valid_data, test_data, _ = raw_data

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config, input_=train_input)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        with tf.name_scope("Valid"):
            valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
            tf.summary.scalar("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            test_input = PTBInput(
                config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = PTBModel(is_training=False, config=eval_config,
                                 input_=test_input)

        models = {"Train": m, "Valid": mvalid, "Test": mtest}
        for name, model in models.items():
            model.export_ops(name)
        metagraph = tf.train.export_meta_graph()
        if tf.__version__ < "1.1.0" and FLAGS.num_gpus > 1:
            raise ValueError("num_gpus > 1 is not supported for TensorFlow versions "
                             "below 1.1.0")
        soft_placement = False
        if FLAGS.num_gpus > 1:
            soft_placement = True
            util.auto_parallel(metagraph, m)

    with tf.Graph().as_default():
        tf.train.import_meta_graph(metagraph)
        for model in models.values():
            model.import_ops()
        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
        with sv.managed_session(config=config_proto) as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                             verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)

            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
    tf.app.run()








