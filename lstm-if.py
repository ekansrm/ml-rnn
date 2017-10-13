"""
LSTM 模型

"""
import numpy as np
import tensorflow as tf
from enum import Enum


class LSTM(object):
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

        dtype = tf.float32      # 数据类型
        init_scale = 0.1        # embedding 初始权重
        num_layers = 2          # 层数
        hidden_size = 200       # 隐藏层大小
        max_epoch = 4
        max_max_epoch = 13
        keep_prob = 1.0                 # 保持概率
        learning_rate = 1.0             # 学习率
        learning_rate_decay = 0.5       # 学习率衰减
        batch_size = 20
        vocab_size = 10000
        num_steps = 20
        rnn_mode = Mode.BASIC   # RNN模型类型

        is_training = False
        max_grad_norm = 5       # 最大梯度

    def __init__(self):
        """
        """
        self._is_training = False
        self._batch_size = None
        self._num_steps = None
        self._config = LSTM.Config()

        self._rnn_params = None
        self._cell = None

        # 损失函数
        self._cost = None

        # 状态
        self._initial_state_name = None
        self._final_state_name = None
        self._initial_state = None
        self._final_state = None

        # 学习率
        self._lr = None
        self._new_lr = None
        self._op_lr_update = None

        # 训练操作子
        self._op_train = None

    ####################################################################################################################
    # 模型配置

    @property
    def is_training(self):
        return self._is_training

    @is_training.setter
    def is_training(self, is_training: bool):
        self._is_training = is_training

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: Config):
        assert isinstance(config, LSTM.Config), "配置类型错误"

        # 只更新变化了的参数
        self.config = config

    def check(self):
        assert self._config is not None, "模型未配置"

    ####################################################################################################################
    # 模型构建

    def _build_rnn_graph(self):
        if self._config.rnn_mode == LSTM.Config.Mode.CUDNN:
            return self._build_rnn_graph_cudnn()
        else:
            return self._build_rnn_graph_lstm()

    def _build_rnn_graph_cudnn(self):

        """Build the inference graph using CUDNN cell."""
        inputs = tf.transpose(inputs, [1, 0, 2])
        self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=self._config.num_layers,
            num_units=self._config.hidden_size,
            input_size=self._config.hidden_size,
            dropout=1 - self._config.keep_prob if self._is_training else 0)
        params_size_t = self._cell.params_size()
        self._rnn_params = tf.get_variable(
            name="lstm_params",
            initializer=tf.random_uniform([params_size_t], -self._config.init_scale, self._config.init_scale),
            validate_shape=False
        )
        c = tf.zeros([self._config.num_layers, self.batch_size, self._config.hidden_size],
                     tf.float32)
        h = tf.zeros([self._config.num_layers, self.batch_size, self._config.hidden_size],
                     tf.float32)
        self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
        outputs, h, c = self._cell(inputs, h, c, self._rnn_params, self._is_training)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.reshape(outputs, [-1, self._config.hidden_size])
        return outputs, (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

    def _build_rnn_graph_lstm(self):
        """
        使用标准的 LSTM 单元构造流图
        :param inputs:
        :return:
        """
        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.

        # 构造一个单元, 可以理解为一层
        lstm_cell = self._make_lstm_cell()
        if self._is_training and self._config.keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                cell=lstm_cell,
                output_keep_prob=self._config.keep_prob
            )

        lstm_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell] * self._config.num_layers
        )

        self._initial_state = lstm_cells.zero_state(self._config.batch_size, self.config.dtype)
        state = self._initial_state

        # 讲输入输入 LSTM的cell, 然后获取输出和最后状态
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if time_step > 0:  # 如果不是第一次, 就把当前变量空间设置为 reuse
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = lstm_cells(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(values=outputs, axis=1), [-1, self._config.hidden_size])

        return output, state

    def _make_lstm_cell(self):
        """
        根据配置, 生产对应的 LSTM 单元. BASIC|BLOCK
        :return:
        """
        if self._config.rnn_mode == LSTM.Config.Mode.BASIC:
            return tf.contrib.rnn.BasicLSTMCell(
                num_units=self._config.hidden_size,
                forget_bias=0.0,
                state_is_tuple=True, # If True, accepted and returned states are 2-tuples of the c_state and m_state. If False, they are concatenated along the column axis. The latter behavior will soon be deprecated.
                reuse=not self._is_training
            )
        if self._config.rnn_mode == LSTM.Config.Mode.BLOCK:
            return tf.contrib.rnn.LSTMBlockCell(
                num_units=self._config.hidden_size,
                forget_bias=0.0,
            )
        raise ValueError("RNN 模式 'rnn_mode' 不支持 '%s' ", self._config.rnn_mode)

    def build(self):
        self._input = input_

        with tf.device("/cpu:0"):
            embedding = tf.get_variable(            # 词向量, 在词典的每个单词, 都对应一个词向量
                name="embedding",                   # - 一个批次的输入 [batch_size, nums_step]
                shape=[self._config.vocab_size, self._config.hidden_size],    # - 输出为 [batch_size, nums_step, hidden_size]
                dtype=self._config.dtype
            )
            inputs = tf.nn.embedding_lookup(
                params=embedding,               # params会成为table, axis 0 成为id, 余下的张量为element
                ids=self._input.input_data      # ids里的每个指会被索引成params的element, 新的张量的shape为, [ *old_shape, *element_shape ]
            )

        # 如果是训练, 需要加 dropout 层
        if self._is_training and self._config.keep_prob < 1:
            inputs = tf.nn.dropout(x=inputs, keep_prob=self._config.keep_prob)

        # 构建RNN
        output, state = self._build_rnn_graph()

        # 构建全连接层, 从 词向量获取词典每一个词成为下一个词的概率
        # 所以, 权重的shape [ 词向量长度, 词典大小 ]
        softmax_w = tf.get_variable(
            name="softmax_w",
            shape=[self._config.hidden_size, self._config.vocab_size],
            dtype=self._config.dtype
        )
        softmax_b = tf.get_variable(
            name="softmax_b",
            shape=[self._config.vocab_size],
            dtype=self._config.dtype
        )
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        logits = tf.reshape(logits, [self._config.hidden_size, self._config.num_steps, self._config.vocab_size])


        # 使用tensorflow的函数计算序列交叉熵
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            input_.targets,
            tf.ones([self._config.batch_size, self._config.num_steps], dtype=self._config.dtype),
            average_across_timesteps=False,
            average_across_batch=True
        )

        # 每个batch的损失
        self._cost = tf.reduce_sum(loss)

        # 更新状态
        self._final_state = state

        # 只在训练模型时定义反向传播操作
        if not self._is_training:
            return

        # 梯度裁剪
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self._cost, tvars),
            self._config.dtype.max_grad_norm
        )

        # 定义训练操作
        self._lr = tf.Variable(0.0, trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)

        self._op_train = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        # 动态更新学习率
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._op_lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._op_lr_update, feed_dict={self._new_lr: lr_value})

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
            self._op_train = tf.get_collection_ref("train_op")[0]
            self._lr = tf.get_collection_ref("lr")[0]
            self._new_lr = tf.get_collection_ref("new_lr")[0]
            self._op_lr_update = tf.get_collection_ref("lr_update")[0]
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

    ####################################################################################################################
    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def initial_state_name(self):
        return self._initial_state_name

    @property
    def final_state_name(self):
        return self._final_state_name

    @property
    def cost(self):
        return self._cost

    @property
    def input(self):
        return self._input

    @property
    def lr(self):
        return self._lr

    @property
    def op_train(self):
        return self._op_train

