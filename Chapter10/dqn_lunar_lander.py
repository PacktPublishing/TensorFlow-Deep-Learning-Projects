import gym
from gym import wrappers
import numpy as np
import random, tempfile, os
from collections import deque
import tensorflow as tf


class Brain:
    """
    A Q-Value approximation obtained using a neural network.
    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, nS, nA, scope="estimator",
                 learning_rate=0.0001,
                 neural_architecture=None,
                 global_step=None, summaries_dir=None):

        self.nS = nS
        self.nA = nA
        self.global_step = global_step
        self.scope = scope
        self.learning_rate = learning_rate

        if not neural_architecture:
            neural_architecture = self.two_layers_network

        # Writes Tensorboard summaries to disk
        with tf.variable_scope(scope):
            # Build the graph
            self.create_network(network=neural_architecture, learning_rate=self.learning_rate)
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_%s" % scope)
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)
            else:
                self.summary_writer = None

    def two_layers_network(self, x, layer_1_nodes=32, layer_2_nodes=32):
        """
        A simple ANN
        """
        layer_1 = tf.contrib.layers.fully_connected(x, layer_1_nodes, activation_fn=tf.nn.relu)
        layer_2 = tf.contrib.layers.fully_connected(layer_1, layer_2_nodes, activation_fn=tf.nn.relu)
        return tf.contrib.layers.fully_connected(layer_2, self.nA, activation_fn=None)

    def create_network(self, network, learning_rate=0.0001):
        """
        Building the Tensorflow graph.
        """

        # Placeholders for states input
        self.X = tf.placeholder(shape=[None, self.nS], dtype=tf.float32, name="X")
        # The r target value
        self.y = tf.placeholder(shape=[None, self.nA], dtype=tf.float32, name="y")

        # Applying the choosen network
        self.predictions = network(self.X)

        # Calculating the loss
        sq_diff = tf.squared_difference(self.y, self.predictions)
        self.loss = tf.reduce_mean(sq_diff)

        # Optimizing parameters using the Adam optimizer
        self.train_op = tf.contrib.layers.optimize_loss(self.loss, global_step=tf.train.get_global_step(),
                                                        learning_rate=learning_rate, optimizer='Adam')

        # Recording summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions)),
            tf.summary.scalar("mean_q_value", tf.reduce_mean(self.predictions))
        ])

    def predict(self, sess, s):
        """
        Predicting q values for actions
        """
        return sess.run(self.predictions, {self.X: s})

    def fit(self, sess, s, r, epochs=1):
        """
        Updating the Q* function estimator
        """
        feed_dict = {self.X: s, self.y: r}

        for epoch in range(epochs):
            res = sess.run([self.summaries, self.train_op, self.loss, self.predictions,
                            tf.train.get_global_step()], feed_dict)
            summaries, train_op, loss, predictions, self.global_step = res

        if self.summary_writer:
            self.summary_writer.add_summary(summaries, self.global_step)


class Memory:
    """
    A memory class based on deque, a list-like container with fast appends and
    pops on either end (from the collections package)
    """
    def __init__(self, memory_size=5000):
        self.memory = deque(maxlen=memory_size)

    def __len__(self):
        return len(self.memory)

    def add_memory(self, s, a, r, s_, status):
        """
        Memorizing the tuple (s a r s_) plus the Boolean flag status,
        reminding if we are at a terminal move or not
        """
        self.memory.append((s, a, r, s_, status))

    def recall_memories(self):
        """
        Returning all the memorized data at once
        """
        return list(self.memory)


class Agent:
    def __init__(self, nS, nA, experiment_dir):
        # Initializing
        self.nS = nS
        self.nA = nA
        self.epsilon = 1.0  # exploration-exploitation ratio
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9994
        self.gamma = 0.99  # reward decay
        self.learning_rate = 0.0001
        self.epochs = 1  # training epochs
        self.batch_size = 32
        self.memory = Memory(memory_size=250000)

        # Creating estimators
        self.experiment_dir = os.path.abspath("./experiments/{}".format(experiment_dir))
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.model = Brain(nS=self.nS, nA=self.nA, scope="q",
                           learning_rate=self.learning_rate,
                           global_step=self.global_step,
                           summaries_dir=self.experiment_dir)
        self.target_model = Brain(nS=self.nS, nA=self.nA, scope="target_q",
                                  learning_rate=self.learning_rate,
                                  global_step=self.global_step)

        # Adding an op to initialize the variables.
        init_op = tf.global_variables_initializer()

        # Adding ops to save and restore all the variables.
        self.saver = tf.train.Saver()

        # Setting up the session
        self.sess = tf.Session()
        self.sess.run(init_op)

    def epsilon_update(self, t):
        """
        Updating epsilon based on experienced episodes
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_weights(self, filename):
        """
        Saving the weights of a model
        """
        save_path = self.saver.save(self.sess, "%s.ckpt" % filename)
        print("Model saved in file: %s" % save_path)

    def load_weights(self, filename):
        """
        Restoring the weights of a model
        """
        self.saver.restore(self.sess, "%s.ckpt" % filename)
        print("Model restored from file")

    def set_weights(self, model_1, model_2):
        """
        Replicates the model parameters of one estimator to another.
          model_1: Estimator to copy the parameters from
          model_2: Estimator to copy the parameters to
        """

        # Enumerating and sorting the parameters of the two models
        model_1_params = [t for t in tf.trainable_variables() if t.name.startswith(model_1.scope)]
        model_2_params = [t for t in tf.trainable_variables() if t.name.startswith(model_2.scope)]
        model_1_params = sorted(model_1_params, key=lambda x: x.name)
        model_2_params = sorted(model_2_params, key=lambda x: x.name)

        # Enumerating the operations to be done
        operations = [coef_2.assign(coef_1) for coef_1, coef_2 in zip(model_1_params, model_2_params)]
        # Executing the operations to be done
        self.sess.run(operations)

    def target_model_update(self):
        """
        Setting the model weights to the target model's ones
        """
        self.set_weights(self.model, self.target_model)

    def act(self, s):
        """
        Having the agent act based on learned Q* function
        or by random choice (based on epsilon)
        """
        # Based on epsilon predicting or randomly choosing the next action
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.nA)
        else:
            # Estimating q for all possible actions
            q = self.model.predict(self.sess, s)[0]
            # Returning the best action
            best_action = np.argmax(q)
            return best_action

    def replay(self):
        # Picking up a random batch from memory
        batch = np.array(random.sample(self.memory.recall_memories(), self.batch_size))
        # Retrieving the sequence of present states
        s = np.vstack(batch[:, 0])
        # Recalling the sequence of actions
        a = np.array(batch[:, 1], dtype=int)
        # Recalling the rewards
        r = np.copy(batch[:, 2])
        # Recalling the sequence of resulting states
        s_p = np.vstack(batch[:, 3])
        # Checking if the reward is relative to a not terminal state
        status = np.where(batch[:, 4] == False)

        # We use the model to predict the rewards by our model and the target model
        next_reward = self.model.predict(self.sess, s_p)
        final_reward = self.target_model.predict(self.sess, s_p)

        if len(status[0]) > 0:
            # Non-terminal update rule using the target model
            # If a reward is not from a terminal state, the reward is just a partial one (r0)
            # We should add the remaining and obtain a final reward using target predictions
            best_next_action = np.argmax(next_reward[status, :][0], axis=1)
            # adding the discounted final reward
            r[status] += np.multiply(self.gamma, final_reward[status, best_next_action][0])

        # We replace the expected rewards for actions when dealing with observed actions and rewards
        expected_reward = self.model.predict(self.sess, s)
        expected_reward[range(self.batch_size), a] = r

        # We re-fit status against predicted/observed rewards
        self.model.fit(self.sess, s, expected_reward, epochs=self.epochs)


class Environment:
    def __init__(self, game="LunarLander-v2"):
        # Initializing
        np.set_printoptions(precision=2)
        self.env = gym.make(game)
        self.env = wrappers.Monitor(self.env, tempfile.mkdtemp(), force=True, video_callable=False)
        self.nS = self.env.observation_space.shape[0]
        self.nA = self.env.action_space.n
        self.agent = Agent(self.nS, self.nA, self.env.spec.id)

        # Cumulative reward
        self.reward_avg = deque(maxlen=100)

    def test(self):
        """
        Routine for testing the learned Q* function
        """
        self.learn(epsilon=0.0, episodes=100, trainable=False, incremental=False)

    def train(self, epsilon=1.0, episodes=1000):
        """
        Routine for training an approximate Q* function
        """
        self.learn(epsilon=epsilon, episodes=episodes, trainable=True, incremental=False)

    def incremental(self, epsilon=0.01, episodes=100):
        """
        Routine for carrying on learning an approximate Q* function
        """
        self.learn(epsilon=epsilon, episodes=episodes, trainable=True, incremental=True)

    def learn(self, epsilon=None, episodes=1000, trainable=True, incremental=False):
        """
        Representing the interaction between the enviroment and the learning agent
        """

        # Restoring weights if required
        if not trainable or (trainable and incremental):
            try:
                print("Loading weights")
                self.agent.load_weights('./weights.h5')
            except:
                print("Exception")
                trainable = True
                incremental = False
                epsilon = 1.0

        # Setting epsilon
        self.agent.epsilon = epsilon

        # Iterating through episodes
        for episode in range(episodes):
            # Initializing a new episode
            episode_reward = 0
            s = self.env.reset()
            # s is put at default values
            s = np.reshape(s, [1, self.nS])

            # Iterating through time frames
            for time_frame in range(1000):

                if not trainable:
                    # If not learning, representing the agent on video
                    self.env.render()

                # Deciding on the next action to take
                a = self.agent.act(s)

                # Performing the action and getting feedback
                s_p, r, status, info = self.env.step(a)
                s_p = np.reshape(s_p, [1, self.nS])

                # Adding the reward to the cumualtive reward
                episode_reward += r

                # Adding the overall experience to memory
                if trainable:
                    self.agent.memory.add_memory(s, a, r, s_p, status)

                # Setting the new state as the current one
                s = s_p

                # Performing experience replay if memory length is greater than the batch length
                if trainable:
                    if len(self.agent.memory) > self.agent.batch_size:
                        self.agent.replay()

                # When the episode is completed, exiting this loop
                if status:
                    if trainable:
                        self.agent.target_model_update()
                    break

            # Exploration vs exploitation
            self.agent.epsilon_update(episode)

            # Running an average of the past 100 episodes
            self.reward_avg.append(episode_reward)
            print("episode: %i score: %.2f avg_score: %.2f"
                  "actions %i epsilon %.2f" % (episode,
                                        episode_reward,
                           np.average(self.reward_avg),
                                            time_frame,
                                               epsilon))
        self.env.close()
                  
        if trainable:
            # Saving the weights for the future
            self.agent.save_weights('./weights.h5')

if __name__ == "__main__":

    lunar_lander = Environment(game="LunarLander-v2")

    lunar_lander.train(epsilon=1.0, episodes=5000)

    #lunar_lander.incremental(episodes=50, epsilon=0.01)

    lunar_lander.test()
