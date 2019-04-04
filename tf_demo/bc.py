import tensorflow as tf 
import numpy as np
import argparse
import os
import shutil
import gym
import getch

from baselines.common.tf_util import load_variables, save_variables, initialize

def log_histogram(values, step, bins=1000):
    """Logs the histogram of a list/vector of values."""
    # Convert to a numpy array
    values = np.array(values)
    
    # Create histogram using numpy        
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Thus, we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)
    return hist

def get_options():
    parser = argparse.ArgumentParser(description='Clone some expert data..')
    parser.add_argument('bc_data', type=str,
        help="The main datastore for this particular expert.")
    parser.add_argument('model_dir', type=str, help="Folder for weights and tensorboard stuff")
    parser.add_argument('model_weights', type=str, help="File name for model weights")

    args = parser.parse_args()
    return args


def process_data(bc_data_dir):
    """
    Runs training for the agent.
    """
    # Load the file store. 
    # In the future (TODO) move this to a seperate thread.
    states, actions = [], []
    shards = [x for x in os.listdir(bc_data_dir) if x.endswith('.npy')]
    print("Processing shards: {}".format(shards))
    for shard in shards:
        shard_path = os.path.join(bc_data_dir, shard)
        with open(shard_path, 'rb') as f:
            data = np.load(f)
            shard_states, unprocessed_actions, rewards, dones = zip(*data)
            shard_states = [x.flatten() for x in shard_states]
            
            # Add the shard to the dataset
            states.extend(shard_states)
            actions.extend(unprocessed_actions)

    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    print("Processed with {} pairs".format(len(states)))
    return states, actions

def create_model():
    """
    Creates the model.
    """
    #state_ph = tf.placeholder(tf.float32, shape=[None, 2])
    state_ph = tf.placeholder(tf.float32, shape=[None, 8])
    # Process the data

    # # Hidden neurons
    with tf.variable_scope("layer1"):
        hidden = tf.layers.dense(state_ph, 64, activation=tf.nn.relu)

    with tf.variable_scope("layer2"):
        hidden = tf.layers.dense(hidden, 64, activation=tf.nn.relu)

    # Make output layers
    with tf.variable_scope("layer3"):
        #logits = tf.layers.dense(hidden, 2) 
        logits = tf.layers.dense(hidden, 4) 
    # Take the action with the highest activation
    with tf.variable_scope("output"):
        action = tf.argmax(input=logits, axis=1)

    return state_ph, action, logits

def create_training(logits):
    """
    Creates the model.
    """
    label_ph = tf.placeholder(tf.int32, shape=[None])

    # Convert it to a onehot. 1-> [1,0,0,0]
    with tf.variable_scope("loss"):
        onehot_labels = tf.one_hot(indices=tf.cast(label_ph, tf.int32), depth=4) # 4 actions

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)
        loss = tf.reduce_mean(loss)

        tf.summary.scalar('loss', loss)

    with tf.variable_scope("training"):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        train_op = optimizer.minimize(loss=loss)

    return train_op, loss, label_ph

def run_main(opts):
    if os.path.exists(opts.model_dir):
        print('Path already exists. Remove? y for yes')
        input_char = getch.getch()
        if not input_char == 'y':
            print('Exiting')
            return
        shutil.rmtree(opts.model_dir)
    os.makedirs(opts.model_dir)
    os.makedirs(os.path.join(opts.model_dir, 'logs'))
    os.makedirs(os.path.join(opts.model_dir, 'weights'))

    # Create the environment with specified arguments
    state_data, action_data = process_data(opts.bc_data)

    #env = gym.make('MountainCar-v0')
    env = gym.make('LunarLander-v2')
    env._max_episode_steps = 1200

    x, model, logits = create_model()
    train, loss, labels = create_training(logits)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

    # Create summaries
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(opts.model_dir, 'logs'), sess.graph)

    print(tf.global_variables())
    sess.run(tf.global_variables_initializer())

    update = 0
    save_freq = 1000
    while True:
        for _ in range (25):
            # Get a random batch from the data
            batch_index = np.random.choice(len(state_data), 64) #Batch size
            state_batch, action_batch = state_data[batch_index], action_data[batch_index]

            # Train the model.
            _, cur_loss, cur_summaries = sess.run([train, loss, merged], feed_dict={
                x: state_data,
                labels: action_data
                })
            print("Loss: {}".format(cur_loss))
            train_writer.add_summary(cur_summaries, update)
            update += 1

        if update % save_freq == 0:
            save_variables(os.path.join(opts.model_dir, 'weights', opts.model_weights), sess=sess)

        done = False
        obs = env.reset()
        rewards = 0
        action_freq = [0 for _ in range(4)]
        while not done:
            env.render()

            # Handle the toggling of different application states
            action = sess.run(model, feed_dict={
                x: [obs.flatten()]
            })[0]

            action_freq[action] += 1
            
            obs, reward, done, info = env.step(action)
            rewards += reward

        reward_summary = tf.Summary(value=[tf.Summary.Value(tag='reward', simple_value=rewards)])
        act_summary = tf.Summary(value=[tf.Summary.Value(tag='act_distribution', histo=log_histogram(action_freq, 1, bins=4))])
        train_writer.add_summary(reward_summary, update // 25)
        train_writer.add_summary(act_summary, update // 25)

        print("Num updates: {}".format(update))
        print("Total reward: {}".format(rewards))
        print("Action dict: {}".format(action_freq))


if __name__ == "__main__":
    # Parse arguments
    opts = get_options()
    # Start the main thread.
    run_main(opts)
