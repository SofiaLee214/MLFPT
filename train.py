import os, sys
sys.path.append(os.getcwd())

import time
import pickle
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime

import utils
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import tflib.plot
import modeltrain

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--training-data', '-i',
                        default='68_linkedin_found_hash_plain.txt',
                        dest='training_data',
                        help='Path to training data file (one password per line)')
    
    parser.add_argument('--output-dir', '-o',
                        required=True,
                        dest='output_dir',
                        help='Output directory')
    
    parser.add_argument('--save-every', '-s',
                        type=int,
                        default=5000,
                        dest='save_every',
                        help='Save model checkpoints after this many iterations')
    
    parser.add_argument('--iters', '-n',
                        type=int,
                        default=200000,
                        dest='iters',
                        help='The number of training iterations')
    
    parser.add_argument('--batch-size', '-b',
                        type=int,
                        default=64,
                        dest='batch_size',
                        help='Batch size')
    
    parser.add_argument('--seq-length', '-l',
                        type=int,
                        default=10,
                        dest='seq_length',
                        help='The maximum password length')
    
    parser.add_argument('--layer-dim', '-d',
                        type=int,
                        default=128,
                        dest='layer_dim',
                        help='The hidden layer dimensionality')
    
    parser.add_argument('--critic-iters', '-c',
                        type=int,
                        default=10,
                        dest='critic_iters',
                        help='The number of discriminator iterations per generator iteration')
    
    parser.add_argument('--lambda', '-p',
                        type=int,
                        default=10,
                        dest='lamb',
                        help='The gradient penalty lambda')
    
    return parser.parse_args()

def load_and_verify_data(path, max_length):
    """Load data and perform verification checks"""
    print(f"\nAttempting to load data from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data file not found: {path}")
    
    file_size = os.path.getsize(path)
    print(f"File size: {file_size/1024/1024:.2f} MB")
    
    # Load the data
    lines, charmap, inv_charmap = utils.load_dataset(path, max_length)
    
    # Verify data
    if len(lines) < 1000:
        print("\nWARNING: Dataset is very small. Consider using a larger dataset for better results.")
    
    print(f"\nDataset Statistics:")
    print(f"Number of samples: {len(lines):,}")
    print(f"Vocabulary size: {len(charmap)}")
    print("Character set:", sorted(charmap.keys()))
    
    # Sample a few lines
    print("\nSample lines from dataset:")
    for line in lines[:5]:
        print("".join(line))
        
    return lines, charmap, inv_charmap

def main():
    args = parse_args()
    
    # Disable eager execution
    tf.compat.v1.disable_eager_execution()
    
    # Load and verify data
    lines, charmap, inv_charmap = load_and_verify_data(args.training_data, args.seq_length)
    
    # Create output directories
    for d in [args.output_dir, 
             os.path.join(args.output_dir, 'checkpoints'),
             os.path.join(args.output_dir, 'samples')]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    # Save character mappings
    with open(os.path.join(args.output_dir, 'charmap.pickle'), 'wb') as f:
        pickle.dump(charmap, f)
    with open(os.path.join(args.output_dir, 'inv_charmap.pickle'), 'wb') as f:
        pickle.dump(inv_charmap, f)
    
    print("\nBuilding model...")
    # Create the model
    real_inputs_discrete = tf.compat.v1.placeholder(tf.int32, shape=[args.batch_size, args.seq_length])
    real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
    fake_inputs = modeltrain.Generator(args.batch_size, args.seq_length, args.layer_dim, len(charmap))
    fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims-1)
    
    disc_real = modeltrain.Discriminator(real_inputs, args.seq_length, args.layer_dim, len(charmap))
    disc_fake = modeltrain.Discriminator(fake_inputs, args.seq_length, args.layer_dim, len(charmap))
    
    # WGAN loss
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    gen_cost = -tf.reduce_mean(disc_fake)
    
    # Gradient penalty
    alpha = tf.random.uniform(shape=[args.batch_size, 1, 1], minval=0., maxval=1.)
    differences = fake_inputs - real_inputs
    interpolates = real_inputs + (alpha * differences)
    gradients = tf.gradients(
        modeltrain.Discriminator(interpolates, args.seq_length, args.layer_dim, len(charmap)),
        [interpolates]
    )[0]
    
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
    gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
    disc_cost += args.lamb * gradient_penalty
    
    # Optimizers
    gen_params = lib.params_with_name('Generator')
    disc_params = lib.params_with_name('Discriminator')
    
    gen_train_op = tf.compat.v1.train.AdamOptimizer(
        learning_rate=1e-4, beta1=0.5, beta2=0.9
    ).minimize(gen_cost, var_list=gen_params)
    
    disc_train_op = tf.compat.v1.train.AdamOptimizer(
        learning_rate=1e-4, beta1=0.5, beta2=0.9
    ).minimize(disc_cost, var_list=disc_params)
    
    # Dataset iterator
    def inf_train_gen():
        while True:
            np.random.shuffle(lines)
            for i in range(0, len(lines)-args.batch_size+1, args.batch_size):
                yield np.array([[charmap[c] for c in l] for l in lines[i:i+args.batch_size]], dtype='int32')
    
    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    
    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())
        
        def generate_samples():
            samples = session.run(fake_inputs)
            samples = np.argmax(samples, axis=2)
            return [''.join(inv_charmap[c] for c in sample) for sample in samples]
        
        gen = inf_train_gen()
        
        for iteration in range(args.iters):
            iter_start_time = time.time()
            
            # Train generator
            if iteration > 0:
                _ = session.run(gen_train_op)
            
            # Train critic
            disc_costs = []
            for _ in range(args.critic_iters):
                _data = next(gen)
                _disc_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={real_inputs_discrete: _data}
                )
                disc_costs.append(_disc_cost)
            
            # Progress monitoring
            if iteration % 10 == 0:
                elapsed = time.time() - start_time
                remaining = (elapsed / (iteration + 1)) * (args.iters - iteration - 1)
                print(f"\rIteration {iteration}/{args.iters} "
                      f"[{elapsed/3600:.1f}h elapsed, {remaining/3600:.1f}h remaining] "
                      f"disc_cost = {np.mean(disc_costs):.4f}", end="")
            
            # Generate samples
            if iteration % 100 == 0 and iteration > 0:
                samples = generate_samples()
                with open(os.path.join(args.output_dir, 'samples', f'samples_{iteration}.txt'), 'w') as f:
                    for sample in samples:
                        f.write(sample + '\n')
            
            # Save model
            if iteration % args.save_every == 0 and iteration > 0:
                checkpoint_path = os.path.join(args.output_dir, 'checkpoints', f'model_{iteration}.ckpt')
                saver = tf.compat.v1.train.Saver()
                saver.save(session, checkpoint_path)
                print(f"\nModel saved to {checkpoint_path}")

if __name__ == '__main__':
    main()