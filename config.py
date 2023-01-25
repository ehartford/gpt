import configparser
config = configparser.ConfigParser()
config.read('config.ini')
default = config['DEFAULT']
config.batch_size = int(default['batch_size'])
config.block_size = int(default['block_size'])
config.max_iters = int(default['max_iters'])
config.eval_interval = int(default['eval_interval'])
config.learning_rate = float(default['learning_rate'])
config.eval_iters = int(default['eval_iters'])
config.n_embd = int(default['n_embd'])
config.n_head = int(default['n_head'])
config.n_layer = int(default['n_layer'])
config.dropout = float(default['dropout'])

with open('input.txt', 'r', encoding='utf8') as f:
    config.chars = sorted(list(set(f.read())))
config.vocab_size = len(config.chars)
