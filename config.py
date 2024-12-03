import argparse

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', default='tokenizer/',
                       help='path to vocab directory')
    parser.add_argument('--train_data', default='train.txt',
                       help='path to training data')    
    parser.add_argument('--valid_data', default='valid.txt',
                       help='path to validation data')  
    parser.add_argument('--output_dir', default='./',
                       help='output directory')    
    parser.add_argument('--n_position', type=int, default=256,
                       help='the number of position')                                     
    parser.add_argument('--n_embd', type=int, default=180,
                       help='the number of embedding')  
    parser.add_argument('--n_layer', type=int, default=12,
                       help='the number of layer')  
    parser.add_argument('--n_head', type=int, default=6, 
                       help='the number of head. It is divisible by n_embd')  
    parser.add_argument('--epochs', type=int, default=400,
                       help='epochs')  
    parser.add_argument('--train_batch', type=int, default=126,
                       help='train batch size')  
    parser.add_argument('--valid_batch', type=int, default=126,
                       help='valid batch size') 
    return parser